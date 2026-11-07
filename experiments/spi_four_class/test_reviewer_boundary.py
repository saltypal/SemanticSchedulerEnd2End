"""Real reviewer orchestration with CPU tensors and scripted model outputs.

These tests establish adapter behavior, not GPU compatibility or VLM accuracy.
"""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from PIL import Image

from experiments.spi_four_class.visual_reliability import assess
from parameters.qwen_reviewer import QwenAnnotationReviewer
from parameters.qwen35_reviewer import Qwen35AnnotationReviewer


class CpuInputs(dict):
    def to(self, device):
        assert device == 'cuda:1'
        return self


class ScriptedProcessor:
    image_processor = SimpleNamespace(patch_size=16)

    def __init__(self, raw, family, missing_pixels=False):
        self.raw, self.family, self.missing_pixels = raw, family, missing_pixels

    def apply_chat_template(self, messages, *, tokenize, enable_thinking, **kwargs):
        assert enable_thinking is False
        if not tokenize:
            suffix = '\n<think>\n\n</think>\n\n' if self.family == 'qwen3_5' else '\n'
            return 'scripted<|im_start|>assistant' + suffix
        inputs = CpuInputs(input_ids=torch.tensor([[1, 2, 3]]))
        if any(item['type'] == 'image' for item in messages[1]['content']) and not self.missing_pixels:
            inputs.update(pixel_values=torch.ones(4, 3), image_grid_thw=torch.tensor([[1, 2, 2]]))
        return inputs

    def decode(self, tokens, **kwargs):
        return self.raw


def reviewer(cls, raw, missing_pixels=False):
    instance = cls('unused-model', 'fixture-revision')
    instance.loaded = True
    instance.torch = torch
    instance.processor = ScriptedProcessor(raw, instance.model_family, missing_pixels)
    instance.model = SimpleNamespace(
        generate=lambda **kwargs: torch.tensor([[1, 2, 3, 4, 5]]),
        generation_config=SimpleNamespace(eos_token_id=5))
    return instance


@pytest.mark.parametrize('cls', [QwenAnnotationReviewer, Qwen35AnnotationReviewer])
@pytest.mark.parametrize('text_only', [True, False])
@pytest.mark.parametrize('raw,level', [
    ('{"action_needed":true,"substantial_consequence":false,"immediate_threat":false,"evidence":"fixture defect"}', 'Medium'),
    ('```json\n{"action_needed":true,"substantial_consequence":true,"immediate_threat":false,"evidence":"fixture damage"}\n```', 'High'),
    ('{"action_needed":null,"substantial_consequence":null,"immediate_threat":null,"evidence":"fixture unclear"}', None),
])
def test_real_runner_to_parser_and_receipts(tmp_path, cls, text_only, raw, level):
    image = tmp_path / 'image.png'
    Image.new('RGB', (32, 32)).save(image)
    with patch.object(torch.cuda, 'synchronize'), patch.object(torch.cuda, 'reset_peak_memory_stats'), \
         patch.object(torch.cuda, 'max_memory_allocated', return_value=0):
        result = assess(reviewer(cls, raw), [] if text_only else [image],
            {'hypothetical_scene': 'fixture'} if text_only else {'instruction': 'fixture'},
            'boundary', tmp_path / 'raw', text_only=text_only)
    assert result['execution']['status'] == 'complete'
    assert result['level'] == level
    receipt = json.loads((tmp_path / 'raw/boundary.request.json').read_text())
    assert receipt['enable_thinking'] is False
    assert bool(receipt['image_inputs']) != text_only
    assert result['execution']['thinking_disabled_verified'] is True
    if text_only:
        assert 'candidate and reference' not in receipt['prefix']
    else:
        assert result['execution']['processed_image_pixels'] == [1024]


@pytest.mark.parametrize('raw,missing_pixels,status', [
    ('not json', False, 'parse_failed'),
    ('{}', True, 'model_failed'),
])
def test_real_runner_preserves_failures(tmp_path, raw, missing_pixels, status):
    image = tmp_path / 'image.png'
    Image.new('RGB', (32, 32)).save(image)
    with patch.object(torch.cuda, 'synchronize'), patch.object(torch.cuda, 'reset_peak_memory_stats'):
        result = assess(reviewer(QwenAnnotationReviewer, raw, missing_pixels), [image],
            {'instruction': 'fixture'}, 'failure', tmp_path / 'raw')
    assert result['execution']['status'] == status
    assert result['level'] is None and not result['available']
    assert (tmp_path / 'raw/failure.failure.json').is_file()
