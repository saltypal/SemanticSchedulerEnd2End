"""Gemma orchestration references; scripted CPU model, no claim of GPU inference."""
from types import SimpleNamespace
from unittest.mock import patch
import torch
from PIL import Image
from parameters.gemma_spi_reviewer import GemmaSPIReviewer
from parameters.spi_protocol import parse_evidence
from experiments.spi_four_class.test_reviewer_boundary import ScriptedProcessor


class Processor(ScriptedProcessor):
    def apply_chat_template(self, messages, *, tokenize, enable_thinking, **kwargs):
        if not tokenize:
            assert enable_thinking is False
            return '<|turn>model\n'
        return super().apply_chat_template(messages, tokenize=tokenize,
            enable_thinking=enable_thinking, **kwargs)


class Model:
    def __init__(self):self.probes=0
    def __call__(self, **kwargs):
        self.probes+=1
        return SimpleNamespace(logits=torch.ones(1,1,5))
    def generate(self, **kwargs):
        assert kwargs['do_sample'] is False
        return torch.tensor([[1,2,3,4,5]])


def test_text_and_image_have_independent_finite_preflights(tmp_path):
    reviewer=GemmaSPIReviewer()
    reviewer.model=Model();reviewer.torch=torch;reviewer.finite_checked_modes=set()
    reviewer.processor=Processor('{"action_needed":false,"substantial_consequence":false,"immediate_threat":false,"evidence":"fixture only"}', 'gemma4')
    image=tmp_path/'image.png';Image.new('RGB',(32,32)).save(image)
    with patch.object(torch.cuda,'synchronize'),patch.object(torch.cuda,'reset_peak_memory_stats'),patch.object(torch.cuda,'max_memory_allocated',return_value=0):
        for i,text in enumerate([True,False,True]):
            result=reviewer.review_structured([] if text else [image],{'instruction':'fixture'},'fixture',
                parse_evidence,str(i),tmp_path/'raw',protocol='fixture',text_only_diagnostic=text)
            assert result['status']=='complete'
            assert result['parsed']['level']=='Low'
            assert (tmp_path/'raw'/f'{i}.request.json').exists()
    assert reviewer.model.probes==2
    assert reviewer.finite_checked_modes=={'text','image'}
