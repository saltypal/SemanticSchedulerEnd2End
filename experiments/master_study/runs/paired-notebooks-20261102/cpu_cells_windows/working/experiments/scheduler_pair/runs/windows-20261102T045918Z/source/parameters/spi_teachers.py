"""Bounded teacher registry; all models must stay on GPU1 with thinking off."""
from pathlib import Path
import gc
import time
from .qwen35_reviewer import Qwen35AnnotationReviewer, MODEL_REVISION
from .qwen_reviewer import QwenAnnotationReviewer, load_review_processor
from .model_settings import ReviewerSettings

TEACHERS = (
    {'name': 'qwen35_9b', 'model': 'Qwen/Qwen3.5-9B', 'revision': MODEL_REVISION,
     'family': 'qwen3_5'},
    {'name': 'qwen3vl_8b', 'model': 'Qwen/Qwen3-VL-8B-Instruct', 'revision': 'main',
     'family': 'qwen3_vl'},
    {'name': 'gemma4_e4b', 'model': 'google/gemma-4-E4B-it', 'revision': 'main',
     'family': 'gemma4'},
)


class UnsupportedTeacher(RuntimeError):
    """A pinned runtime cannot faithfully run this candidate; never auto-upgrade."""


class QuantizedQwen3Reviewer(QwenAnnotationReviewer):
    def load(self):
        if self.loaded:
            return
        import torch
        from transformers import Qwen3VLForConditionalGeneration, BitsAndBytesConfig
        if torch.cuda.device_count() != 2:
            raise RuntimeError('two assigned GPUs required by the experiment contract')
        self.torch = torch
        start = time.perf_counter()
        self.processor = load_review_processor(self.model_path, self.settings)
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_path,
            local_files_only=True, dtype=torch.float16, attn_implementation='sdpa',
            device_map={'': 1}, quantization_config=quant).eval()
        devices = sorted({str(p.device) for p in self.model.parameters()})
        if devices != ['cuda:1']:
            raise RuntimeError('teacher spilled outside GPU1: ' + str(devices))
        self.provenance = {'model_id': 'Qwen/Qwen3-VL-8B-Instruct',
            'revision': self.revision, 'quantization': 'NF4_double_quant_FP16_compute',
            'enable_thinking': False, 'do_sample': False, 'device': self.device,
            'load_seconds': time.perf_counter() - start,
            'footprint_bytes': self.model.get_memory_footprint(), 'parameter_devices': devices}
        self.loaded = True


def load_teacher(spec):
    """Resolve main to immutable revision before snapshot/download; record it."""
    from huggingface_hub import HfApi, snapshot_download
    if spec['family'] == 'gemma4':
        # Its processor and thinking/pixel receipts require a separately verified
        # runtime. Do not silently swap the pinned Qwen environment mid-campaign.
        raise UnsupportedTeacher('Gemma4 challenger deferred: separate verified processor/runtime required; no weights downloaded')
    revision = spec['revision']
    if revision == 'main':
        revision = HfApi().model_info(spec['model'], revision='main').sha
    print('Loading teacher', spec['name'], 'revision', revision, flush=True)
    path = snapshot_download(spec['model'], revision=revision,
        allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])
    settings = ReviewerSettings(max_new_tokens=240, max_pixels=262144,
        image_max_side=None, prompt_variant='semantic_contract_v5')
    cls = Qwen35AnnotationReviewer if spec['family'] == 'qwen3_5' else QuantizedQwen3Reviewer
    reviewer = cls(Path(path), revision, settings=settings)
    reviewer.maximum_generation_seconds = 35
    reviewer.maximum_input_tokens = 4096
    reviewer.require_finite_preflight = True
    try:
        reviewer.load()
    except Exception:
        reviewer.close()
        gc.collect()
        raise
    return reviewer
