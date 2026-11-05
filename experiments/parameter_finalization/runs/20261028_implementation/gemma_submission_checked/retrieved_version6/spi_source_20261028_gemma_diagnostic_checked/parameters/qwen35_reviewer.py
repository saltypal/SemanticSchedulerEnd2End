"""Qwen3.5-9B challenger; requires its isolated Transformers 5.3 runtime."""
from pathlib import Path
from dataclasses import asdict
import importlib.util
import time
from .qwen_reviewer import QwenAnnotationReviewer,load_review_processor
from .pilot_producers import file_sha256
from .annotation_review import review_system_prompt
from .contracts import digest

MODEL_ID='Qwen/Qwen3.5-9B'
MODEL_REVISION='c202236235762e1c871ad0ccb60c8ee5ba337b9a'


class Qwen35AnnotationReviewer(QwenAnnotationReviewer):
    model_family='qwen3_5'

    def load(self):
        if self.loaded:return
        import torch,transformers
        if torch.cuda.device_count()!=2:raise RuntimeError('two T4 devices required before 9B loading')
        optional={name:importlib.util.find_spec(name) is not None for name in ('fla','causal_conv1d')}
        if any(optional.values()):
            raise RuntimeError('unverified optional DeltaNet kernels inherited: '+str(optional))
        from transformers import Qwen3_5ForConditionalGeneration,BitsAndBytesConfig
        torch.cuda.set_device(1)
        if torch.cuda.get_device_capability(1)<(6,0):raise RuntimeError('GPU does not meet NF4 requirements')
        self.torch=torch;started=time.perf_counter()
        self.processor=load_review_processor(self.model_path,self.settings)
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
        self.model=Qwen3_5ForConditionalGeneration.from_pretrained(self.model_path,local_files_only=True,
            dtype=torch.float16,attn_implementation='sdpa',device_map={'':1},quantization_config=quant).eval()
        from bitsandbytes.nn import Linear4bit
        quantized=sum(isinstance(module,Linear4bit) for module in self.model.modules())
        if not quantized:raise RuntimeError('9B load did not activate NF4 quantization')
        devices=sorted({str(p.device) for p in self.model.parameters()})
        if devices!=[self.device]:raise RuntimeError('9B model spilled outside reviewer GPU1: '+str(devices))
        self.provenance={'model_id':MODEL_ID,'revision':self.revision,
            'weights_sha256':{p.name:file_sha256(p) for p in Path(self.model_path).glob('*.safetensors')},
            'device':self.device,'dtype':'float16','quantization':'bitsandbytes_nf4_double_quant_fp16_compute',
            'attention':'sdpa','optional_kernel_presence':optional,'delta_backend':'built_in_torch',
            'parameter_devices':devices,'load_seconds':time.perf_counter()-started,
            'model_footprint_bytes':self.model.get_memory_footprint(),
            'linear4bit_modules':quantized,
            'transformers':transformers.__version__,'enable_thinking':False,'do_sample':False,
            'mode':'annotation_assisted_visible_json','not_independent_ground_truth':True,
            'settings':asdict(self.settings),'review_protocol':self.settings.prompt_variant,
            'max_image_pixels':self.settings.max_pixels,'max_new_tokens':self.max_new_tokens,
            'effective_processor_size':dict(self.processor.image_processor.size),
            'system_prompt_sha256':digest(review_system_prompt(self.settings.prompt_variant))}
        self.loaded=True
