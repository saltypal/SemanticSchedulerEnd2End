"""Offline VLM exclusively on GPU1; GPU0 reserved for visual feature extraction."""
from dataclasses import asdict
import hashlib
import importlib.util
from pathlib import Path
import time

from experiments.spi_classifier_repair.evidence_priority import PROMPT as BASE_PROMPT, parse_evidence
from parameters.model_settings import ReviewerSettings
from parameters.qwen_reviewer import load_review_processor, thinking_template_receipt, pixel_input_receipt
from parameters.qwen35_reviewer import MODEL_ID, MODEL_REVISION

SYSTEM_PROMPT = BASE_PROMPT + "\nBe concise: use at most two observations, each under20 words. Return JSON only."
PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
USER_TEXT = "Assess this image."


class BulkTeacher:
    def __init__(self, path, device, enable_thinking=False, system_prompt=None, parser=None):
        import torch
        from transformers import Qwen3_5ForConditionalGeneration, BitsAndBytesConfig
        from bitsandbytes.nn import Linear4bit
        self.enable_thinking=bool(enable_thinking)
        self.system_prompt=system_prompt or SYSTEM_PROMPT
        self.parser=parser or parse_evidence
        self.device = int(device)
        if torch.cuda.device_count() != 2 or self.device != 1:
            raise RuntimeError("two assigned GPUs required")
        if any(importlib.util.find_spec(n) is not None for n in ('fla','causal_conv1d')):
            raise RuntimeError("unverified optional DeltaNet kernels present")
        self.settings = ReviewerSettings(max_new_tokens=192, max_pixels=262144,
                                        image_max_side=None, prompt_variant='semantic_contract_v5')
        with torch.cuda.device(self.device):
            self.processor = load_review_processor(path, self.settings)
            q = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                                  bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                path, local_files_only=True, dtype=torch.float16, attn_implementation='sdpa',
                device_map={'':self.device}, quantization_config=q).eval()
        if {str(p.device) for p in self.model.parameters()} != {f'cuda:{self.device}'}:
            raise RuntimeError("teacher device spill")
        if not any(isinstance(m,Linear4bit) for m in self.model.modules()):
            raise RuntimeError("NF4 not active")
        self.checked = False

    def review(self, image):
        import torch
        from PIL import ImageOps
        start = time.perf_counter()
        picture = ImageOps.exif_transpose(image).convert('RGB')
        messages = [{'role':'system','content':[{'type':'text','text':self.system_prompt}]},
                    {'role':'user','content':[{'type':'image','image':picture},
                                              {'type':'text','text':USER_TEXT}]}]
        if self.enable_thinking:
            rendered=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=True)
            receipt={'enable_thinking':True,'template_sha256':hashlib.sha256(rendered.encode()).hexdigest()}
        else:
            receipt = thinking_template_receipt(self.processor,messages,'qwen3_5')
        inputs = None
        try:
            with torch.cuda.device(self.device):
                inputs = self.processor.apply_chat_template(messages,tokenize=True,
                    add_generation_prompt=True,return_dict=True,return_tensors='pt',enable_thinking=self.enable_thinking).to(f'cuda:{self.device}')
                receipt.update(pixel_input_receipt(inputs,self.processor,self.settings.max_pixels))
                if inputs['input_ids'].shape[1] > 4096:
                    raise ValueError("input token budget exceeded")
                if not self.checked:
                    with torch.inference_mode():
                        probe=self.model(**inputs,use_cache=False,logits_to_keep=1)
                    if not torch.isfinite(probe.logits).all().item():
                        raise ValueError("nonfinite visual preflight")
                    del probe
                    self.checked=True
                torch.cuda.synchronize(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)
                began=time.perf_counter()
                with torch.inference_mode():
                    generated=self.model.generate(**inputs,max_new_tokens=2048 if self.enable_thinking else 192,do_sample=False,
                                                  logits_to_keep=1,max_time=120 if self.enable_thinking else 45)
                torch.cuda.synchronize(self.device)
                tokens=generated[0,inputs['input_ids'].shape[1]:]
                raw=self.processor.decode(tokens,skip_special_tokens=True).strip()
                receipt.update(generation_seconds=time.perf_counter()-began,
                    input_tokens=int(inputs['input_ids'].shape[1]),generated_tokens=int(len(tokens)),
                    peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                    device=self.device,finite_preflight_passed=True)
            if not self.enable_thinking and ('<think>' in raw or '</think>' in raw):
                raise ValueError("generated thinking despite disabled template")
            parsed,recovered=self.parser(raw)
            return {**parsed,'status':'complete','raw_response':raw,'execution':receipt,
                    'format_recovery':recovered,'wall_seconds':time.perf_counter()-start}
        except Exception as error:
            return {'level':None,'available':False,'status':'failed','error':type(error).__name__,
                    'reason':str(error),'raw_response':locals().get('raw',''),
                    'execution':receipt,'wall_seconds':time.perf_counter()-start}
        finally:
            del inputs
