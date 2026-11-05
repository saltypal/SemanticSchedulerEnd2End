"""Gemma-specific offline SPI reviewer. No Qwen template or pixel-grid assumptions."""
import hashlib
import json
from pathlib import Path
import time

MODEL_ID = 'google/gemma-4-E4B-it'


class GemmaSPIReviewer:
    def __init__(self):
        self.model = None
        self.provenance = {}

    def load(self):
        if self.model is not None:
            return
        import torch
        import transformers
        from transformers import AutoProcessor, Gemma4ForConditionalGeneration, BitsAndBytesConfig
        from huggingface_hub import HfApi, snapshot_download
        if transformers.__version__ != '5.5.0':
            raise RuntimeError('Gemma SPI requires the independently recorded Transformers 5.5.0 runtime')
        if torch.cuda.device_count() != 2:
            raise RuntimeError('two assigned GPUs required; reviewer owns GPU1')
        revision = HfApi().model_info(MODEL_ID).sha
        path = snapshot_download(MODEL_ID, revision=revision,
            allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.jinja'])
        self.torch = torch
        start = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        self.model = Gemma4ForConditionalGeneration.from_pretrained(path,
            local_files_only=True, dtype=torch.float16, device_map={'': 1},
            quantization_config=quant, attn_implementation='sdpa').eval()
        devices = sorted({str(p.device) for p in self.model.parameters()})
        if devices != ['cuda:1']:
            raise RuntimeError('Gemma spilled outside GPU1: ' + str(devices))
        self.provenance = {'model_id': MODEL_ID, 'revision': revision,
            'transformers': transformers.__version__, 'quantization': 'NF4_double_quant_FP16',
            'parameter_devices': devices, 'enable_thinking': False,
            'load_seconds': time.perf_counter()-start,
            'model_footprint_bytes': self.model.get_memory_footprint(),
            'mode': 'image_only_spi_diagnostic', 'independent_ground_truth': False}
        self.finite_checked_modes = set()

    def review_structured(self, image_paths, payload, system_prompt, parser, case_id,
                          raw_dir, *, protocol, text_only_diagnostic=False,
                          text_only_prefix=None):
        if bool(image_paths) == bool(text_only_diagnostic):
            raise ValueError('explicit text-only mode or supplied image required')
        self.load()
        from PIL import Image, ImageOps
        output = Path(raw_dir)
        output.mkdir(parents=True, exist_ok=True)
        content, receipts = [], []
        for path in image_paths:
            with Image.open(path) as original:
                picture = ImageOps.exif_transpose(original).convert('RGB')
            content.append({'type': 'image', 'image': picture})
            receipts.append({'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                             'supplied_size': list(picture.size)})
        content.append({'type': 'text', 'text': (text_only_prefix or '') + json.dumps(payload)})
        messages = [{'role': 'system', 'content': [{'type': 'text', 'text': system_prompt}]},
                    {'role': 'user', 'content': content}]
        request_path = output / (case_id + '.request.json')
        start = time.perf_counter()
        raw = ''
        special = ''
        inputs = None
        receipt = {'image_inputs': receipts, 'enable_thinking': False, 'protocol': protocol,
                   'system': system_prompt, 'payload': payload}
        try:
            rendered = self.processor.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            if '<|think|>' in rendered or not rendered.endswith('<|turn>model\n'):
                raise RuntimeError('Gemma thinking remained enabled in rendered template')
            receipt['rendered_template'] = rendered
            inputs = self.processor.apply_chat_template(messages, tokenize=True,
                add_generation_prompt=True, enable_thinking=False,
                return_dict=True, return_tensors='pt').to('cuda:1')
            receipt['tensor_shapes'] = {k: list(v.shape) for k,v in inputs.items()
                                        if self.torch.is_tensor(v)}
            pixels = inputs.get('pixel_values')
            if image_paths and (pixels is None or not pixels.numel()):
                raise RuntimeError('Gemma image pixels absent')
            if pixels is not None:
                receipt['pixel_tensor_sha256'] = hashlib.sha256(
                    pixels.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()
            if inputs['input_ids'].shape[1] > 4096:
                raise RuntimeError('Gemma input exceeds 4096-token budget')
            request_path.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
            mode = 'text' if text_only_diagnostic else 'image'
            if mode not in self.finite_checked_modes:
                with self.torch.inference_mode():
                    probe = self.model(**inputs, use_cache=False, logits_to_keep=1)
                if not self.torch.isfinite(probe.logits).all().item():
                    raise RuntimeError('Gemma nonfinite preflight logits')
                del probe
                self.finite_checked_modes.add(mode)
            self.torch.cuda.synchronize(1)
            self.torch.cuda.reset_peak_memory_stats(1)
            with self.torch.inference_mode():
                generated = self.model.generate(**inputs, max_new_tokens=240,
                    do_sample=False, max_time=35, logits_to_keep=1)
            self.torch.cuda.synchronize(1)
            tokens = generated[0, inputs['input_ids'].shape[1]:]
            special = self.processor.decode(tokens, skip_special_tokens=False)
            (output / (case_id + '.special.txt')).write_text(special, encoding='utf-8')
            if '<|channel>thought' in special or '<|think|>' in special:
                raise RuntimeError('Gemma generated unexpected thinking content')
            raw = self.processor.decode(tokens, skip_special_tokens=True).strip()
            (output / (case_id + '.txt')).write_text(raw, encoding='utf-8')
            parsed, recovered = parser(raw)
            return {'status': 'complete', 'parsed': parsed, 'image_inputs': receipts,
                'request_ref': str(request_path), 'wall_seconds': time.perf_counter()-start,
                'parse_recovery_used': recovered, 'thinking_disabled_verified': True,
                'generated_tokens': len(tokens), 'enable_thinking': False,
                'peak_gpu1_bytes': self.torch.cuda.max_memory_allocated(1)}
        except Exception as error:
            request_path.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
            failure = output / (case_id + '.failure.json')
            failure.write_text(json.dumps({'error':type(error).__name__, 'message':str(error),
                                          'raw':raw, 'special_output':special}), encoding='utf-8')
            return {'status':'parse_failed' if raw else 'model_failed',
                'message':str(error), 'image_inputs':receipts, 'request_ref':str(request_path),
                'wall_seconds':time.perf_counter()-start}
        finally:
            del inputs

    def close(self):
        self.model = None
        if hasattr(self, 'processor'):
            del self.processor
        if hasattr(self, 'torch'):
            self.torch.cuda.empty_cache()


def load_gemma(spec):
    reviewer = GemmaSPIReviewer()
    try:
        reviewer.load()
    except Exception:
        reviewer.close()
        raise
    return reviewer
