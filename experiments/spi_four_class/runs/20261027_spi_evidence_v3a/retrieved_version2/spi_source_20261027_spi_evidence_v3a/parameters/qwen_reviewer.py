"""Bounded annotation-assisted Qwen review,explicitly on GPU1."""
from pathlib import Path
from dataclasses import replace,asdict
import hashlib
import json
import time
from .annotation_review import reviewer_input,parse_review,review_system_prompt,REVIEW_PROTOCOL
from .model_settings import ReviewerSettings
from .contracts import digest
from .pilot_producers import file_sha256


def build_review_messages(content,prompt_variant=REVIEW_PROTOCOL):
    """Multimodal ProcessorMixin requires typed content lists for every role."""
    return [{'role':'system','content':[{'type':'text','text':review_system_prompt(prompt_variant)}]},
            {'role':'user','content':content}]


def thinking_template_receipt(processor,messages,family='qwen3_vl'):
    rendered=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    if not isinstance(rendered,str):raise ValueError('review template must render to text')
    tail=rendered.rsplit('<|im_start|>assistant',1)[-1]
    if family=='qwen3_5':
        if not tail.endswith('\n<think>\n\n</think>\n\n'):
            raise RuntimeError('Qwen3.5 non-thinking template suffix not applied')
    elif '<think>' in tail and '</think>' not in tail:
        raise RuntimeError('review template left a thinking block open')
    return {'enable_thinking':False,'thinking_disabled_verified':True,
            'template_sha256':hashlib.sha256(rendered.encode()).hexdigest(),'model_family':family}


def load_review_processor(model_path,settings):
    """Fast Qwen image processors use size, not a late-set max_pixels alias."""
    from transformers import AutoProcessor
    size={'shortest_edge':min(65536,settings.max_pixels),'longest_edge':settings.max_pixels}
    processor=AutoProcessor.from_pretrained(str(model_path),local_files_only=True,size=size)
    if dict(processor.image_processor.size)!=size:raise ValueError('review processor did not apply the requested pixel budget')
    return processor


def pixel_input_receipt(inputs,processor,max_pixels):
    grid=inputs.get('image_grid_thw')
    if grid is None:raise RuntimeError('reviewer image-grid metadata absent')
    patch=int(processor.image_processor.patch_size)
    areas=[int(row[1])*int(row[2])*patch*patch for row in grid.detach().cpu().tolist()]
    if any(area>max_pixels for area in areas):raise RuntimeError('reviewer actual image pixels exceeded configured budget')
    return {'pixel_tensor_shape':list(inputs['pixel_values'].shape),
        'image_grid_thw':grid.detach().cpu().tolist(),'processed_image_pixels':areas,
        'pixel_tensor_sha256':hashlib.sha256(inputs['pixel_values'].detach().cpu().contiguous().numpy().tobytes()).hexdigest()}


class QwenAnnotationReviewer:
    model_family='qwen3_vl'
    def __init__(self,model_path,revision,device='cuda:1',max_new_tokens=None,settings=None):
        if device!='cuda:1':raise ValueError('reviewer must explicitly own GPU1')
        self.model_path,self.revision,self.device=str(model_path),revision,device
        self.settings=settings or ReviewerSettings()
        if max_new_tokens is not None:self.settings=replace(self.settings,max_new_tokens=max_new_tokens)
        self.max_new_tokens=self.settings.max_new_tokens;self.loaded=False

    def load(self):
        if self.loaded:return
        import torch
        from transformers import Qwen3VLForConditionalGeneration
        if torch.cuda.device_count()<2:raise RuntimeError('two assigned GPUs required')
        self.torch=torch;started=time.perf_counter()
        self.processor=load_review_processor(self.model_path,self.settings)
        self.model=Qwen3VLForConditionalGeneration.from_pretrained(self.model_path,local_files_only=True,
            torch_dtype=torch.float16,attn_implementation='sdpa',device_map={'':self.device}).eval()
        devices=sorted({str(p.device) for p in self.model.parameters()})
        if devices!=[self.device]:raise RuntimeError('reviewer placement is not exclusively GPU1: '+str(devices))
        weights={p.name:file_sha256(p) for p in Path(self.model_path).glob('*.safetensors')}
        self.provenance={'model_id':'Qwen/Qwen3-VL-4B-Instruct','revision':self.revision,
            'weights_sha256':weights,'device':self.device,'dtype':'float16','attention':'sdpa',
            'max_image_pixels':self.settings.max_pixels,'max_new_tokens':self.max_new_tokens,
            'do_sample':False,'enable_thinking':False,'load_seconds':time.perf_counter()-started,'parameter_devices':devices,
            'mode':'annotation_assisted_visible_json','not_independent_ground_truth':True}
        self.provenance.update(review_protocol=self.settings.prompt_variant,settings=asdict(self.settings),
            effective_processor_size=dict(self.processor.image_processor.size),
            system_prompt_sha256=digest(review_system_prompt(self.settings.prompt_variant)))
        self.loaded=True

    def configure(self,settings):
        """Change a recorded search setting without reloading the model weights."""
        if not isinstance(settings,ReviewerSettings):raise TypeError('ReviewerSettings required')
        if not self.loaded:
            self.settings=settings;self.max_new_tokens=settings.max_new_tokens;self.load();return
        if settings.max_pixels!=self.settings.max_pixels:
            self.processor=load_review_processor(self.model_path,settings)
        self.settings=settings;self.max_new_tokens=settings.max_new_tokens
        self.provenance.update(max_image_pixels=settings.max_pixels,max_new_tokens=settings.max_new_tokens,
            effective_processor_size=dict(self.processor.image_processor.size),
            review_protocol=settings.prompt_variant,settings=asdict(settings),
            system_prompt_sha256=digest(review_system_prompt(settings.prompt_variant)))

    def review(self,image_paths,candidate,reference,mission,rubric,history,case_id,raw_dir,
               text_only_diagnostic=False):
        if bool(image_paths)==bool(text_only_diagnostic):
            raise ValueError('supply images for visual review, or explicitly request the text-only diagnostic')
        payload=reviewer_input(candidate,reference,mission,rubric,history,claim_scope=self.settings.claim_scope)
        payload['review_protocol']=self.settings.prompt_variant
        return self.review_structured(image_paths,payload,review_system_prompt(self.settings.prompt_variant),
            parse_review,case_id,raw_dir,protocol=self.settings.prompt_variant,
            text_only_diagnostic=text_only_diagnostic)

    def review_structured(self,image_paths,payload,system_prompt,parser,case_id,raw_dir,*,protocol,
                          text_only_diagnostic=False,text_only_prefix=None):
        """Shared bounded inference; caller supplies a versioned contract/parser."""
        if bool(image_paths)==bool(text_only_diagnostic):
            raise ValueError('supply images for visual review, or explicitly request the text-only diagnostic')
        self.load()
        from PIL import Image,ImageOps
        raw_dir=Path(raw_dir);raw_dir.mkdir(parents=True,exist_ok=True)
        content=[];images=[];receipts=[]
        # Current image is first;past image identifiers are logged in history.
        for path in image_paths:
            with Image.open(path) as image:
                picture=image.convert('RGB')
                if self.settings.image_max_side is not None:
                    picture=ImageOps.contain(picture,(self.settings.image_max_side,self.settings.image_max_side))
            images.append(picture);content.append({'type':'image','image':picture})
            receipts.append({'sha256':file_sha256(path),'supplied_size':list(picture.size)})
        if text_only_prefix is not None and (not text_only_diagnostic or not isinstance(text_only_prefix,str)):
            raise ValueError('custom text prefix requires a text-only diagnostic')
        prefix=((text_only_prefix if text_only_prefix is not None else
                 'Text-only protocol diagnostic; no image is supplied. Use the provided candidate and reference.\n')
                if text_only_diagnostic else
                'Image1 is the current observation; subsequent images are declared past views.\n')
        content.append({'type':'text','text':prefix+json.dumps(payload,allow_nan=False)})
        messages=[{'role':'system','content':[{'type':'text','text':system_prompt}]},
                  {'role':'user','content':content}]
        # Preserve the actual model input, not just a nominal prompt identifier.
        # Pixel bytes stay in the source image; receipts identify exact content.
        rendered=self.processor.apply_chat_template(messages,tokenize=False,
            add_generation_prompt=True,enable_thinking=False)
        request_path=raw_dir/(case_id+'.request.json')
        request_path.write_text(json.dumps({'system':system_prompt,'payload':payload,
            'prefix':prefix,'rendered_template':rendered,'image_inputs':receipts,
            'protocol':protocol,'enable_thinking':False},indent=2),encoding='utf-8')
        start=time.perf_counter();raw='';inputs=None;generation={}
        try:
            generation=thinking_template_receipt(self.processor,messages,self.model_family)
            inputs=self.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                        return_dict=True,return_tensors='pt',enable_thinking=False).to(self.device)
            if not text_only_diagnostic and ('pixel_values' not in inputs or inputs['pixel_values'].numel()==0):
                raise RuntimeError('vision pixels absent from reviewer input')
            generation.update({'text_only_diagnostic':True} if text_only_diagnostic else
                              pixel_input_receipt(inputs,self.processor,self.settings.max_pixels))
            self.torch.cuda.synchronize(1);self.torch.cuda.reset_peak_memory_stats(1)
            # Only the final input position is needed for next-token generation.
            # Avoid materializing a vocabulary-sized output for every JSON token.
            generation_kwargs={'max_new_tokens':self.max_new_tokens,'do_sample':False,'logits_to_keep':1}
            generation.update(input_tokens=int(inputs['input_ids'].shape[1]),logits_to_keep=1)
            wall_cap=getattr(self,'maximum_generation_seconds',None)
            if wall_cap is not None:generation_kwargs['max_time']=wall_cap
            if inputs['input_ids'].shape[1]>getattr(self,'maximum_input_tokens',8192):
                raise RuntimeError('review input exceeds declared token budget; no silent truncation')
            if getattr(self,'require_finite_preflight',False) and not getattr(self,'_finite_preflight_passed',False):
                with self.torch.inference_mode():
                    probe=self.model(**inputs,use_cache=False,logits_to_keep=1)
                if not self.torch.isfinite(probe.logits).all().item():
                    raise RuntimeError('reviewer preflight produced nonfinite last-token logits')
                del probe
                self._finite_preflight_passed=True
            generation['finite_preflight_passed']=getattr(self,'_finite_preflight_passed',None)
            generated_at=time.perf_counter()
            with self.torch.inference_mode():
                generated=self.model.generate(**inputs,**generation_kwargs)
            self.torch.cuda.synchronize(1)
            generation.update(generation_seconds=time.perf_counter()-generated_at,generation_wall_cap_seconds=wall_cap)
            tokens=generated[0,inputs['input_ids'].shape[1]:]
            eos=self.model.generation_config.eos_token_id
            eos=[] if eos is None else [eos] if isinstance(eos,int) else eos
            generation.update(generated_tokens=len(tokens),input_tokens=int(inputs['input_ids'].shape[1]),
                hit_token_limit=len(tokens)>=self.max_new_tokens and (not len(tokens) or int(tokens[-1]) not in eos))
            raw=self.processor.decode(tokens,skip_special_tokens=True).strip()
            raw_path=raw_dir/(case_id+'.txt');raw_path.write_text(raw,encoding='utf-8')
            if '<think>' in raw or '</think>' in raw:raise ValueError('unexpected generated thinking content')
            parsed,recovered=parser(raw)
            return {'case_id':case_id,'status':'complete','parsed':parsed,'raw_response_ref':str(raw_path),
                    'parse_recovery_used':recovered,'wall_seconds':time.perf_counter()-start,
                    **generation,'image_inputs':receipts,'request_ref':str(request_path),
                    'prompt_sha256':digest({'system':system_prompt,'user':payload}),
                    'review_protocol':protocol,'peak_gpu1_bytes':self.torch.cuda.max_memory_allocated(1),
                    'assessment_kind':('text_only_protocol_diagnostic' if text_only_diagnostic else
                                       'annotation_assisted_model_review'),'model_revision':self.revision}
        except Exception as error:
            path=raw_dir/(case_id+'.failure.json')
            path.write_text(json.dumps({'error':type(error).__name__,'message':str(error),'raw':raw},indent=2))
            return {'case_id':case_id,'status':'parse_failed' if raw else 'model_failed',
                    'error':type(error).__name__,'message':str(error),'raw_response_ref':str(path),
                    'wall_seconds':time.perf_counter()-start,'image_inputs':receipts,
                    'request_ref':str(request_path),**generation,
                    'review_protocol':protocol}
        finally:
            del inputs

    def close(self):
        if hasattr(self,'model'):del self.model
        if hasattr(self,'processor'):del self.processor
        import gc
        gc.collect()
        if hasattr(self,'torch'):self.torch.cuda.empty_cache()
