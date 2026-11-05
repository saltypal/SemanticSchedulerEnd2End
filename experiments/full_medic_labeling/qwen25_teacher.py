"""Pinned Qwen2.5VL7B teacher; no reasoning-mode generation."""
import hashlib,time
class BulkTeacher:
    def __init__(self,path,device,enable_thinking=False,system_prompt=None,parser=None):
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor,BitsAndBytesConfig
        if enable_thinking:raise ValueError('Qwen2.5 adapter does not enable reasoning mode')
        self.device=int(device);self.system_prompt=system_prompt;self.parser=parser
        if torch.cuda.device_count()!=2 or self.device!=1:raise RuntimeError('two assigned GPUs required, teacher onGPU1')
        self.processor=AutoProcessor.from_pretrained(path,local_files_only=True,min_pixels=65536,max_pixels=262144)
        q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
        self.model=Qwen2_5_VLForConditionalGeneration.from_pretrained(path,local_files_only=True,dtype=torch.float16,
            attn_implementation='sdpa',device_map={'':self.device},quantization_config=q).eval()
        if {str(p.device) for p in self.model.parameters()}!={'cuda:1'}:raise RuntimeError('teacher device spill')
        self.checked=False
    def review(self,image):
        import torch
        from PIL import ImageOps
        start=time.perf_counter();raw='';receipt={'enable_thinking':False,'model_family':'qwen2_5_vl','device':1}
        try:
            image=ImageOps.exif_transpose(image).convert('RGB')
            messages=[{'role':'system','content':[{'type':'text','text':self.system_prompt}]},
                {'role':'user','content':[{'type':'image','image':image},{'type':'text','text':'Assess this image.'}]}]
            template=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            receipt['template_sha256']=hashlib.sha256(template.encode()).hexdigest()
            inputs=self.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to('cuda:1')
            receipt.update(input_tokens=int(inputs['input_ids'].shape[1]),image_grid_thw=inputs['image_grid_thw'].tolist(),pixel_tensor_shape=list(inputs['pixel_values'].shape))
            if inputs['input_ids'].shape[1]>4096:raise ValueError('input token cap exceeded')
            with torch.inference_mode():
                if not self.checked:
                    probe=self.model(**inputs,use_cache=False,logits_to_keep=1)
                    if not torch.isfinite(probe.logits).all():raise ValueError('nonfinite model preflight')
                    del probe;self.checked=True
                torch.cuda.synchronize(1);torch.cuda.reset_peak_memory_stats(1);began=time.perf_counter()
                generated=self.model.generate(**inputs,max_new_tokens=384,do_sample=False,max_time=45)
                torch.cuda.synchronize(1)
            tokens=generated[0,inputs['input_ids'].shape[1]:];raw=self.processor.decode(tokens,skip_special_tokens=True).strip()
            receipt.update(generation_seconds=time.perf_counter()-began,generated_tokens=len(tokens),peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(1))
            parsed,recovered=self.parser(raw)
            return {**parsed,'status':'complete','raw_response':raw,'execution':receipt,'format_recovery':recovered,'wall_seconds':time.perf_counter()-start}
        except Exception as error:
            return {'available':False,'status':'failed','reason':str(error),'error':type(error).__name__,'raw_response':raw,'execution':receipt,'wall_seconds':time.perf_counter()-start}
