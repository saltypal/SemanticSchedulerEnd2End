"""Frozen shared image features, with tensor-only dual-GPU forward wrappers."""
from pathlib import Path
import hashlib
import time
import numpy as np

from .contracts import MISSIONS, digest
from .extractor import VisualEvidence

SCENE_PROMPTS = {
    'road': 'a photograph of a road or street', 'residential': 'a photograph of residential buildings',
    'indoor': 'a photograph of an indoor scene', 'forest': 'a photograph of a forest',
    'water': 'a photograph of a river or flooded area', 'people': 'a photograph containing people',
    'vehicle': 'a photograph containing vehicles', 'fire': 'a photograph of a fire',
}
MODEL_IDS = {'clip': 'openai/clip-vit-base-patch32', 'siglip': 'google/siglip-base-patch16-224'}


class FrozenVisionProvider:
    def __init__(self, backbone='clip', model_path=None, device='cpu', data_parallel=False,
                 batch_size=8, local_files_only=True,preprocessing='aspect_preserving_pad'):
        if backbone not in MODEL_IDS or batch_size < 1:
            raise ValueError('unsupported backbone or batch size')
        self.backbone, self.model_path = backbone, str(model_path or MODEL_IDS[backbone])
        self.device, self.data_parallel, self.batch_size = device, data_parallel, batch_size
        self.local_files_only = local_files_only
        if preprocessing not in ('aspect_preserving_pad','checkpoint_native'):raise ValueError('unknown visual preprocessing')
        self.preprocessing=preprocessing
        self.loaded = False
        self.timings = []
        self.device_ownership = []
        self.text_cache = {}

    def load(self):
        if self.loaded:
            return
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=self.local_files_only)
        self.model = AutoModel.from_pretrained(self.model_path, local_files_only=self.local_files_only).to(self.device).eval()
        class ImageForward(torch.nn.Module):
            def __init__(self, model):
                super().__init__(); self.model = model
            def forward(self, pixel_values):
                feature = self.model.get_image_features(pixel_values=pixel_values)
                if not torch.is_tensor(feature):
                    feature = feature.pooler_output
                feature = torch.nn.functional.normalize(feature.float(), dim=-1)
                owner = torch.full((feature.shape[0],), pixel_values.device.index if pixel_values.is_cuda else -1,
                                   dtype=torch.int64, device=pixel_values.device)
                return feature, owner
        self.forward = ImageForward(self.model).eval()
        if self.data_parallel:
            if not str(self.device).startswith('cuda') or torch.cuda.device_count() < 2:
                raise RuntimeError('requested DataParallel requires two assigned GPUs')
            self.forward = torch.nn.DataParallel(self.forward, device_ids=[0,1]).eval()
        config_path = Path(self.model_path)/'config.json'
        weights_path=next((Path(self.model_path)/name for name in ('model.safetensors','pytorch_model.bin') if (Path(self.model_path)/name).exists()),None)
        weights_sha=None
        if weights_path is not None:
            h=hashlib.sha256()
            with weights_path.open('rb') as handle:
                for chunk in iter(lambda:handle.read(1024*1024),b''):h.update(chunk)
            weights_sha=h.hexdigest()
        self.provenance = dict(backbone=self.backbone, model_id=MODEL_IDS[self.backbone],
                               model_path=self.model_path, config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest() if config_path.exists() else digest(self.model.config.to_dict()),
                               producer_scope='frozen pretrained embeddings; prompt similarities are not probabilities',
                               weights_sha256=weights_sha,
                               preprocessing=self.preprocessing,processor_sha256=digest(self.processor.image_processor.to_dict()), device=self.device,
                               data_parallel_requested=self.data_parallel, model_load_seconds=time.perf_counter()-started)
        self.provenance.update(
            image_processor=self.processor.image_processor.to_dict(),
            scene_prompts=dict(SCENE_PROMPTS),
            model_dtype=str(next(self.model.parameters()).dtype),
            embedding_dimension=getattr(self.model.config,'projection_dim',None),
            image_size=int(self.model.config.vision_config.image_size),
            text_max_positions=int(getattr(self.model.config.text_config,'max_position_embeddings',77 if self.backbone=='clip' else 64)),
            text_padding='max_length' if self.backbone=='siglip' else 'longest_in_batch',
            text_truncation=False, image_feature_normalization='L2', text_feature_normalization='L2',
            similarity='normalized image/text dot product; no logit scale or softmax',
            inference_mode=True, batch_size=self.batch_size)
        self.loaded = True

    @staticmethod
    def _pad(image, size=224):
        from PIL import Image
        image = image.convert('RGB')
        scale = min(size/image.width, size/image.height)
        resized = image.resize((max(1,round(image.width*scale)),max(1,round(image.height*scale))),Image.Resampling.BICUBIC)
        canvas=Image.new('RGB',(size,size),(127,127,127))
        canvas.paste(resized,((size-resized.width)//2,(size-resized.height)//2))
        return canvas

    def _texts(self, prompts):
        self.load()
        missing = [p for p in dict.fromkeys(prompts) if p not in self.text_cache]
        if missing:
            kwargs = dict(text=missing, padding='max_length' if self.backbone=='siglip' else True,
                          truncation=False, return_tensors='pt')
            inputs = self.processor(**kwargs).to(self.device)
            maximum = int(getattr(self.model.config.text_config,'max_position_embeddings',77 if self.backbone=='clip' else 64))
            if inputs['input_ids'].shape[1] > maximum:
                raise ValueError('observable clause exceeds model text context; split it explicitly')
            with self.torch.inference_mode():
                feature = self.model.get_text_features(**inputs)
                if not self.torch.is_tensor(feature): feature=feature.pooler_output
                feature=self.torch.nn.functional.normalize(feature.float(),dim=-1).cpu().numpy()
            self.text_cache.update(zip(missing,feature))
        return np.stack([self.text_cache[p] for p in prompts])

    def image_inputs(self,images):
        if self.preprocessing=='checkpoint_native':
            return self.processor(images=[image.convert('RGB') for image in images],return_tensors='pt')
        size=int(getattr(self.model.config.vision_config,'image_size',224))
        return self.processor(images=[self._pad(image,size) for image in images],
            do_resize=False,do_center_crop=False,return_tensors='pt')

    def extract_batch(self, images, missions):
        self.load()
        if len(images)!=len(missions): raise ValueError('images and missions differ')
        if not images: return []
        claims=dict(c for mission in list(MISSIONS.values())+list(missions) for c in mission.conditions)
        text_map={**claims,**{'scene_'+k:v for k,v in SCENE_PROMPTS.items()}}
        names=list(text_map); text_features=self._texts(list(text_map.values()))
        output=[]
        for start in range(0,len(images),self.batch_size):
            batch=images[start:start+self.batch_size]
            inputs=self.image_inputs(batch)
            pixels=inputs['pixel_values'].to(self.device)
            if str(self.device).startswith('cuda'): self.torch.cuda.synchronize(self.device)
            began=time.perf_counter()
            with self.torch.inference_mode(): features,owners=self.forward(pixels)
            if str(self.device).startswith('cuda'):
                if self.data_parallel:
                    for i in self.forward.device_ids:self.torch.cuda.synchronize(i)
                else:self.torch.cuda.synchronize(self.device)
            elapsed=time.perf_counter()-began
            features=features.cpu().numpy(); owners=owners.cpu().tolist()
            self.device_ownership.extend(owners)
            similarities=features@text_features.T
            self.timings.append({'batch_size':len(batch),'forward_seconds':elapsed,'device_owners':owners,
                                 'pixel_tensor_shape':list(pixels.shape)})
            for j,(image,vector) in enumerate(zip(batch,features)):
                scores=dict(zip(names,map(float,similarities[j])))
                output.append(VisualEvidence(embedding=vector,
                    similarities={k:v for k,v in scores.items() if not k.startswith('scene_')},
                    scene_scores={k[6:]:float(np.clip((v+1)/2,0,1)) for k,v in scores.items() if k.startswith('scene_')},
                    width=image.width,height=image.height,statuses={'encoder':'ok','detector':'not_requested'},provenance=self.provenance))
        return output

    def verify_parallel(self, images, mission):
        if len(images)<4: raise ValueError('parallel preflight needs at least four images')
        self.load()
        pixels=self.image_inputs(images)['pixel_values'].to(self.device)
        with self.torch.inference_mode():
            parallel,owners=self.forward(pixels)
            single,_=self.forward.module(pixels)
        difference=float((parallel-single).abs().max().cpu())
        used=sorted(set(owners.cpu().tolist()))
        if used != [0,1] or not self.torch.allclose(parallel,single,rtol=1e-4,atol=1e-5):
            raise RuntimeError('DataParallel ownership or numerical equivalence failed')
        return dict(max_absolute_difference=difference,device_owners=owners.cpu().tolist(),both_gpus_used=True)

    def close(self):
        import gc
        for name in ('forward','model','processor'):
            if hasattr(self,name): delattr(self,name)
        self.loaded=False;gc.collect()
        if hasattr(self,'torch') and self.torch.cuda.is_available(): self.torch.cuda.empty_cache()


class GroundedEvidenceProvider:
    """One optional detector; outputs do not certify distress or urgency."""
    def __init__(self, model_path, device='cpu', local_files_only=True):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self.torch,self.device=torch,device
        self.processor=AutoProcessor.from_pretrained(str(model_path),local_files_only=local_files_only)
        self.model=AutoModelForZeroShotObjectDetection.from_pretrained(str(model_path),local_files_only=local_files_only).to(device).eval()

    def add(self,image,evidence,entities=('building','road','person','vehicle')):
        import inspect
        inputs=self.processor(images=image.convert('RGB'),text='. '.join(entities)+'.',return_tensors='pt').to(self.device)
        with self.torch.inference_mode(): output=self.model(**inputs)
        post=self.processor.post_process_grounded_object_detection
        key='box_threshold' if 'box_threshold' in inspect.signature(post).parameters else 'threshold'
        result=post(output,inputs.input_ids,text_threshold=.25,target_sizes=[(image.height,image.width)],**{key:.35})[0]
        labels=result.get('text_labels',result.get('labels'))
        evidence.detections=tuple({'label':str(label).strip().rstrip('.'),'box':box.cpu().tolist(),'score':float(score)} for box,score,label in zip(result['boxes'],result['scores'],labels))
        evidence.statuses['detector']='ok'
        return evidence

    def close(self):
        del self.model,self.processor
        import gc
        gc.collect()
        if self.torch.cuda.is_available():self.torch.cuda.empty_cache()
