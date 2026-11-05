"""Image-level claims are distinct from object detections and text similarities."""
from pathlib import Path
import hashlib
import math
import time
from .contracts import digest,finite_tree

NATIVE_FACTS={
    'bridges_any':'bridge.present','buildings_any':'building.present',
    'buildings_affected_or_greater':'building.affected_or_greater',
    'buildings_minor_or_greater':'building.minor_or_greater','debris_any':'debris.present',
    'flooding_any':'flooding.present','flooding_structures':'building.flooded',
    'roads_any':'road.present','roads_damage':'road.damaged',
    'trees_any':'tree.present','trees_damage':'tree.damaged','water_any':'water.present'}
MODEL_ID='MITLL/LADI-v2-classifier-small-reference'
MODEL_REVISION='6ba5a0300601284bc6b69dddb46c8991a8ac196c'


def image_claim_predictions(evidence):
    """Reject malformed producer claims; no dataset labels enter this contract."""
    output={}
    for key,row in evidence.provenance.get('image_claim_predictions',{}).items():
        if not isinstance(key,str) or not isinstance(row,dict):continue
        value=row.get('value');score=row.get('score');threshold=row.get('threshold')
        if type(value) is not bool or not isinstance(row.get('method'),str):continue
        if type(score) not in (int,float) or not math.isfinite(score) or not 0<=score<=1:continue
        if type(threshold) not in (int,float) or not math.isfinite(threshold) or not 0<=threshold<=1:continue
        if value!=(score>threshold):continue
        output[key]=dict(row)
    return output


class NativeAerialClaimProvider:
    """Optional aerial-domain producer, not a universal scene classifier.

    Reference weights were trained on LADI train. Default 0.5 is the published
    decision rule, not a calibrated threshold. Official preprocessing is preserved.
    """
    def __init__(self,model_path=None,device='cpu',threshold=.5,batch_size=2,preprocessing='checkpoint_native'):
        if type(threshold) not in (float,int) or not math.isfinite(threshold) or not 0<=threshold<=1 or type(batch_size) is not int or batch_size<1:
            raise ValueError('invalid classifier settings')
        if preprocessing not in ('checkpoint_native','aspect_preserving_pad'):raise ValueError('unknown classifier preprocessing')
        self.model_path=model_path;self.device=device;self.threshold=float(threshold)
        self.batch_size=batch_size;self.loaded=False;self.timings=[]
        self.preprocessing=preprocessing

    def load(self):
        if self.loaded:return
        import torch
        from transformers import AutoImageProcessor,AutoModelForImageClassification
        started=time.perf_counter();self.torch=torch
        location=str(self.model_path or MODEL_ID)
        kwargs={'local_files_only':True} if self.model_path else {'revision':MODEL_REVISION}
        self.processor=AutoImageProcessor.from_pretrained(location,**kwargs)
        self.model=AutoModelForImageClassification.from_pretrained(location,use_safetensors=True,**kwargs).to(self.device).eval()
        self.labels={int(k):v for k,v in self.model.config.id2label.items()}
        if set(self.labels.values())!=set(NATIVE_FACTS):raise ValueError('native classifier label ontology mismatch')
        if self.model_path:
            weights=list(Path(self.model_path).glob('*.safetensors'))
        else:
            from huggingface_hub import hf_hub_download
            weights=[Path(hf_hub_download(MODEL_ID,'model.safetensors',revision=MODEL_REVISION))]
        hashes={}
        for path in weights:
            h=hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda:handle.read(1024*1024),b''):h.update(chunk)
            hashes[path.name]=h.hexdigest()
        self.provenance={'model_id':MODEL_ID if not self.model_path else 'local:'+str(Path(self.model_path).resolve()),
            'revision':MODEL_REVISION if not self.model_path else None,'weights_sha256':hashes,
            'architecture':self.model.config.model_type,'labels':{str(k):v for k,v in self.labels.items()},
            'preprocessing':self.processor.to_dict(),'input_policy':self.preprocessing,
            'preprocessing_sha256':digest({'processor':self.processor.to_dict(),'input_policy':self.preprocessing}),
            'declared_domain':'aerial disaster imagery',
            'training_exposure':'official LADI training split' if not self.model_path else 'unverified local checkpoint',
            'threshold':self.threshold,'threshold_source':'published default' if self.threshold==.5 else 'explicit experimental threshold',
            'calibrated':False,'load_seconds':time.perf_counter()-started,'device':self.device}
        if not finite_tree(self.provenance):raise ValueError('native classifier provenance must use finite values and string keys')
        self.loaded=True

    def predict(self,images):
        if not images:return []
        self.load();output=[]
        for start in range(0,len(images),self.batch_size):
            began=time.perf_counter();batch=images[start:start+self.batch_size]
            images_rgb=[i.convert('RGB') for i in batch]
            if self.preprocessing=='aspect_preserving_pad':
                from .vision import FrozenVisionProvider
                crop=self.processor.to_dict().get('crop_size',{})
                size=crop.get('height',448) if isinstance(crop,dict) else int(crop)
                images_rgb=[FrozenVisionProvider._pad(image,size) for image in images_rgb]
                inputs=self.processor(images=images_rgb,do_resize=False,do_center_crop=False,return_tensors='pt').to(self.device)
            else:inputs=self.processor(images=images_rgb,return_tensors='pt').to(self.device)
            if str(self.device).startswith('cuda'):self.torch.cuda.synchronize(self.device)
            with self.torch.inference_mode():scores=self.model(**inputs).logits.sigmoid()
            if str(self.device).startswith('cuda'):self.torch.cuda.synchronize(self.device)
            if scores.ndim!=2 or scores.shape!=(len(batch),len(self.labels)) or not self.torch.isfinite(scores).all():
                raise ValueError('invalid native classifier output')
            for vector in scores.cpu().tolist():
                output.append({NATIVE_FACTS[self.labels[index]]:{'value':score>self.threshold,
                    'score':float(score),'threshold':self.threshold,'calibrated':False,
                    'native_label':self.labels[index],'method':'ladi_reference_multilabel_sigmoid_v1'}
                    for index,score in enumerate(vector)})
            self.timings.append({'images':len(batch),'wall_seconds':time.perf_counter()-began,
                'device':self.device,'pixel_tensor_shape':list(inputs['pixel_values'].shape)})
        return output

    def enrich(self,images,evidence):
        if len(images)!=len(evidence):raise ValueError('images and evidence differ')
        for item,claims in zip(evidence,self.predict(images)):
            item.provenance={**item.provenance,'image_claim_predictions':claims,
                'image_claim_producer':self.provenance}
            item.probabilities.update({key:row['score'] for key,row in claims.items()})
            item.statuses['image_claim_classifier']='ok'
        return evidence

    def close(self):
        for name in ('model','processor'):
            if hasattr(self,name):delattr(self,name)
        self.loaded=False
