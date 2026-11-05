"""Grounding DINO alternative producer for the existing scene contract."""
from pathlib import Path
import hashlib
from .pilot_parameters import VOCABULARY
from .scene_measurements import SceneMeasurements, measure_detections


def suppress_duplicates(rows,iou=.5,maximum=300):
    """Class-aware rectangular NMS after shared validity/clipping checks."""
    kept=[]
    for row in sorted(rows,key=lambda r:r['score'] or 0,reverse=True):
        a=row['box'];area=(a[2]-a[0])*(a[3]-a[1]);duplicate=False
        for old in kept:
            if row['label']!=old['label']:continue
            b=old['box'];intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
            union=area+(b[2]-b[0])*(b[3]-b[1])-intersection
            if union>0 and intersection/union>iou:duplicate=True;break
        if not duplicate:kept.append(row)
        if len(kept)>=maximum:break
    return kept


class GroundingDINOSceneProvider:
    def __init__(self,checkpoint,device='cpu',vocabulary=VOCABULARY,threshold=.35,text_threshold=.25):
        self.path=Path(checkpoint);self.device=device;self.vocabulary=tuple(vocabulary)
        if not self.vocabulary or any(not isinstance(v,str) or not v.strip() for v in self.vocabulary):
            raise ValueError('nonempty object vocabulary required')
        if not 0<threshold<=1 or not 0<text_threshold<=1:raise ValueError('invalid detector thresholds')
        self.threshold,self.text_threshold=threshold,text_threshold
        self.model=None

    def load(self):
        if self.model is not None:return
        import torch
        from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
        self.torch=torch
        self.processor=AutoProcessor.from_pretrained(str(self.path),local_files_only=True)
        self.model=AutoModelForZeroShotObjectDetection.from_pretrained(str(self.path),local_files_only=True).to(self.device).eval()
        self.provenance={'producer':'grounding_dino','vocabulary':list(self.vocabulary),
            'threshold':self.threshold,'text_threshold':self.text_threshold,
            'weights_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in self.path.glob('*.safetensors')},'device':self.device,
            'interpretation':'box/text matching, not verified condition or priority'}

    def measure(self,image):
        from PIL import ImageOps
        try:
            self.load();image=ImageOps.exif_transpose(image).convert('RGB')
            inputs=self.processor(images=image,text='. '.join(self.vocabulary)+'.',return_tensors='pt').to(self.device)
            with self.torch.inference_mode():output=self.model(**inputs)
            result=self.processor.post_process_grounded_object_detection(output,inputs.input_ids,
                threshold=self.threshold,text_threshold=self.text_threshold,target_sizes=[(image.height,image.width)])[0]
            names=result.get('text_labels',result.get('labels'))
            if names is None or any(not isinstance(n,str) for n in names):
                raise ValueError('detector did not return textual phrases')
            rows=[{'label':name,'score':float(score),'box':box.tolist()}
                  for name,score,box in zip(names,result['scores'],result['boxes'])]
            # Phrases may contain multiple query words; do not silently merge
            # these into a predeclared YOLO category. Audit actual phrases.
            provenance={**self.provenance,'vocabulary':sorted(set(names)),
                        'class_aware_nms_iou':.5,'max_detections':300,
                        'category_scope':'retained text phrases; no inferred urgency or merged taxonomy'}
            preliminary=measure_detections(rows,image.width,image.height,provenance=provenance)
            kept=suppress_duplicates(preliminary.audit['detections'])
            final=measure_detections(kept,image.width,image.height,max_detections=300,provenance=provenance)
            final.audit.update(raw_proposals=len(rows),rejected_boxes=preliminary.audit['rejected_boxes'])
            return final
        except Exception as error:
            return SceneMeasurements.unavailable(type(error).__name__+': '+str(error))

    def close(self):
        self.model=None
