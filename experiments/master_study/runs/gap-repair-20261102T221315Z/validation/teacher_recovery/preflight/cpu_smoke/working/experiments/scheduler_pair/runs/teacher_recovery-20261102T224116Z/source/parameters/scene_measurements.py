"""Vocabulary-limited YOLO detections and exact rectangular scene measurements."""
from collections import Counter
from dataclasses import dataclass, field
import math
from pathlib import Path

from .model_settings import DetectorSettings
from .pilot_parameters import VOCABULARY, canonical_label, union_area


@dataclass(frozen=True)
class SceneMeasurements:
    instances: int | None = None
    categories: int | None = None
    largest_box_fraction: float | None = None
    union_box_coverage: float | None = None
    saturated: bool = False
    status: str = 'unavailable'
    reason: str | None = None
    audit: dict = field(default_factory=dict)

    def __post_init__(self):
        for key in ('instances','categories'):
            value=getattr(self,key)
            if value is not None and (type(value) is not int or not 0<=value<=65535):
                raise ValueError('invalid scene count: '+key)
        for key in ('largest_box_fraction','union_box_coverage'):
            value=getattr(self,key)
            if value is not None and (isinstance(value,bool) or not math.isfinite(value) or not 0<=value<=1):
                raise ValueError('invalid scene fraction: '+key)
        if type(self.saturated) is not bool:raise ValueError('invalid detector saturation flag')
        if self.instances is not None and self.categories is not None and self.categories>self.instances:
            raise ValueError('distinct categories cannot exceed detected instances')
        if self.instances is not None and self.categories is not None and self.instances>0 and self.categories==0:
            raise ValueError('detected instances require at least one category')
        if (self.largest_box_fraction is not None and self.union_box_coverage is not None and
                self.largest_box_fraction>self.union_box_coverage+1e-12):
            raise ValueError('largest box cannot exceed the union coverage')
        if self.instances==0 and self.largest_box_fraction is not None:
            raise ValueError('no largest box exists without detections')
        if self.instances==0 and self.union_box_coverage not in (None,0):
            raise ValueError('empty detections cannot cover image area')

    @classmethod
    def unavailable(cls,reason):
        return cls(status='unavailable',reason=str(reason))

    def values(self):
        return {k:getattr(self,k) for k in ('instances','categories','largest_box_fraction','union_box_coverage')}


def measure_detections(detections, width, height, *, max_detections=300, provenance=None):
    """Measure retained XYXY rectangles in EXIF-oriented image pixels.

    Invalid/nonfinite/zero-area boxes are rejected. Detector scores and names
    are audit evidence only; categories do not receive importance weights.
    """
    if type(width) is not int or type(height) is not int or width<=0 or height<=0:
        raise ValueError('positive oriented image dimensions required')
    if type(max_detections) is not int or max_detections<=0:raise ValueError('invalid maximum detections')
    if not isinstance(detections,(list,tuple)):raise ValueError('detections must be a finite list')
    boxes=[];names=[];rejected=0;accepted=[]
    known=None if not provenance or 'vocabulary' not in provenance else {canonical_label(v) for v in provenance['vocabulary']}
    for row in detections:
        try:
            if not isinstance(row['label'],str):raise ValueError('invalid detection label')
            label=canonical_label(row['label'])
            coords=row['box']
            if not label or len(coords)!=4:raise ValueError('invalid detection')
            if known is not None and label not in known:raise ValueError('unknown detector category')
            values=[float(x) for x in coords]
            if not all(math.isfinite(x) for x in values):raise ValueError('nonfinite box')
            x1,y1,x2,y2=values
            x1=max(0.,min(float(width),x1));x2=max(0.,min(float(width),x2))
            y1=max(0.,min(float(height),y1));y2=max(0.,min(float(height),y2))
            if x2<=x1 or y2<=y1:raise ValueError('zero-area or reversed box')
            score=row.get('score')
            if score is not None:
                score=float(score)
                if not math.isfinite(score) or not 0<=score<=1:raise ValueError('invalid detection score')
            box=[x1,y1,x2,y2];boxes.append(box);names.append(label)
            accepted.append({'label':label,'box':box,'score':score})
        except (KeyError,TypeError,ValueError,OverflowError):rejected+=1
    pixels=float(width*height)
    areas=[(b[2]-b[0])*(b[3]-b[1])/pixels for b in boxes]
    return SceneMeasurements(len(boxes),len(set(names)),max(areas) if areas else None,
        min(1.,union_area(boxes)/pixels),len(detections)>=max_detections,'ok',None,
        {'category_counts':dict(sorted(Counter(names).items())),'detections':accepted,
         'rejected_boxes':rejected,'width':width,'height':height,'producer':provenance or {},
         'interpretation':'detector-dependent rectangular extents, not physical distance or urgency'})


class YOLOWorldSceneProvider:
    """Lazy YOLO-only adapter. A local weights path must be supplied explicitly."""
    def __init__(self,checkpoint,device='cpu',settings=None,vocabulary=VOCABULARY):
        self.checkpoint=Path(checkpoint);self.device=device;self.settings=settings or DetectorSettings()
        self.vocabulary=tuple(vocabulary);self.model=None;self.load_error=None;self.provenance={}
        if not self.vocabulary or len(set(map(canonical_label,self.vocabulary)))!=len(self.vocabulary):
            raise ValueError('nonempty canonical unique vocabulary required')

    def load(self):
        if self.model is not None:return
        if self.load_error is not None:raise RuntimeError(self.load_error)
        try:
            if not self.checkpoint.is_file():raise FileNotFoundError('YOLO checkpoint not found: '+str(self.checkpoint))
            import hashlib
            from ultralytics import YOLOWorld
            model=YOLOWorld(str(self.checkpoint)).to(self.device)
            model.set_classes(list(self.vocabulary))
            self.provenance={'checkpoint':str(self.checkpoint),'weights_sha256':hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
                'device':self.device,'vocabulary':list(self.vocabulary),'settings':self.settings.predict_kwargs(),
                'settings_sha256':self.settings.sha256}
            self.model=model
        except Exception as error:
            self.load_error=type(error).__name__+': '+str(error)
            raise RuntimeError(self.load_error) from error

    def measure(self,image):
        from PIL import ImageOps
        try:
            self.load()
            oriented=ImageOps.exif_transpose(image).convert('RGB')
            result=self.model.predict(source=oriented,device=self.device,**self.settings.predict_kwargs())[0]
            predictor=getattr(self.model,'predictor',None)
            if predictor is not None:
                resolved=vars(predictor.args)
                self.provenance['effective_predict_settings']={name:resolved.get(name) for name in
                    ('task','mode','conf','iou','imgsz','max_det','agnostic_nms','half','augment','rect','batch','device')}
            boxes=result.boxes.xyxy.detach().cpu().tolist()
            scores=result.boxes.conf.detach().cpu().tolist()
            classes=result.boxes.cls.detach().cpu().tolist()
            if not (len(boxes)==len(scores)==len(classes)):raise ValueError('inconsistent detector output lengths')
            rows=[{'label':result.names[int(label)],'box':box,'score':float(score)}
                  for box,score,label in zip(boxes,scores,classes)]
            return measure_detections(rows,oriented.width,oriented.height,
                                      max_detections=self.settings.max_detections,provenance=self.provenance)
        except Exception as error:
            return SceneMeasurements.unavailable(type(error).__name__+': '+str(error))


class UnavailableSceneProvider:
    """Explicit placeholder when the detector is not configured."""
    def __init__(self,reason):self.reason=str(reason)
    def measure(self,image):return SceneMeasurements.unavailable(self.reason)
