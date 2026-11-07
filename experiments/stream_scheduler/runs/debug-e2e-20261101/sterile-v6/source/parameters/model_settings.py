"""Small explicit inference contracts shared by producers and search manifests."""
from dataclasses import dataclass,asdict
import math
from .contracts import digest


@dataclass(frozen=True)
class DetectorSettings:
    confidence:float=.25
    iou:float=.5
    image_size:int=640
    max_detections:int=300
    agnostic_nms:bool=False

    def __post_init__(self):
        for value in (self.confidence,self.iou):
            if type(value) not in (int,float) or not math.isfinite(value) or not 0<value<=1:
                raise ValueError('detector confidence/IoU must be finite in (0,1]')
        if type(self.image_size) is not int or self.image_size<32 or self.image_size%32:
            raise ValueError('detector size must be a positive multiple of32')
        if type(self.max_detections) is not int or self.max_detections<=0 or type(self.agnostic_nms) is not bool:
            raise ValueError('invalid detector NMS limits')

    @property
    def sha256(self):return digest(asdict(self))

    def predict_kwargs(self):
        return {'conf':self.confidence,'iou':self.iou,'imgsz':self.image_size,
                'max_det':self.max_detections,'agnostic_nms':self.agnostic_nms,
                'half':False,'augment':False,'verbose':False}


@dataclass(frozen=True)
class ReviewerSettings:
    max_new_tokens:int=384
    max_pixels:int=512*512
    image_max_side:int|None=512
    prompt_variant:str='candidate_provenance_v2'
    claim_scope:str='all'
    enable_thinking:bool=False

    def __post_init__(self):
        if self.enable_thinking is not False:raise ValueError('thinking must be explicitly disabled for this study')
        if type(self.max_new_tokens) is not int or not 64<=self.max_new_tokens<=1024:
            raise ValueError('review response budget must be64..1024 tokens')
        if type(self.max_pixels) is not int or not 4096<=self.max_pixels<=1024*1024:
            raise ValueError('review pixel budget outside bounded range')
        if self.image_max_side is not None and (type(self.image_max_side) is not int or self.image_max_side<64):
            raise ValueError('invalid review pre-resize size')
        if self.prompt_variant not in ('candidate_provenance_v2','candidate_provenance_concise_v3',
                                       'candidate_provenance_explicit_v4','semantic_contract_v5','scene_v2_offline_v1'):
            raise ValueError('unknown reviewer prompt variant')
        if self.claim_scope not in ('all','mission'):raise ValueError('unknown review claim scope')

    @property
    def sha256(self):return digest(asdict(self))
