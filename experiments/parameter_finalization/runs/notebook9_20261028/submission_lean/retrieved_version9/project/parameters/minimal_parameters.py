"""The final four-output candidate; old eight-field extractors stay historical.

Query has no KB side effects. Commit only after serialization/acceptance succeeds.
The 28-byte wire layout is an additive experimental protocol, not a radio format.
"""
from dataclasses import dataclass
import math
import struct
import numpy as np

from .extractor import unit, embedding_signature
from .pilot_parameters import context_similarity_key
from .scene_measurements import SceneMeasurements,UnavailableSceneProvider

WIRE = struct.Struct("<BBHIIHQIBB")
SCENE_WIRE = struct.Struct("<BHHBB")


@dataclass(frozen=True)
class MinimalHeader:
    source: int
    session: int
    message: int
    profile: int
    scheduling_class: int | None = None  # 0 Critical, 1 High, 2 Medium, 3 Low
    relevance: float | None = None      # raw image/text cosine, [-1,1]
    novelty: float | None = None        # appearance distance, [0,1]
    capture_time_ms: int | None = None
    allowed_delay_ms: int | None = None
    cold_start: bool = False

    def __post_init__(self):
        for key,bits in (("source",16),("session",32),("message",32),("profile",16)):
            value=getattr(self,key)
            if type(value) is not int or not 0<=value<2**bits: raise ValueError("invalid routing field: "+key)
        if self.scheduling_class is not None and (type(self.scheduling_class) is not int or not 0<=self.scheduling_class<=3):
            raise ValueError("invalid scheduling class")
        for name,low,high in (("relevance",-1,1),("novelty",0,1)):
            value=getattr(self,name)
            if value is not None and (isinstance(value,bool) or not math.isfinite(value) or not low<=value<=high):
                raise ValueError("invalid "+name)
        for name,bits in (("capture_time_ms",64),("allowed_delay_ms",32)):
            value=getattr(self,name)
            if value is not None and (type(value) is not int or not 0<value<2**bits): raise ValueError("invalid "+name)
        if type(self.cold_start) is not bool or self.cold_start and self.novelty is None:
            raise ValueError("cold start requires available novelty")

    def to_bytes(self):
        flags=(self.scheduling_class or 0)
        for bit,value in ((2,self.scheduling_class),(3,self.relevance),(4,self.novelty),(5,self.capture_time_ms),(6,self.allowed_delay_ms)):
            if value is not None: flags|=1<<bit
        if self.cold_start:flags|=128
        r=0 if self.relevance is None else round((self.relevance+1)*127.5)
        n=0 if self.novelty is None else round(self.novelty*255)
        return WIRE.pack(1,flags,self.source,self.session,self.message,self.profile,
                         self.capture_time_ms or 0,self.allowed_delay_ms or 0,r,n)

    @classmethod
    def from_bytes(cls, payload):
        if len(payload)!=WIRE.size:raise ValueError("minimal header must be exactly 28 bytes")
        version,f,source,session,message,profile,capture,delay,r,n=WIRE.unpack(payload)
        if version!=1:raise ValueError("unsupported minimal-header version")
        return cls(source,session,message,profile,(f&3) if f&4 else None,
                   r/127.5-1 if f&8 else None,n/255 if f&16 else None,
                   capture if f&32 else None,delay if f&64 else None,bool(f&128))


@dataclass(frozen=True)
class ExtendedParameterHeader:
    """Version 2: existing 28-byte core plus seven scene bytes."""
    core: MinimalHeader
    scene: SceneMeasurements

    def to_bytes(self):
        values=self.scene.values()
        flags=sum((1<<i) for i,name in enumerate(values) if values[name] is not None)
        if self.scene.saturated:flags|=16
        a=0 if values['largest_box_fraction'] is None else round(values['largest_box_fraction']*255)
        b=0 if values['union_box_coverage'] is None else round(values['union_box_coverage']*255)
        core=self.core.to_bytes()
        return bytes([2])+core[1:]+SCENE_WIRE.pack(flags,values['instances'] or 0,values['categories'] or 0,a,b)

    @classmethod
    def from_bytes(cls,payload):
        if len(payload)!=WIRE.size+SCENE_WIRE.size or payload[0]!=2:
            raise ValueError('extended header must be version 2 and 35 bytes')
        core=MinimalHeader.from_bytes(bytes([1])+payload[1:WIRE.size])
        flags,instances,categories,largest,coverage=SCENE_WIRE.unpack(payload[WIRE.size:])
        if flags & 0xe0:raise ValueError('reserved scene bits must be zero')
        raw=(instances,categories,largest,coverage)
        if any(not flags & (1<<i) and value!=0 for i,value in enumerate(raw)):
            raise ValueError('unavailable scene fields must have zero encoding')
        scene=SceneMeasurements(instances if flags&1 else None,categories if flags&2 else None,
            largest/255 if flags&4 else None,coverage/255 if flags&8 else None,bool(flags&16),
            'decoded',None,{})
        return cls(core,scene)


def compact_parameter_record(bundle):
    """JSON-friendly selected values; detailed producer observations remain audit-only."""
    return {'schema':'better_minimal_parameters_v2','spi':bundle['spi']['level'],
        'spi_display':('Normal' if bundle['spi']['level'] in (None,'Low') else bundle['spi']['level']),
        'spi_available':bundle['spi']['available'],
        'spi_reason':bundle['spi'].get('reason'),
        'relevance':bundle['relevance']['cosine'],'novelty':bundle['novelty']['value'],
        'novelty_cold_start':bundle['novelty']['cold_start'],
        'capture_time_ms':bundle['capture']['capture_time_ms'],'allowed_delay_ms':bundle['capture']['allowed_delay_ms'],
        'scene':bundle['scene'].values()}


class MinimalParameterExtractor:
    def __init__(self, provider=None, spi_classifier=None, scene_provider=None):
        self.provider,self.spi_classifier,self.scene_provider=provider,spi_classifier,scene_provider

    def extract(self, image, mission, metadata, memory):
        if self.provider is None:raise RuntimeError("visual provider required")
        evidence=self.provider.extract_batch([image],[mission])[0]
        scene=self.scene_provider.measure(image) if self.scene_provider is not None else None
        return self.extract_evidence(evidence,mission,metadata,memory,scene)

    def extract_batch(self,images,missions,metadata,memories):
        """Read-only batch: all queries see each memory as it exists at call time."""
        if self.provider is None:raise RuntimeError('visual provider required')
        if not (len(images)==len(missions)==len(metadata)==len(memories)):raise ValueError('batch inputs differ in length')
        evidence=self.provider.extract_batch(images,missions)
        if len(evidence)!=len(images):raise ValueError('visual provider returned wrong batch length')
        scenes=[self.scene_provider.measure(image) for image in images] if self.scene_provider else [None]*len(images)
        return [self.extract_evidence(e,m,d,k,s) for e,m,d,k,s in zip(evidence,missions,metadata,memories,scenes)]

    def extract_evidence(self,evidence,mission,metadata,memory,scene=None):
        if memory.source_id!=metadata.get("source_id") or memory.mission_id!=mission.mission_id:
            raise ValueError("source/mission KB identity mismatch")
        vector=unit(evidence.embedding); signature=embedding_signature(evidence)
        spi={"available":False,"level":None,"scheduling_class":None,"reason":"SPI model artifact not loaded"}
        if self.spi_classifier is not None:
            try:
                if vector is None:raise ValueError("image embedding unavailable")
                spi=self.spi_classifier.predict(vector,signature)
            except (ValueError,FloatingPointError) as error:
                spi={"available":False,"level":None,"scheduling_class":None,"reason":str(error)}
        relevance=evidence.similarities.get(context_similarity_key(mission))
        if relevance is not None:
            try:
                if isinstance(relevance,bool) or not math.isfinite(relevance) or not -1.00001<=relevance<=1.00001:
                    relevance=None
            except (TypeError,ValueError):relevance=None
        if relevance is not None:relevance=float(np.clip(relevance,-1,1))
        relevance_reason=None if relevance is not None else evidence.provenance.get('context_reason','mission/image similarity unavailable')
        previous=[r["embedding"] for r in memory.entries if r["embedding"] is not None and
                  r["embedding"].shape==(vector.shape if vector is not None else ()) and r["embedding_signature"]==signature]
        cold=vector is not None and not memory.entries
        novelty=1. if cold else ((1-float(np.clip(max(vector@v for v in previous),-1,1)))/2 if previous else None)
        novelty_reason=('image embedding unavailable' if vector is None else
                        'cold-start default; no measured prior difference' if cold else
                        'no comparable prior embeddings' if novelty is None else None)
        captured=metadata.get("capture_time_ms")
        capture_reason=None
        if captured is None:capture_reason='capture time not supplied'
        elif metadata.get("timestamp_trusted") is not True:capture_reason='capture time not trusted'
        elif type(captured) is not int or captured<=0:capture_reason='invalid capture timestamp'
        if capture_reason is not None:captured=None
        now=metadata.get("now_ms")
        if captured is not None and type(now) is int and captured>now:
            captured=None;capture_reason='capture time is later than supplied current time'
        header=MinimalHeader(metadata["source_numeric_id"],metadata["session_id"],metadata["message_id"],
            metadata["profile_id"],spi["scheduling_class"],relevance,novelty,captured,mission.allowed_delay_ms,cold)
        result={"schema":"better_minimal_parameters_v1","spi":spi,
                "relevance":{"available":relevance is not None,"cosine":relevance,"is_probability":False,"reason":relevance_reason},
                "novelty":{"available":novelty is not None,"value":novelty,"cold_start":cold,
                           "reason":novelty_reason,"meaning":"appearance difference from this UE's prior embeddings; not verified new facts"},
                "capture":{"capture_time_ms":captured,"capture_reason":capture_reason,
                           "allowed_delay_ms":mission.allowed_delay_ms,
                           "allowed_delay_reason":None if mission.allowed_delay_ms is not None else 'operator delay not supplied'},
                "header":header,"kb_version":memory.version,"feature_signature":signature}
        if scene is not None:
            if not isinstance(scene,SceneMeasurements):raise TypeError('scene must be SceneMeasurements')
            result.update(schema='better_minimal_parameters_v2',scene=scene,header=ExtendedParameterHeader(header,scene))
            result['compact']=compact_parameter_record(result)
        return result

    def serialize_and_commit(self,evidence,mission,metadata,memory,scene=None):
        bundle=self.extract_evidence(evidence,mission,metadata,memory,scene)
        payload=bundle["header"].to_bytes()
        memory.commit(evidence,metadata["sequence"])
        return bundle,payload

    def transmit_batch(self,images,missions,metadata,memories):
        """Batch visual work, then query/serialize/commit strictly in input order."""
        if self.provider is None:raise RuntimeError('visual provider required')
        if not (len(images)==len(missions)==len(metadata)==len(memories)):raise ValueError('batch inputs differ in length')
        evidence=self.provider.extract_batch(images,missions)
        if len(evidence)!=len(images):raise ValueError('visual provider returned wrong batch length')
        scenes=[self.scene_provider.measure(image) for image in images] if self.scene_provider else [None]*len(images)
        return [self.serialize_and_commit(e,m,d,k,s) for e,m,d,k,s in zip(evidence,missions,metadata,memories,scenes)]


class FinalParameterExtractor(MinimalParameterExtractor):
    """Always emits the v2 contract, even when detector evidence is missing."""
    def __init__(self,provider=None,spi_classifier=None,scene_provider=None):
        super().__init__(provider,spi_classifier,scene_provider or UnavailableSceneProvider('scene detector not configured'))

    def extract_evidence(self,evidence,mission,metadata,memory,scene=None):
        if scene is None:scene=SceneMeasurements.unavailable('scene evidence not supplied')
        return super().extract_evidence(evidence,mission,metadata,memory,scene)
