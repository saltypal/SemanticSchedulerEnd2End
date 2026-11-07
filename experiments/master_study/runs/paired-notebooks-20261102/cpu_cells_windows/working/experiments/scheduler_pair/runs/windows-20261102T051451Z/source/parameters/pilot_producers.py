"""Explicit GPU0 global/local evidence production; annotations never enter here."""
from pathlib import Path
import hashlib
import time
import numpy as np
from .contracts import digest
from .vision import FrozenVisionProvider
from .pilot_parameters import VOCABULARY,canonical_label,context_similarity_key
from .model_settings import DetectorSettings

CONTRASTS={
 'building.affected_or_greater':('a photograph of a building affected by a disaster','a photograph of an unaffected intact building'),
 'road.damaged':('a photograph of a physically damaged road','a photograph of an intact undamaged road'),
 'road.blocked':('a photograph of a road blocked by an obstruction','a photograph of a clear unobstructed road'),
 'incident.informative':('an informative photograph showing a disaster incident','an unrelated decorative photograph with no incident information')}


def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


class PilotVisionProducer:
    def __init__(self,clip_path,missions,yolo_checkpoint='yolov8s-worldv2.pt',device='cuda:0',image_claim_provider=None,enable_detector=True,detector_settings=None,clip_preprocessing='aspect_preserving_pad',vocabulary=None):
        if device not in ('cuda:0','cpu'):raise ValueError('producer requires explicit GPU0 or CPU preflight')
        if not missions:raise ValueError('at least one configured mission is required')
        if type(enable_detector) is not bool:raise ValueError('enable_detector must be boolean')
        self.global_provider=FrozenVisionProvider('clip',clip_path,device=device,batch_size=8,local_files_only=True,preprocessing=clip_preprocessing)
        self.missions=list(missions.values());self.checkpoint=yolo_checkpoint;self.device=device;self.timings=[]
        self.loaded=False
        self.image_claim_provider=image_claim_provider
        self.enable_detector=enable_detector
        self.detector_settings=detector_settings or DetectorSettings()
        self.vocabulary=tuple(VOCABULARY if vocabulary is None else vocabulary)
        if (not self.vocabulary or any(not isinstance(v,str) or not v.strip() for v in self.vocabulary)
                or len({canonical_label(v) for v in self.vocabulary})!=len(self.vocabulary)):
            raise ValueError('detector vocabulary requires unique nonempty canonical category names')

    def load(self):
        if self.loaded:return
        self.global_provider.load()
        if not self.enable_detector:
            self.detector=None;self.provenance={'detector':'not_requested','device':self.device}
            self.loaded=True;return
        from ultralytics import YOLOWorld
        self.detector=YOLOWorld(self.checkpoint).to(self.device)
        self.detector.set_classes(list(self.vocabulary))
        path=Path(self.detector.ckpt_path)
        self.provenance={'detector':'YOLO-Worldv2-small','checkpoint':str(path),
            'weights_sha256':file_sha256(path),'vocabulary':list(self.vocabulary),
            'canonical_vocabulary':[canonical_label(v) for v in self.vocabulary],
            'vocabulary_sha256':digest(self.vocabulary),'conf_threshold':self.detector_settings.confidence,
            'nms_iou':self.detector_settings.iou,'inference_size':self.detector_settings.image_size,
            'settings_sha256':self.detector_settings.sha256,'predict_settings':self.detector_settings.predict_kwargs(),
            'device':self.device,'classes_are_not_emergency_truth':True}
        self.loaded=True

    def extract_batch(self,images,missions=None):
        if not images:
            if missions is not None and len(missions):raise ValueError('images and missions differ')
            return []
        missions=[self.missions[0]]*len(images) if missions is None else list(missions)
        if len(images)!=len(missions):raise ValueError('images and missions differ')
        self.load();started=time.perf_counter()
        result=self.global_provider.extract_batch(images,missions)
        profiles={mission.version:mission for mission in [*self.missions,*missions]}
        tasks=[clause for mission in profiles.values() for clause in mission.conditions]
        tasks.extend((context_similarity_key(mission),mission.paragraph) for mission in profiles.values())
        task_features=self.global_provider._texts([text for _,text in tasks])
        contrast_names=list(CONTRASTS)
        contrast_features=self.global_provider._texts([text for names in CONTRASTS.values() for text in names])
        for image,evidence in zip(images,result):
            vector=np.asarray(evidence.embedding)
            evidence.similarities.update({claim:float(vector@feature) for (claim,_),feature in zip(tasks,task_features)})
            rows=[]
            if self.detector is not None:
                settings=getattr(self,'detector_settings',DetectorSettings())
                output=self.detector.predict(source=image,device=self.device,**settings.predict_kwargs())[0]
                predictor=getattr(self.detector,'predictor',None)
                if predictor is not None:
                    # Capture resolved library defaults as well as our overrides.
                    args=vars(predictor.args)
                    self.provenance['effective_predict_settings']={k:args.get(k) for k in (
                        'task','mode','conf','iou','imgsz','max_det','agnostic_nms','half','augment',
                        'rect','batch','classes','save','save_txt','save_conf','save_crop','show',
                        'visualize','embed','retina_masks','vid_stride','stream_buffer','compile')}
                for box,score,label in zip(output.boxes.xyxy.detach().cpu().tolist(),
                        output.boxes.conf.detach().cpu().tolist(),output.boxes.cls.detach().cpu().tolist()):
                    rows.append({'label':canonical_label(output.names[int(label)]),'score':float(score),'box':box})
            evidence.detections=tuple(rows);evidence.statuses['detector']='ok' if self.detector is not None else 'not_requested'
            scores=vector@contrast_features.T
            contrasts={key:{'positive_similarity':float(scores[2*i]),'negative_similarity':float(scores[2*i+1]),
                            'positive_text':CONTRASTS[key][0],'negative_text':CONTRASTS[key][1],
                            'decision_threshold_margin':0.,'calibrated':False} for i,key in enumerate(contrast_names)}
            evidence.provenance={**evidence.provenance,'detector_provenance':self.provenance,
                                 'condition_contrasts':contrasts}
        self.timings.append({'images':len(images),'wall_seconds':time.perf_counter()-started,'device':self.device})
        if self.image_claim_provider is not None:self.image_claim_provider.enrich(images,result)
        return result

    def close(self):
        if hasattr(self,'detector'):del self.detector
        self.global_provider.close()
        if self.image_claim_provider is not None:self.image_claim_provider.close()


def snapshot_model(model_id,revision=None):
    """Pin a public model revision and record files;never download optimizer archives."""
    from huggingface_hub import HfApi,snapshot_download
    info=HfApi().model_info(model_id,revision=revision)
    path=snapshot_download(model_id,revision=info.sha,
        allow_patterns=['*.safetensors','*.json','*.txt','*.model','*.tiktoken','*.jinja','merges.txt','vocab.json'])
    if not list(Path(path).glob('*.safetensors')):
        path=snapshot_download(model_id,revision=info.sha,allow_patterns=['pytorch_model.bin','*.json','*.txt','*.model'])
    return path,info.sha
