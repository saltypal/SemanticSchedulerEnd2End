"""Integration regressions for the actual extractor/producer contracts."""
from types import SimpleNamespace
import numpy as np
import pytest
from parameters.extractor import VisualEvidence
from parameters.pilot_parameters import PilotMemory,PilotParameterExtractor,context_similarity_key
from parameters.pilot_producers import PilotVisionProducer
from parameters.annotation_pilot import representation
from parameters.ue_profiles import load_ue_profiles


def test_public_image_api_propagates_mission_and_uses_operator_paragraph():
    import torch
    from PIL import Image
    mission=load_ue_profiles()[0]['building_damage']
    forwarded=[]
    class GlobalProvider:
        def extract_batch(self,images,missions):
            forwarded.extend(missions)
            return [VisualEvidence(embedding=(1.,0.),width=i.width,height=i.height) for i in images]
        def _texts(self,texts):
            return np.array([(0.,1.) if text==mission.paragraph else (1.,0.) for text in texts])
    producer=PilotVisionProducer.__new__(PilotVisionProducer)
    producer.load=lambda:None
    producer.missions=[mission];producer.device='cpu';producer.timings=[]
    producer.image_claim_provider=None
    producer.global_provider=GlobalProvider();producer.provenance={'scope':'fixture_only'}
    producer.detector=SimpleNamespace(predict=lambda **kwargs:[SimpleNamespace(
        boxes=SimpleNamespace(xyxy=torch.empty((0,4)),conf=torch.empty(0),cls=torch.empty(0)),names={})])
    extractor=PilotParameterExtractor(provider=producer)
    image=Image.new('RGB',(10,10))
    bundle=extractor.extract(image,mission,{'source_id':'ue','image_token':'one'},PilotMemory('ue',mission.mission_id))
    assert forwarded==[mission]
    assert bundle.parameters['relevance'].value==pytest.approx(.5) # Full paragraph cosine0; clause cosine1.
    assert bundle.parameters['relevance'].supporting['text_source']=='operator_paragraph'
    assert bundle.parameters['spatial'].available and bundle.parameters['spatial'].value['target_count']==0
    assert producer.timings[-1]['device']=='cpu'
    full=representation(bundle,mission)
    compact=representation(bundle,mission,'compact')
    assert full['representation']=='structured'
    assert not full['is_ablation'] and compact['is_ablation']
    assert not full['claim_availability']['building.affected_or_greater']['available']
    assert 'condition_contrast_evidence' in full['parameters']['scene']['value']
    assert 'building.present' not in full['claims'] # No detections cannot prove image absence.
    assert not compact['claim_availability']['building.affected_or_greater']['available']
    assert 'building.affected_or_greater' not in compact['claims']
    assert extractor.extract_batch([],[],[],[])==[]
    with pytest.raises(ValueError,match='differ'):producer.extract_batch([image],[])


def test_context_failure_does_not_turn_into_zero_relevance():
    mission=load_ue_profiles()[0]['building_damage']
    evidence=VisualEvidence(similarities={context_similarity_key(mission):float('nan')})
    result=PilotParameterExtractor().extract_context_relevance(evidence,mission)
    assert not result.available and result.value is None and 'invalid' in result.reason
