from dataclasses import replace
import numpy as np
import pytest
from parameters import MissionSpec, MethodSpecification, SemanticParameterExtractor, SenderMemory
from parameters.contracts import MISSIONS, flatten_bundle
from parameters.extractor import VisualEvidence
from parameters.serialization import HeaderCodec


def sample(embedding=(1.,0.),probability=.8):
    return VisualEvidence(embedding=embedding,similarities={'building_damage':.6},
                          probabilities={'building_damage':probability},
                          scene_scores={'road':.3},statuses={'encoder':'ok','detector':'not_requested'})


def metadata(token='a',**extra):
    return {'image_token':token,'source_id':'u','sequence':0,**extra}


def test_query_before_commit_and_novelty_not_rarity():
    mission=MISSIONS['building_damage'];kb=SenderMemory('u',mission.mission_id)
    ex=SemanticParameterExtractor();e=sample()
    a=ex.extract_evidence(e,mission,metadata(),kb)
    assert kb.version==0 and a.parameters['novelty'].value['cold_start']==1
    kb.commit(e,0)
    b=ex.extract_evidence(e,mission,metadata('b'),kb)
    assert b.parameters['novelty'].value['distance']==0
    assert b.parameters['rarity'].value>0
    with pytest.raises(ValueError):kb.commit(e,0)


def test_semantic_delta_when_appearance_is_same():
    mission=MISSIONS['building_damage'];kb=SenderMemory('u',mission.mission_id)
    kb.commit(sample(probability=.2),0)
    ex=SemanticParameterExtractor(MethodSpecification(novelty='semantic_delta'))
    bundle=ex.extract_evidence(sample(probability=.9),mission,metadata(),kb)
    assert bundle.parameters['novelty'].value['distance']==0
    assert bundle.parameters['novelty'].value['added_building_damage']==pytest.approx(.7)


def test_failed_serialization_does_not_commit():
    class Provider:
        def extract_batch(self,images,missions):return [sample() for _ in images]
    kb=SenderMemory('u','building_damage');ex=SemanticParameterExtractor(provider=Provider())
    def fail(bundle):raise ValueError('wire failure')
    with pytest.raises(ValueError):ex.transmit_stream([object()],[MISSIONS['building_damage']],[metadata()],[kb],fail)
    assert kb.version==0


def test_temporal_fields_and_region_are_metadata_not_urgency():
    mission=replace(MISSIONS['building_damage'],allowed_delay_ms=1000,region=(0.,0.,2.,2.))
    ex=SemanticParameterExtractor();kb=SenderMemory('u',mission.mission_id)
    row=ex.extract_evidence(sample(),mission,metadata(capture_time_ms=1000,now_ms=2500,timestamp_trusted=True,
                 expected_decode_ms=200,coordinates=(1.,1.),location_trusted=True),kb)
    assert row.temporal['age_ms']==1500 and row.temporal['completion_slack_ms']==-700
    assert row.parameters['geographic'].value==1
    assert not ex.extract_temporal_metadata(mission,metadata(capture_time_ms=1000,now_ms=900,timestamp_trusted=True))['available']
    assert ex.extract_temporal_metadata(mission,metadata())['age_ms'] is None


def test_runtime_truth_rejected_and_long_paragraph_preserved():
    with pytest.raises(ValueError):VisualEvidence.from_dict({'labels':{'building_damage':1}})
    mission=replace(MISSIONS['building_damage'],paragraph='Inspect buildings. '*500,
                    unsupported_clauses=('infer hidden casualties',))
    assert len(mission.paragraph)>1000 and mission.unsupported_clauses
    ex=SemanticParameterExtractor();kb=SenderMemory('u',mission.mission_id)
    assert ex.extract_evidence(sample(),mission,metadata(),kb).parameters['relevance'].value==pytest.approx(.8)


def test_missing_is_not_zero_and_target_area_is_not_distance():
    mission=MISSIONS['building_damage'];ex=SemanticParameterExtractor();kb=SenderMemory('u',mission.mission_id)
    evidence=sample();missing=ex.extract_evidence(evidence,mission,metadata(),kb)
    assert not missing.parameters['spatial'].available
    evidence.statuses['detector']='ok';evidence.width=evidence.height=100
    zero=ex.extract_evidence(evidence,mission,metadata(),kb)
    assert zero.parameters['spatial'].available and zero.parameters['spatial'].value['largest_area']==0
    evidence.detections=({'label':'building','box':[0,0,50,50],'score':.9},)
    area=ex.extract_evidence(evidence,mission,metadata(),kb)
    assert area.parameters['spatial'].value['largest_area']==.25
    assert area.parameters['spatial'].supporting['physical_distance_available'] is False


@pytest.mark.parametrize('bits',[8,16])
def test_wire_roundtrip_corruption_and_exact_size(bits):
    spec=MethodSpecification(quantization_bits=bits);ex=SemanticParameterExtractor(spec)
    bundle=ex.extract_evidence(sample(),MISSIONS['building_damage'],metadata(),SenderMemory('u','building_damage'))
    flat=flatten_bundle(bundle);codec=HeaderCodec(sorted(flat),spec.sha256,bits)
    wire=codec.encode(bundle);decoded=codec.decode(wire)
    assert len(wire)==codec.describe()['header_bytes']
    assert max(abs(flat[k]-decoded['values'][k]) for k in flat)<=.5/((1<<bits)-1)+1e-12
    damaged=bytearray(wire);damaged[-1]^=1
    with pytest.raises(ValueError):codec.decode(bytes(damaged))


def test_batch_provider_computation_and_explicit_memory_ownership():
    class Provider:
        calls=0
        def extract_batch(self,images,missions):self.calls+=1;return [sample() for _ in images]
    provider=Provider();ex=SemanticParameterExtractor(provider=provider)
    memories=[SenderMemory('u','building_damage'),SenderMemory('u','building_damage')]
    results=ex.extract_batch([1,2],[MISSIONS['building_damage']]*2,[metadata('a'),metadata('b')],memories)
    assert provider.calls==1 and all(kb.version==0 for kb in memories)
    assert results[0].parameters==results[1].parameters


def test_identity_and_invalid_predictions():
    ex=SemanticParameterExtractor();kb=SenderMemory('u','building_damage')
    with pytest.raises(ValueError):ex.extract_evidence(sample(),MISSIONS['building_damage'],metadata(source_id='other'),kb)
    e=sample(embedding=(float('nan'),0));e.similarities={}
    b=ex.extract_evidence(e,MISSIONS['building_damage'],metadata(),kb)
    assert not b.parameters['novelty'].available and not b.parameters['relevance'].available


def test_wire_rejects_another_extraction_specification():
    spec=MethodSpecification();bundle=SemanticParameterExtractor(spec).extract_evidence(sample(),MISSIONS['building_damage'],metadata(),SenderMemory('u','building_damage'))
    other=MethodSpecification(novelty='semantic_delta')
    codec=HeaderCodec(sorted(flatten_bundle(bundle)),other.sha256)
    with pytest.raises(ValueError,match='specification'):codec.encode(bundle)


def test_exported_heads_and_streamed_kb_reproduce_runtime_inputs():
    from parameters.learning import ClaimHeads,HeadedProvider
    class Provider:
        def extract_batch(self,images,missions):
            return [VisualEvidence(embedding=(1.,0.),similarities={'building_damage':.6}) for image in images]
    artifact={'schema':'better_claim_heads_v1','support':{},'models':{'building_damage':{'coefficient':[2.,0.],'intercept':0.}}}
    heads=ClaimHeads.from_dict(artifact);spec=MethodSpecification(relevance='claim_probability',novelty='semantic_delta')
    extractor=SemanticParameterExtractor(spec,HeadedProvider(Provider(),heads));memory=SenderMemory('u','building_damage')
    fields=('relevance','novelty.distance','novelty.cold_start','novelty.added_building_damage')
    codec=HeaderCodec(fields,spec.sha256)
    rows=extractor.transmit_stream([1,2],[MISSIONS['building_damage']]*2,[metadata('a',sequence=0),metadata('b',sequence=1)],[memory,memory],codec.encode)
    assert rows[0][0].parameters['relevance'].value==pytest.approx(1/(1+np.exp(-2)))
    assert rows[1][0].parameters['novelty'].value['distance']==0
    assert rows[1][0].kb_version==1 and memory.version==2
    assert codec.decode(rows[1][1])['values']['novelty.cold_start']==0
