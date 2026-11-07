import pytest
from parameters.extractor import VisualEvidence
from parameters.pilot_parameters import PilotParameterExtractor,PilotMemory,predicted_concepts
from parameters.annotation_pilot import representation
from parameters.annotation_review import direct_fact_metrics
from parameters.image_claims import image_claim_predictions
from parameters.ue_profiles import load_ue_profiles


def claim(value,score):
    return {'value':value,'score':score,'threshold':.5,'method':'fixture_native_classifier','calibrated':False}


def extract(provenance):
    mission=load_ue_profiles()[0]['building_damage']
    evidence=VisualEvidence(embedding=(1.,0.),width=100,height=100,statuses={'detector':'ok'},provenance=provenance)
    extractor=PilotParameterExtractor()
    bundle=extractor.extract_evidence(evidence,mission,{'source_id':'u','image_token':'x'},PilotMemory('u',mission.mission_id))
    return representation(bundle,mission),evidence,bundle


def test_non_detection_and_raw_text_contrasts_cannot_be_native_absence_or_damage():
    output,_,_=extract({'condition_contrasts':{'building.affected_or_greater':{
        'positive_similarity':.7,'negative_similarity':.1}}})
    assert 'building.present' not in output['claims']
    assert 'building.affected_or_greater' not in output['claims']
    assert not output['claim_availability']['building.present']['available']
    assert output['parameters']['scene']['value']['condition_contrast_evidence']
    assert 'household.target_present' not in output['claims']


def test_image_classifier_can_supply_true_and_false_without_boxes():
    output,evidence,bundle=extract({'image_claim_predictions':{
        'building.present':claim(True,.9),'building.affected_or_greater':claim(False,.1)}})
    assert output['claims']['building.present'] is True
    assert output['claims']['building.affected_or_greater'] is False
    assert bundle.parameters['spatial'].value['target_count']==0 # Observed boxes only.
    assert output['claim_availability']['building.present']['available']
    assert predicted_concepts(evidence)=={'building'}
    memory=PilotMemory('u','building_damage');memory.commit(evidence,0)
    assert memory.history()[0]['concepts']==['building']


def test_malformed_or_inconsistent_classifier_decision_cannot_be_published():
    evidence=VisualEvidence(provenance={'image_claim_predictions':{
        'wrong':claim(True,.1),'nan':claim(True,float('nan')),'valid':claim(False,.2)}})
    assert set(image_claim_predictions(evidence))=={'valid'}


def test_abstention_cannot_improve_all_fact_score_by_hiding_mistakes():
    reference={'facts':{'building.present':True,'road.present':False}}
    metrics=direct_fact_metrics({'building.present':True},reference)
    assert metrics['accuracy']==1 and metrics['coverage']==.5
    assert metrics['correct_fraction_of_all_reference_facts']==.5
    empty=direct_fact_metrics({},reference)
    assert empty['accuracy'] is None and empty['coverage']==0
    assert empty['correct_fraction_of_all_reference_facts']==0


def test_repair_metrics_compare_class_balance_and_penalize_unknowns():
    from parameters.claim_repair_pilot import binary_metrics
    result=binary_metrics([True,True,False,False],[True,None,True,False])
    assert result['coverage']==.75 and result['accuracy_including_unknown']==.5
    assert result['balanced_accuracy_including_unknown']==.5 and result['f1']==.5
    with pytest.raises(ValueError):binary_metrics([True],[])


def test_classifier_outage_is_not_historical_negative_condition_evidence():
    mission=load_ue_profiles()[0]['building_damage'];memory=PilotMemory('u',mission.mission_id)
    old=VisualEvidence(embedding=(1.,0.),width=100,height=100,statuses={'detector':'ok'})
    memory.commit(old,0)
    new=VisualEvidence(embedding=(1.,0.),width=100,height=100,statuses={'detector':'ok'},
        provenance={'image_claim_predictions':{'building.affected_or_greater':claim(True,.9)}})
    extractor=PilotParameterExtractor()
    novelty=extractor.extract_novelty(new,mission,memory)
    assert novelty.value['distance']==0 and novelty.value['new_concept_fraction'] is None
    assert not novelty.supporting['concept_delta_available']
    assert not extractor.extract_scarcity(new,mission,memory).available
    memory.commit(new,1)
    assert extractor.extract_novelty(new,mission,memory).value['new_concept_fraction']==0
    assert extractor.extract_scarcity(new,mission,memory).supporting['valid_history_count']==1


def test_summary_preserves_all_reference_denominators():
    from parameters.annotation_pilot import summarize
    row={'view':'structured','review':{'status':'complete','parsed':{
        'parameters_consistent':True,'representation_sufficient':False}},'reference_grading_available':True,
        'direct_reference_metrics':direct_fact_metrics({'building.present':True},
            {'facts':{'building.present':True,'road.present':False}})}
    result=summarize([row])['structured']
    assert result['direct_reference_fact_count']==2 and result['direct_known_fact_comparisons']==1
    assert result['direct_reference_coverage']==.5 and result['direct_all_reference_correct_fraction']==.5


def test_shared_provider_can_keep_unvalidated_localization_explicitly_disabled():
    from PIL import Image
    import numpy as np
    from parameters.pilot_producers import PilotVisionProducer
    mission=load_ue_profiles()[0]['building_damage']
    class Global:
        provenance={'scope':'fixture_only'}
        def load(self):pass
        def _texts(self,texts):return np.array([[1.,0.] for text in texts])
        def extract_batch(self,images,missions):
            return [VisualEvidence(embedding=(1.,0.),width=i.width,height=i.height) for i in images]
    class Claims:
        def enrich(self,images,evidence):
            for item in evidence:item.provenance['image_claim_predictions']={'building.present':claim(True,.9)}
    producer=PilotVisionProducer('fixture',{mission.mission_id:mission},device='cpu',
        image_claim_provider=Claims(),enable_detector=False)
    producer.global_provider=Global()
    extractor=PilotParameterExtractor(provider=producer)
    bundle=extractor.extract(Image.new('RGB',(10,10)),mission,{'source_id':'u','image_token':'x'},PilotMemory('u',mission.mission_id))
    assert representation(bundle,mission)['claims']['building.present'] is True
    assert not bundle.parameters['spatial'].available and not bundle.parameters['complexity'].available
    assert bundle.parameters['rarity'].available


def test_model_load_produces_runtime_valid_provenance_with_string_class_ids(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from transformers import AutoImageProcessor,AutoModelForImageClassification
    from parameters.image_claims import NativeAerialClaimProvider,NATIVE_FACTS
    from parameters.contracts import finite_tree
    class Model:
        config=SimpleNamespace(id2label=dict(enumerate(NATIVE_FACTS)),model_type='bit')
        def to(self,device):return self
        def eval(self):return self
    monkeypatch.setattr(AutoImageProcessor,'from_pretrained',lambda *a,**k:SimpleNamespace(
        to_dict=lambda:{'size':{'shortest_edge':448},'do_center_crop':True}))
    monkeypatch.setattr(AutoModelForImageClassification,'from_pretrained',lambda *a,**k:Model())
    (tmp_path/'model.safetensors').write_bytes(b'fixture-only-not-model-weights')
    provider=NativeAerialClaimProvider(model_path=tmp_path);provider.load()
    assert finite_tree(provider.provenance)
    assert all(isinstance(key,str) for key in provider.provenance['labels'])
    assert provider.provenance['revision'] is None
    assert provider.provenance['training_exposure']=='unverified local checkpoint'
