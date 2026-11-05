import json
import pytest
from parameters.ue_profiles import load_ue_profiles,create_ue_memories
from parameters.extractor import SemanticParameterExtractor,VisualEvidence


def test_six_ues_share_five_fixed_missions_without_priority_weights():
    missions,bindings=load_ue_profiles()
    assert len(bindings)==6 and len(missions)==5
    first,second=bindings[:2]
    assert first.mission_id==second.mission_id=='building_damage'
    assert missions[first.mission_id].version==missions[second.mission_id].version
    assert 'structural_flooding' not in missions
    assert all(m.allowed_delay_ms is None for m in missions.values())
    assert all(len(m.conditions)==1 for m in missions.values())


def test_same_task_relevance_same_image_different_sender_novelty():
    missions,bindings=load_ue_profiles();memories=create_ue_memories(missions,bindings)
    mission=missions['building_damage'];extractor=SemanticParameterExtractor()
    evidence=VisualEvidence(embedding=(1.,0.),similarities={'building_damage':.6})
    a=memories[('ue01',mission.version)];b=memories[('ue02',mission.version)]
    a.commit(evidence,0)
    outputs=[extractor.extract_evidence(evidence,mission,
             {'image_token':'same-image','source_id':source},kb)
             for source,kb in (('ue01',a),('ue02',b))]
    assert outputs[0].parameters['relevance'].value==outputs[1].parameters['relevance'].value
    assert outputs[0].parameters['novelty'].value['distance']==0
    assert outputs[1].parameters['novelty'].value['cold_start']==1
    assert b.version==0


def test_same_image_different_mission_uses_matching_task_evidence():
    missions,bindings=load_ue_profiles();memories=create_ue_memories(missions,bindings)
    extractor=SemanticParameterExtractor()
    evidence=VisualEvidence(embedding=(1.,0.),similarities={'building_damage':.6,'household_inventory':-.2})
    values=[]
    for source,task in (('ue01','building_damage'),('ue05','household_inventory')):
        mission=missions[task]
        result=extractor.extract_evidence(evidence,mission,
             {'image_token':'fixture','source_id':source},memories[(source,mission.version)])
        values.append(result.parameters['relevance'].value)
    assert values==pytest.approx([.8,.4]) # Supplied similarities; not measured CLIP accuracy.


def test_registry_rejects_duplicate_ues_and_unknown_missions(tmp_path):
    from pathlib import Path
    original=Path(__file__).resolve().parents[1]/'config/ue_task_profiles.json'
    data=json.loads(original.read_text())
    data['ues'].append(dict(data['ues'][0]));path=tmp_path/'bad.json';path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='duplicate UE'):load_ue_profiles(path)
    data['ues'].pop();data['ues'][0]['mission_id']='invented';path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='unknown mission'):load_ue_profiles(path)
