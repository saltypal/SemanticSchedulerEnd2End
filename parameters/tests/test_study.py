from dataclasses import replace
import copy
import numpy as np
import pytest
from parameters.contracts import MISSIONS,MethodSpecification,ParameterValue
from parameters.extractor import VisualEvidence,SemanticParameterExtractor,SenderMemory
from parameters.evaluation import Observation,Episode,replay,training_decisions,FeatureEncoder,grouped_split
from parameters.learning import fit_scorer,ClaimHeads,Scorer
from parameters.study import choose_configuration
from Evaluation.decoder_semantic_benchmark import coverage_metrics


def episode():
    extractor=SemanticParameterExtractor(MethodSpecification(backbone='fixture'))
    observations=[]
    for i,score in enumerate((.9,.8,.7,.1)):
        evidence=VisualEvidence(embedding=(1.,i/4),similarities={'building_damage':2*score-1})
        bundle=extractor.extract_evidence(evidence,MISSIONS['building_damage'],
                    {'source_id':'u','image_token':str(i)},SenderMemory('u','building_damage'))
        observations.append(Observation(str(i),i%2,bundle,score))
    facts={'0':{'b1'},'1':{'b1'},'2':{'b2'},'3':set()}
    return Episode(tuple(observations),facts,{'b1':1.,'b2':1.},'building_damage','exact_copy',('g1','g2','g3'),7)


def test_independent_coverage_reference_and_runtime_truth_boundary():
    ep=episode()
    assert coverage_metrics(['0','1','2','3'],ep.facts,ep.weights,k=3)['A_K']==pytest.approx(2/3)
    assert coverage_metrics(['0','2','1','3'],ep.facts,ep.weights,k=3)['A_K']==pytest.approx(5/6)
    decisions=training_decisions([ep],cutoff=3);encoder=FeatureEncoder.fit(decisions,('relevance',))
    matrix=encoder.matrix(ep.observations,(),0,4)
    changed=replace(ep,facts={k:set() for k in ep.facts})
    assert np.array_equal(matrix,encoder.matrix(changed.observations,(),0,4))
    assert replay(ep,'relevance')['order']==replay(changed,'relevance')['order']


@pytest.mark.parametrize('kind',['linear','tree'])
def test_real_fit_portable_export_and_history_ablation(kind):
    ep=episode();decisions=training_decisions([ep]*2,cutoff=3)
    policy,fit=fit_scorer(decisions,('relevance',),kind=kind,min_leaf=2)
    assert fit['export_max_error']<1e-7
    assert all(key.startswith('relevance') for key in policy.encoder.fields)
    before=policy.scores(ep.observations,(),0,4)
    changed=[]
    for obs in ep.observations:
        parameters=dict(obs.bundle.parameters);parameters['novelty']=ParameterValue({'distance':.99},True,'fixture')
        changed.append(replace(obs,bundle=replace(obs.bundle,parameters=parameters)))
    assert np.allclose(before,policy.scores(changed,(),0,4))
    bad=copy.deepcopy(policy.artifact);bad['model']['intercept']=float('nan')
    with pytest.raises((ValueError,TypeError)):Scorer(bad)


def test_grouped_split_never_separates_derivatives():
    records=[{'image_id':str(i),'group':'parent'+str(i//2)} for i in range(12)]
    for train,validation in grouped_split(records):
        assert not {r['group'] for r in train}&{r['group'] for r in validation}


def test_head_training_does_not_read_heldout_labels():
    records=[{'image_id':str(i),'labels':{'building_damage':int(i%2)}} for i in range(24)]
    evidence={str(i):VisualEvidence(embedding=np.array([i%2,1.])) for i in range(24)}
    head=ClaimHeads().fit(records[:20],evidence)
    a=head.apply(evidence)['23'].probabilities['building_damage']
    for record in records[20:]:record['labels']['building_damage']=1-record['labels']['building_damage']
    other=ClaimHeads().fit(records[:20],evidence)
    assert other.apply(evidence)['23'].probabilities['building_damage']==a
    reloaded=ClaimHeads.from_dict(head.to_dict())
    assert reloaded.apply(evidence)['23'].probabilities['building_damage']==pytest.approx(a)


def test_selection_prefers_simpler_near_best():
    rows=[{'groups':['relevance','novelty'],'mean_A10':.70,'alpha':.1},
          {'groups':['relevance'],'mean_A10':.697,'alpha':1.}]
    assert choose_configuration(rows)['groups']==['relevance']


def test_quantized_replay_can_change_ties_and_must_be_measured():
    from parameters.cloud_study import project_episode
    from parameters.serialization import HeaderCodec
    spec=MethodSpecification(backbone='fixture',enabled_groups=('relevance',),feature_names=('relevance',))
    extractor=SemanticParameterExtractor(spec);rows=[]
    for token,value in [('a',.501),('b',.502)]:
        bundle=extractor.extract_evidence(VisualEvidence(similarities={'building_damage':2*value-1}),MISSIONS['building_damage'],
                                         {'source_id':'u','image_token':token},SenderMemory('u','building_damage'))
        rows.append(Observation(token,0,bundle))
    ep=Episode(tuple(rows),{'a':set(),'b':{'useful'}},{'useful':1.},'building_damage','native',('ga','gb'),7)
    wire=project_episode(ep,spec,HeaderCodec(spec.feature_names,spec.sha256))
    assert replay(ep,'relevance')['order']==['b','a']
    assert replay(wire,'relevance')['order']==['a','b']
    assert replay(ep,'relevance')['metrics']['A_K']!=replay(wire,'relevance')['metrics']['A_K']
