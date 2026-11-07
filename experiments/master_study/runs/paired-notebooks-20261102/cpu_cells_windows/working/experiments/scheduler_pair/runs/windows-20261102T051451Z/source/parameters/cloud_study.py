"""Nested grouped parameter selection on cached frozen visual features."""
from dataclasses import replace
from pathlib import Path
import itertools
import json
import math
import time
import numpy as np

from .artifacts import Run,write_json
from .contracts import GROUPS,MISSIONS,MethodSpecification,ParameterValue,flatten_bundle,digest
from .calibration import fit_fold_heads
from .evaluation import grouped_split,feature_subsets,training_decisions,replay
from .study import episodes_for,score_episodes,choose_configuration
from .learning import fit_scorer
from .extractor import VisualEvidence
from .serialization import HeaderCodec
from Evaluation.decoder_semantic_benchmark import coverage_metrics

KNOWN=('building_damage','road_damage')
BASELINES=('fifo','round_robin','random','relevance','novelty','incident_filter','historical_I')


def project_episode(episode,specification,codec=None):
    observations=[]
    for observation in episode.observations:
        parameters={k:(v if k in specification.enabled_groups or k=='reliability' else ParameterValue(None,False,'disabled','disabled by specification')) for k,v in observation.bundle.parameters.items()}
        bundle=replace(observation.bundle,parameters=parameters,specification_sha256=specification.sha256)
        if codec is not None:
            decoded=codec.decode(codec.encode(bundle))['values']
            rebuilt={}
            for group,result in parameters.items():
                fields={k[len(group)+1:]:v for k,v in decoded.items() if k.startswith(group+'.')}
                if group in decoded:rebuilt[group]=replace(result,value=decoded[group],available=True)
                elif fields:
                    nested={}
                    for key,value in fields.items():
                        cursor=nested;parts=key.split('.')
                        for part in parts[:-1]:cursor=cursor.setdefault(part,{})
                        cursor[parts[-1]]=value
                    rebuilt[group]=replace(result,value=nested,available=True)
                elif group=='reliability':rebuilt[group]=result
                else:rebuilt[group]=ParameterValue(None,False,'wire_mask','no transmitted available field')
            bundle=replace(bundle,parameters=rebuilt)
        observations.append(replace(observation,bundle=bundle))
    return replace(episode,observations=tuple(observations))


def macro_score(episodes,policy):
    results={mission:score_episodes([e for e in episodes if e.mission==mission],policy) for mission in KNOWN}
    # Never let an unavailable comparator improve its mean by dropping queues.
    for mission,result in results.items():
        offered=[e for e in episodes if e.mission==mission]
        if not offered or len(result['rows'])!=len(offered):
            return None,results
    values=[v['mean_A10'] for v in results.values() if v['mean_A10'] is not None]
    return None if len(values)!=len(KNOWN) else float(np.mean(values)),results


def prepare_fold(train,validation,raw,backbone,relevance,novelty,seed=7):
    heads,calibration,consumer_split=fit_fold_heads(train,raw,seed)
    enriched=heads.apply(raw)
    head_artifact={**heads.to_dict(),'calibrated':calibration}
    producer=next(iter(raw.values())).provenance if raw else {}
    spec=MethodSpecification(backbone=backbone,relevance=relevance,novelty=novelty,
            calibration={k:v for k,v in calibration.items() if v.get('available')},
            producer_revisions={'claim_heads_sha256':digest(head_artifact),
                                'image_weights_sha256':producer.get('weights_sha256'),
                                'image_model_revision':producer.get('model_revision')})
    train_episodes=[];validation_episodes=[];missing=[]
    for mission in KNOWN:
        episodes,none=episodes_for(train,enriched,mission,spec,count=3,seed=seed)
        train_episodes+=episodes;missing+=none
        episodes,none=episodes_for(validation,enriched,mission,spec,count=5,seed=10000+seed)
        validation_episodes+=episodes;missing+=none
    decisions=training_decisions(train_episodes)
    return spec,head_artifact,enriched,decisions,validation_episodes,missing,consumer_split


def inner_search(records,caches,output,run,seed=7):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    folds=grouped_split(records,3,seed);rows=[];errors=[]
    for fold,(train,validation) in enumerate(folds):
        run.budget();fold_rows=[];fold_dir=output/('inner_'+str(fold));fold_dir.mkdir(exist_ok=True)
        for backbone,raw in caches.items():
            for relevance,novelty in itertools.product(('similarity','claim_probability'),('embedding','semantic_delta')):
                run.budget();key=backbone+'__'+relevance+'__'+novelty
                spec,heads,enriched,decisions,heldout,missing,consumer_split=prepare_fold(train,validation,raw,backbone,relevance,novelty,seed+fold)
                write_json(fold_dir/(key+'_consumers.json'),consumer_split)
                if not decisions or not heldout:
                    errors.append({'fold':fold,'method':key,'reason':'insufficient supported episodes'});continue
                available=tuple(g for g in GROUPS if any(k.split('.')[0]==g for d in decisions for o in d['ready'] for k in flatten_bundle(o.bundle)))
                for subset in feature_subsets(available):
                    for alpha in (.1,1.,10.):
                        run.budget()
                        try:
                            policy,fit=fit_scorer(decisions,subset,alpha=alpha,provenance={'inner_fold':fold,'consumer_groups':consumer_split})
                            score,_=macro_score(heldout,policy)
                            row={'fold':fold,'method':key,'backbone':backbone,'relevance':relevance,'novelty':novelty,
                                 'groups':list(subset),'alpha':alpha,'family':'linear','mean_A10':score,
                                 'header_bytes':HeaderCodec(policy.encoder.fields,spec.sha256).describe()['header_bytes'],**fit}
                            fold_rows.append(row)
                        except ValueError as error:errors.append({'fold':fold,'method':key,'groups':subset,'reason':str(error)})
                    if len(fold_rows)%30==0:
                        write_json(fold_dir/'grid.json',fold_rows)
                        print('Inner fold',fold,'method',key,'completed configurations',len(fold_rows),flush=True)
                write_json(fold_dir/'grid.json',fold_rows)
        rows+=fold_rows
    aggregated=[]
    keys={(r['method'],tuple(r['groups']),r['alpha']) for r in rows}
    for method,groups,alpha in sorted(keys):
        values=[r for r in rows if r['method']==method and tuple(r['groups'])==groups and r['alpha']==alpha and r['mean_A10'] is not None]
        if len({r['fold'] for r in values})!=3:continue
        scores=np.array([r['mean_A10'] for r in values]);prototype=values[0]
        aggregated.append({**{k:prototype[k] for k in ('method','backbone','relevance','novelty','groups','alpha','family')},
                           'mean_A10':float(scores.mean()),'standard_error':float(scores.std(ddof=1)/np.sqrt(3)),
                           'header_bytes':max(r['header_bytes'] for r in values),'fold_scores':scores.tolist()})
    # Fixed nonlinear check only on core/full and three selected distinct groups.
    top=sorted(aggregated,key=lambda r:-r['mean_A10'])
    if top:
        method=top[0]['method'];candidates=[r for r in top if r['method']==method]
        subset_candidates=[]
        full=tuple(g for g in GROUPS if any(g in r['groups'] for r in candidates))
        top_distinct=[]
        for row in candidates:
            if tuple(row['groups']) not in top_distinct:top_distinct.append(tuple(row['groups']))
            if len(top_distinct)==3:break
        for subset in [tuple(g for g in ('relevance','novelty') if g in full),full]+top_distinct:
            if subset not in subset_candidates:subset_candidates.append(subset)
        for subset in subset_candidates:
            for depth,leaf in itertools.product((2,3),(5,10)):
                scores=[];header_sizes=[]
                for fold,(train,validation) in enumerate(folds):
                    run.budget();prototype=top[0]
                    spec,heads,enriched,decisions,heldout,missing,consumers=prepare_fold(train,validation,caches[prototype['backbone']],prototype['backbone'],prototype['relevance'],prototype['novelty'],seed+fold)
                    try:
                        policy,fit=fit_scorer(decisions,subset,kind='tree',depth=depth,min_leaf=leaf)
                        score,_=macro_score(heldout,policy)
                    except ValueError:score=None
                    if score is not None:scores.append(score);header_sizes.append(HeaderCodec(policy.encoder.fields,spec.sha256).describe()['header_bytes'])
                if len(scores)==3:
                    aggregated.append({**{k:top[0][k] for k in ('method','backbone','relevance','novelty')},
                        'groups':list(subset),'alpha':0.,'family':'tree','depth':depth,'min_leaf':leaf,
                        'mean_A10':float(np.mean(scores)),'standard_error':float(np.std(scores,ddof=1)/np.sqrt(3)),
                        'header_bytes':max(header_sizes),'fold_scores':scores})
    selected=choose_configuration(aggregated)
    write_json(output/'all_grid_rows.json',rows);write_json(output/'aggregated_grid.json',aggregated)
    write_json(output/'errors_and_missing.json',errors);write_json(output/'selected.json',selected)
    return selected,aggregated


def refit_evaluate(train,heldout,caches,chosen,output,seed=7):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    spec,heads,enriched,decisions,episodes,missing,consumer_split=prepare_fold(train,heldout,caches[chosen['backbone']],chosen['backbone'],chosen['relevance'],chosen['novelty'],seed)
    policy,fit=fit_scorer(decisions,tuple(chosen['groups']),alpha=chosen['alpha'],kind=chosen['family'],
                          depth=chosen.get('depth',2),min_leaf=chosen.get('min_leaf',5),provenance={'consumer_split':consumer_split})
    spec=replace(spec,enabled_groups=tuple(chosen['groups']),feature_names=policy.encoder.fields)
    full=[project_episode(e,spec) for e in episodes]
    score,methods=macro_score(full,policy)
    codecs={bits:HeaderCodec(policy.encoder.fields,replace(spec,quantization_bits=bits).sha256,bits) for bits in (8,16)}
    quantized={};wire_episodes={}
    for bits,codec in codecs.items():
        qspec=replace(spec,quantization_bits=bits)
        wire_episodes[bits]=[project_episode(e,qspec,codec) for e in episodes]
        qscore,qmethods=macro_score(wire_episodes[bits],policy)
        quantized[str(bits)]={'mean_A10':qscore,'difference_from_full':None if score is None or qscore is None else qscore-score,
                             'header':codec.describe(),'orders_changed':sum(replay(a,policy)['order']!=replay(b,policy)['order'] for a,b in zip(full,wire_episodes[bits]))}
    # Parameter-pipeline comparators retain their own producer outputs. Also
    # report the stricter comparison on the selected transmitted fields.
    baselines={name:macro_score(episodes,name)[0] for name in BASELINES}
    same_header_baselines={name:macro_score(full,name)[0] for name in BASELINES}
    write_json(output/'selected_method.json',spec.to_dict());write_json(output/'claim_heads.json',heads)
    write_json(output/'selected_scorer.json',policy.artifact);write_json(output/'header_registry.json',codecs[8].describe())
    write_json(output/'consumer_split.json',consumer_split);write_json(output/'quantization.json',quantized)
    write_json(output/'results.json',{'mean_A10':score,'baselines':baselines,
               'baseline_information':'own unquantized parameter pipeline; not same selected header',
               'same_header_baselines':same_header_baselines,'methods':methods,'missing':missing,'fit':fit})
    return {'score':score,'baselines':baselines,'quantization':quantized,'spec':spec,'heads':heads,
            'policy':policy,'episodes':episodes,'wire_episodes':wire_episodes,'missing':missing}


def paired_group_interval(episodes,policy,baseline,atom_groups,repetitions=1000,reference_episodes=None):
    """Reweight fact clusters in paired fixed queues; not a bootstrap of queue seeds."""
    groups=sorted({group for episode in episodes for group in episode.group_ids})
    if len(groups)<10:return {'available':False,'reason':'fewer than10 known groups; descriptive only'}
    reference_episodes=episodes if reference_episodes is None else reference_episodes
    if len(episodes)!=len(reference_episodes) or any(
            [o.token for o in a.observations]!=[o.token for o in b.observations]
            for a,b in zip(episodes,reference_episodes)):
        raise ValueError('paired policies must receive identical candidate identities and order')
    predicted=[replay(e,policy) for e in episodes];reference=[replay(e,baseline) for e in reference_episodes]
    if any(r is None for r in reference):return {'available':False,'reason':'baseline unavailable on same queues'}
    rng=np.random.default_rng(19);deltas=[]
    for _ in range(repetitions):
        counts=dict(zip(groups,rng.multinomial(len(groups),np.full(len(groups),1/len(groups)))))
        per_task={}
        for ep,a,b in zip(episodes,predicted,reference):
            weights={f:w*counts.get(atom_groups.get(f),0) for f,w in ep.weights.items()}
            ca=coverage_metrics(a['order'],ep.facts,weights,k=10)['A_K'];cb=coverage_metrics(b['order'],ep.facts,weights,k=10)['A_K']
            if ca is not None:per_task.setdefault((ep.mission,ep.arm),[]).append(ca-cb)
        if per_task:deltas.append(float(np.mean([np.mean(values) for values in per_task.values()])))
    return {'available':bool(deltas),'lower':float(np.quantile(deltas,.025)) if deltas else None,
            'upper':float(np.quantile(deltas,.975)) if deltas else None,'groups':len(groups),'bootstrap_repetitions':len(deltas),
            'unit':'highest known duplicate/parent component; event IDs unavailable',
            'scope':'conditional fixed-queue parent-cluster uncertainty; not event-level generalization'}


def select_campaign(records,caches,output,seconds=7200,exposed_ids=None):
    if exposed_ids is None:
        raise ValueError('explicit historical/pilot exposure ledger required; fresh-final selection fails closed')
    run=Run(output,{'stage':'selection','seconds':seconds,'known_missions':KNOWN,'final_fraction':.2,
                    'outer_folds':3,'inner_folds':3,'selection_rule':'max(.005,one standard error) then smaller configuration'})
    excluded=set(exposed_ids)
    exposed_groups={r['group'] for r in records if r['image_id'] in excluded or r.get('development_exposed',False)}
    eligible=[r for r in records if r['group'] not in exposed_groups]
    groups=sorted({r['group'] for r in eligible});rng=np.random.default_rng(17);rng.shuffle(groups)
    final_groups=set(groups[:len(groups)//5])
    development=[r for r in records if r['group'] not in final_groups]
    final=[r for r in records if r['group'] in final_groups]
    write_json(run.path/'data/split_manifest.json',{'development_groups':sorted({r['group'] for r in development}),
              'final_groups':sorted(final_groups),'exposed_ids_excluded_from_final':len(excluded),'event_grouping_verified':False})
    outer_results=[]
    with run.stage('01_nested_grouped_selection'):
        for outer,(train,validation) in enumerate(grouped_split(development,3,7)):
            selected,grid=inner_search(train,caches,run.path/'metrics'/('outer_'+str(outer))/'selection',run,seed=100+outer)
            if selected is None:raise RuntimeError('inner selection has insufficient supported groups/labels')
            result=refit_evaluate(train,validation,caches,selected,run.path/'metrics'/('outer_'+str(outer))/'evaluation',seed=1000+outer)
            outer_results.append({'fold':outer,'selected':selected,'mean_A10':result['score'],'baselines':result['baselines']})
            write_json(run.path/'metrics/outer_results.json',outer_results)
            print('Outer fold',outer,'completed A10',result['score'],flush=True)
    with run.stage('02_freeze_configuration'):
        selected,grid=inner_search(development,caches,run.path/'metrics/final_selection',run,seed=777)
        if selected is None:raise RuntimeError('cannot freeze unsupported configuration')
        baseline_scores={name:np.mean([r['baselines'][name] for r in outer_results if r['baselines'][name] is not None])
                         for name in BASELINES if all(r['baselines'][name] is not None for r in outer_results)}
        baseline=max(baseline_scores,key=baseline_scores.get)
        write_json(run.path/'models/selection_freeze.json',{'configuration':selected,'baseline':baseline,'baseline_development_scores':baseline_scores,
                   'final_labels_used_for_selection':False,'flooding_scheduler_labels_used_for_selection':False})
    with run.stage('03_locked_final'):
        if not final:
            write_json(run.path/'status.json',{'status':'partial','reason':'no eligible fresh final groups'});return run.path
        result=refit_evaluate(development,final,caches,selected,run.path/'models/frozen',seed=12345)
        # Quantization choice is made from outer development results, not final labels.
        bits=8
        for row in outer_results:
            quant=json.loads((run.path/'metrics'/('outer_'+str(row['fold']))/'evaluation/quantization.json').read_text())
            if quant['8']['difference_from_full'] is not None and quant['8']['difference_from_full']<-.005:bits=16
        spec=replace(result['spec'],quantization_bits=bits)
        codec=HeaderCodec(result['policy'].encoder.fields,spec.sha256,bits)
        episodes=[project_episode(e,spec,codec) for e in result['episodes']]
        score,methods=macro_score(episodes,result['policy']);reference=macro_score(result['episodes'],baseline)[0]
        atom_groups={}
        by_parent={r['parent_id']:r['group'] for r in records}
        for ep in episodes:
            for atom in ep.weights:atom_groups[atom]=by_parent.get(atom.split('|')[1])
        interval=paired_group_interval(episodes,result['policy'],baseline,atom_groups,
                                       reference_episodes=result['episodes'])
        delta=None if score is None or reference is None else score-reference
        supported_gain=delta is not None and delta>=.01 and interval.get('available') and interval.get('lower',-1)>0
        write_json(run.path/'models/frozen/selected_method.json',spec.to_dict());write_json(run.path/'models/frozen/header_registry.json',codec.describe())
        write_json(run.path/'metrics/locked_final.json',{'A10':score,'baseline':baseline,'baseline_A10':reference,'delta':delta,
                   'paired_group_interval':interval,'gain_criterion_met':bool(supported_gain),'methods':methods,'wire_bits_per_score':bits})
        # Scheduler-unseen mission, using the same frozen producers/spec/scorer.
        transfer=[]
        for mission in ('affected_people','structural_flooding'):
            if mission=='structural_flooding' and spec.relevance=='claim_probability':
                # No flooding head was fitted: retain missingness, do not train/calibrate on target truth.
                pass
            # Recreate producer predictions with frozen saved heads; no target fitting.
            from .learning import ClaimHeads
            enriched=ClaimHeads.from_dict(result['heads']).apply(caches[selected['backbone']])
            candidate,_=episodes_for(final,enriched,mission,spec,count=5,seed=654)
            for ep in candidate:
                wire_ep=project_episode(ep,spec,codec);transfer.append({'mission':mission,'result':replay(wire_ep,result['policy'])})
        write_json(run.path/'metrics/transfer_stress.json',transfer)
        decisions={group:{'status':'retained' if group in spec.enabled_groups else 'not_selected_or_insufficient_evidence',
                         'interpretation':'selection result on supported observation/copy arms; not universal importance'} for group in GROUPS}
        write_json(run.path/'models/parameter_decisions.json',decisions)
        report=['# parameters cloud selection','',f'Frozen groups: {spec.enabled_groups}.',
                f'Final A10: {score}; baseline {baseline}: {reference}; delta {delta}.',
                f'Header: {codec.describe()["header_bytes"]} bytes with {bits}-bit scores.',
                f'Predeclared gain criterion met: {bool(supported_gain)}.',
                'Native observations and exact copies do not establish natural cross-view asset matching.',
                'Known groups are parent/duplicate components; event/site identities are unavailable.',
                'Frozen vision models may already recognize flooding. This is scheduler-mission transfer only.',
                'No individual-distress, operational-urgency, actual-codec or end-to-end wireless performance claim.']
        (run.path/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
        write_json(run.path/'status.json',{'status':'complete','seconds':time.monotonic()-run.started,'gain_criterion_met':bool(supported_gain)})
    return run.path
