"""Validation-only test of one hypothesis: defer exact repeats before learned choice."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def make_hybrid(base,fields):
    from Scheduler.decoder_scheduler import Decision
    from Scheduler.scene_policy import evidence_signature,tie_key
    import numpy as np

    class RepeatFilteredScorer:
        profile_id=None
        kind=base.kind+'_repeat_filtered'
        def reset(self):base.reset()
        @property
        def fallback_count(self):return base.fallback_count
        def select(self,context):
            rows=context.ready_candidates
            if not rows:return None
            scores=base.scores(context)
            if not np.isfinite(scores).all():return base.select(context)
            seen={s for h in context.successful_header_history
                  if (s:=evidence_signature(h,fields)) is not None}
            # Score the original full context; filtering must not change queue means.
            i=min(range(len(rows)),key=lambda j:(evidence_signature(rows[j],fields) in seen,
                                                -scores[j],tie_key(rows[j])))
            return Decision(rows[i].message_id,base.kind+'_repeat_filtered',float(scores[i]))
    return RepeatFilteredScorer()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    sys.path.insert(0,str(a.source.resolve()))
    from Scheduler.training.scene_corpus import CausalCorpus
    from Scheduler.training.scene_study import evaluate
    from Scheduler.training.scene_evaluation import build_choice_probes,evaluate_choices,summarize_rollouts
    from Scheduler.scene_policy import ScenePolicy,evidence_signature
    corpus=CausalCorpus(a.data,a.source/'upstream')
    fields=tuple(i for i,name in enumerate(corpus.registry.fields) if name.startswith(('claim/','detection/')))
    path=a.study/'algorithms/ranknet/model.json';artifact=json.loads(path.read_text())
    candidates={'ranknet':ScenePolicy(artifact),'ranknet_repeat_filtered':make_hybrid(ScenePolicy(artifact),fields)}
    probes=build_choice_probes(corpus,traces=32)
    result={'scope':'Post-audit development hypothesis, no training or pilot evaluation',
        'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'data_sha256':{name:hashlib.sha256((a.data/name).read_bytes()).hexdigest()
                       for name in ('records.json','features.jsonl')},
        'snapshot_source':str(a.source.resolve()),'history_limit':32,'results':{}}
    for name,policy in candidates.items():
        score,details=evaluate(corpus,'validation',policy,traces=32)
        summary=summarize_rollouts(details);choice=evaluate_choices(policy,probes)
        result['results'][name]={'J':score,'summary':summary,'choices':choice}
        (a.output/(name+'_traces.json')).write_text(json.dumps(details,indent=2,allow_nan=False))
        print(name,'J',score,'distinct_useful@10',summary['distinct_useful_at_completions']['10']['mean'],
              'shared_choice_accuracy',choice['tie_aware_top1_accuracy'],flush=True)
    signatures={}
    # Evaluator-only collision audit; payload/image IDs never enter policy inputs.
    for index in range(32):
        env=corpus.make('validation',41000+index,pattern=('no_repeat','local_repeat','cross_repeat')[index%3],
                        load=('light','congested','burst')[(index//3)%3])
        from Evaluation.scene_queue import strict_json
        for message in env.episode.messages:
            values,mask,cold=corpus.registry.extract(strict_json(message.header_bytes))
            from Scheduler.scene_policy import SceneCandidate
            candidate=SceneCandidate(message.message_id,message.source_id,message.arrival_time,
                message.sequence,values,mask,cold,corpus.registry.sha256)
            signature=evidence_signature(candidate,fields)
            if signature is not None:signatures.setdefault(signature,set()).add(message.payload_ref)
    result['distinct_validation_images']=len(set().union(*signatures.values()))
    result['exact_signatures_shared_by_different_images']=sum(len(v)>1 for v in signatures.values())
    result['seconds']=time.perf_counter()-start
    (a.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('END hybrid hypothesis',result['seconds'],'seconds; cross-image signature collisions',
          result['exact_signatures_shared_by_different_images'],flush=True)


if __name__=='__main__':main()
