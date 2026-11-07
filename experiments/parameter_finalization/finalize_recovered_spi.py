"""Recover a truthful readiness decision from verified saved v13 training artifacts."""
import hashlib,json
from pathlib import Path
from parameters.spi_release import candidate_release
from parameters.spi_model_selection import support_gate
from parameters.artifacts import write_json

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'experiments/parameter_finalization/runs/notebook9_20261028'
BASE=RUN/'submission_v13_features/retrieved_version13/experiments/parameter_finalization/runs/notebook9_20261028_v13/data_run'
OUTPUT=RUN/'recovered_readiness_v13_complete'

def main():
 OUTPUT.mkdir(exist_ok=False)
 model=BASE/'spi_classifier/models/spi.json';validation=BASE/'spi_classifier/metrics/validation.json';labels_path=BASE/'spi_training/labels.json'
 receipt=candidate_release(model,validation,ROOT/'parameters/config/parameter_release_v2.json',OUTPUT/'release_candidate.json',root=BASE)
 labels=json.loads(labels_path.read_text());support=support_gate(labels)
 receipt.update(class_support=support['details'],teacher_selection=json.loads((ROOT/'parameters/config/final_labeling_teacher.json').read_text()),reference_type='teacher_agreement',independent_urgency_validated=False,
  diagnostic_review={'accepted':False,'unresolved_decisive_contradictions':None,'reason':'User-selected pseudo-label teacher; no independent urgency validation.'})
 write_json(OUTPUT/'cache_readiness.json',receipt)
 report=json.loads(validation.read_text())
 result={'status':'not_ready','runtime_promotion_allowed':receipt['runtime_promotion_allowed'],'release_checks':receipt['release_checks'],
  'label_count':len(labels),'class_counts':{name:sum(r.get('level')==name and r.get('available') is True for r in labels) for name in ('Low','Medium','High','Critical')},
  'validation':report['student'],'recovery_scope':'same saved model and validation, no refitting or relabeling',
  'input_hashes':{str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (model,validation,labels_path)}}
 write_json(OUTPUT/'decision.json',result)
 if receipt['runtime_promotion_allowed']:raise RuntimeError('Unexpected passing receipt; inspect before any further work')
 print(json.dumps(result,indent=2))

if __name__=='__main__':main()
