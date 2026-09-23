"""Worker factory returns a sequence of BenchmarkCodec objects from a versioned source tree."""
import argparse
import importlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request',required=True,type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding='utf-8'))
    from INFRA.dependencies import verify_revision
    from INFRA.runs import provenance, seed_models, write_json
    from EncDec.Evaluation.kodak_benchmark import append_rows_csv, dataset_files
    from EncDec.Evaluation.audited_benchmark import benchmark_with_outcomes
    verify_revision(request['source_root'],request['revision'])
    # Isolated process owns this import path and dies when the benchmark ends.
    sys.path.insert(0,request['source_root'])
    module_name,function_name = request['factory'].split(':')
    factory = getattr(importlib.import_module(module_name),function_name)
    config = request['config']
    output = Path(request['output'])
    seed_models(config['seed'])
    write_json(output/'worker_provenance.json',provenance())
    files = dataset_files(Path(config['paths']['kodak']))
    if config['action']=='smoke':
        files=files[:1]
    produced = factory(config)
    if isinstance(produced, tuple) and len(produced) == 2 and isinstance(produced[1], dict):
        codecs, factory_provenance = produced
        write_json(output/'factory_provenance.json',factory_provenance)
    else:
        codecs = produced
    outcomes=[]
    for codec in codecs:
        rows, status = benchmark_with_outcomes(codec,files,run_id=request['run_id'],project_root=Path(__file__).resolve().parents[1],
            timing_repetitions=config['benchmark']['timing_repetitions'],compute_lpips=config['benchmark']['compute_lpips'])
        append_rows_csv(output/'benchmark.csv',rows)
        outcomes.extend({'method':codec.name,**item} for item in status)
    if not outcomes:
        raise ValueError('Factory produced no benchmark codecs')
    write_json(output/'worker_result.json',{'outcomes':outcomes,'failed':sum(item['status']=='failed' for item in outcomes),'external_revision':request['revision']})

if __name__=='__main__':
    main()
