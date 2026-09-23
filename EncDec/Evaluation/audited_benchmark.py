"""Preserve CSV schema while making every failure and missing stage explicit."""
from datetime import datetime, timezone
from PIL import Image
from EncDec.Evaluation.kodak_benchmark import CSV_COLUMNS, SCHEMA_VERSION, benchmark_codec

def benchmark_with_outcomes(codec, files, *, run_id, project_root, timing_repetitions=1, compute_lpips=False):
    rows, outcomes = [], []
    for index, path in enumerate(files):
        stages = {'extraction_time_ms':None,'encoding_time_ms':None,'channel_time_ms':None,
                  'decoding_time_ms':None,'scheduling_time_ms':None,'latent_memory_bytes':None}
        try:
            row = benchmark_codec(codec,[path],run_id=run_id,contributor='modular-runtime',project_root=project_root,
                timing_repetitions=timing_repetitions,warmup_repetitions=1 if index == 0 else 0,
                compute_lpips=compute_lpips,notes='status=success')[0]
            stages['encoding_time_ms'] = row['encode_time_ms']
            if not getattr(codec,'transport_fn',None):
                stages['decoding_time_ms'] = row['decode_time_ms']
            stages.update(getattr(codec,'last_measurements',{}))
            outcome = {'image_id':path.stem,'status':'success','measurements':stages}
        except Exception as error:
            row = {column:None for column in CSV_COLUMNS}
            with Image.open(path) as image:
                width,height = image.size
            row.update(schema_version=SCHEMA_VERSION,run_id=run_id,timestamp_utc=datetime.now(timezone.utc).isoformat(),
                contributor='modular-runtime',method=codec.name,method_version=codec.version,operating_point=codec.operating_point,
                device=codec.device,image_id=path.stem,width=width,height=height,source_bits=width*height*24,
                timing_repetitions=timing_repetitions,notes=f'status=failed; {type(error).__name__}: {error}')
            outcome = {'image_id':path.stem,'status':'failed','error':f'{type(error).__name__}: {error}','measurements':stages}
        rows.append(row)
        outcomes.append(outcome)
    return rows, outcomes
