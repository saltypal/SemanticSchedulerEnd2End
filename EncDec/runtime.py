"""Module entry points using one benchmark contract for codecs."""
from pathlib import Path

def build_codec(config):
    from EncDec.Evaluation.kodak_benchmark import JPEGCodec, RawRGBCodec, EncDecCodec
    from INFRA.runs import select_device
    name = config['implementation']
    options = config['benchmark']
    if name == 'raw':
        return RawRGBCodec()
    if name == 'jpeg':
        return JPEGCodec(options['jpeg_quality'])
    if name == 'swinjscc':
        from EncDec.Models.SwinJSCC.model_loader import load_stage1_swinjscc
        from INFRA.Channels.channel_factory import create_channel
        channel = create_channel(config['transport']['name'])
        device = select_device(config['device'])
        adapter = load_stage1_swinjscc(config['paths']['model_dir'],device,config['paths']['upstream_swin'])
        for model in adapter._require_built():
            model.eval()
        from INFRA.hashing import file_sha256
        identity = file_sha256(Path(config['paths']['model_dir'])/'stage1_swinjscc_full.pt')
        codec = EncDecCodec('SwinJSCC',adapter,device=device,version=identity,
            operating_point=f"{channel.name};snr_db={options['snr_db']};rate={options['rate']}",
            encode_options={'snr_db':options['snr_db'],'rate':options['rate']},
            transport_fn=lambda latent: channel.transmit(latent, options['snr_db']))
        return codec
    if name == 'deepjscc':
        from INFRA.dependencies import verify_revision
        from EncDec.Models.DeepJSCC.adapter import DeepJSCCAdapter
        dependency = config.get('dependencies',{}).get('deepjscc',{})
        if not dependency:
            raise FileNotFoundError('Configure dependencies.deepjscc source_root, revision, checkpoint; use external_worker for incompatible environments')
        verify_revision(dependency['source_root'],dependency['revision'])
        device = select_device(config['device'])
        adapter = DeepJSCCAdapter(dependency['checkpoint'],dependency['source_root']).build(device)
        from INFRA.Channels.channel_factory import create_channel
        channel = create_channel(config['transport']['name'])
        from INFRA.hashing import file_sha256
        return EncDecCodec('DeepJSCC',adapter,device=device,version=file_sha256(dependency['checkpoint']),
            operating_point=f"{channel.name};snr_db={options['snr_db']}",
            transport_fn=lambda latent: channel.transmit(latent,options['snr_db']))
    raise ValueError(f'{name} requires the configured isolated benchmark worker')

def execute(config, run):
    if config['action'] == 'train':
        from EncDec.training import train
        return train(config,run)
    from EncDec.catalog import availability, CATALOG
    if config['action'] in ('validate','availability'):
        return {'catalog':availability(config),'contract':'BenchmarkCodec encode/decode/complexity', 'model_inference_tested':False}
    implementation = config['implementation']
    dependency = config.get('dependencies',{}).get(implementation,{})
    if implementation in ('external_worker','ntscc') or (implementation == 'deepjscc' and dependency.get('factory')):
        from INFRA.workers import run_benchmark_worker
        return run_benchmark_worker(config,run)
    from INFRA.runs import seed_models, write_json
    from EncDec.Evaluation.kodak_benchmark import dataset_files, append_rows_csv
    from EncDec.Evaluation.audited_benchmark import benchmark_with_outcomes
    from EncDec.Evaluation.benchmark_report import summarize_compatible, plot_quality
    seed_models(config['seed'])
    codec = build_codec(config)
    files = dataset_files(Path(config['paths']['kodak']))
    if config['action'] == 'smoke':
        files = files[:1]
    from INFRA.hashing import file_sha256
    write_json(run.path/'dataset_manifest.json',[{'image_id':path.stem,'path':str(path),'sha256':file_sha256(path)} for path in files])
    rows, outcomes = benchmark_with_outcomes(codec,files,run_id=run.id,project_root=Path(__file__).resolve().parents[1],
        timing_repetitions=config['benchmark']['timing_repetitions'],compute_lpips=config['benchmark']['compute_lpips'])
    append_rows_csv(run.path/'benchmark.csv',rows)
    write_json(run.path/'outcomes.json',outcomes)
    summaries = summarize_compatible(rows)
    plot_quality(rows,run.path/'reconstruction.png')
    failed = sum(item['status']=='failed' for item in outcomes)
    return {'status':'benchmark_complete' if not failed else 'benchmark_with_failures','images':len(files),
            'failed':failed,'summary':summaries,'timing_note':'decode_time_ms includes transport for legacy adapters; separate measured stages are in outcomes.json when available',
            'capabilities':CATALOG[config['implementation']]}
