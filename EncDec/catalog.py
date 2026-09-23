"""Capabilities describe implementations, not inferred model accuracy."""
from pathlib import Path
CATALOG = {
    'raw': {'adapter':'RawRGBCodec','representation':'digital_bits','preprocessing':'RGB8, unchanged size','checkpoint':'none','environment':'Pillow + torch metrics'},
    'jpeg': {'adapter':'JPEGCodec','representation':'digital_bits','preprocessing':'RGB8, unchanged size; subsampling=0','checkpoint':'none','environment':'Pillow + torch metrics'},
    'swinjscc': {'adapter':'SwinJSCCAdapter','representation':'complex_channel_uses','preprocessing':'RGB [0,1] BCHW; dimensions multiple of 128','checkpoint':'hash-validated stage1_manifest.json','environment':'manifest-locked Python/PyTorch + pinned patched Swin source'},
    'deepjscc': {'adapter':'DeepJSCCAdapter or repaired worker','representation':'complex_channel_uses','preprocessing':'external model contract; benchmark prohibits resizing','checkpoint':'configured external checkpoint','environment':'versioned external source; isolated worker supported'},
    'ntscc': {'adapter':'repaired_notebook_models benchmark wrapper','representation':'complex_channel_uses','preprocessing':'external model contract; benchmark prohibits resizing','checkpoint':'configured external checkpoint','environment':'external NTSCC/semcom_benchmark environment; worker required'},
}

def availability(config):
    result = {name:{**details,'available':name in ('raw','jpeg'),'validation':'not run'} for name, details in CATALOG.items()}
    folder = Path(config['paths']['model_dir'])
    required = ['stage1_manifest.json','stage1_swinjscc_full.pt','stage1_swinjscc_encoder.pt','stage1_swinjscc_decoder.pt']
    missing = [str(folder/name) for name in required if not (folder/name).is_file()]
    source = Path(config['paths']['upstream_swin'])/'.stage1a_patch.json'
    if not source.is_file():
        missing.append(str(source))
    result['swinjscc'].update(available=not missing, missing=missing, validation='assets present; runtime validation required' if not missing else 'missing assets')
    for name in ('deepjscc','ntscc'):
        dependency = config.get('dependencies',{}).get(name,{})
        checkpoint = dependency.get('checkpoint') or dependency.get(f'{name}_checkpoint')
        result[name].update(available=bool(checkpoint and Path(checkpoint).is_file()),
                            validation='external dependency not configured' if not dependency else 'configured; worker smoke required')
    return result
