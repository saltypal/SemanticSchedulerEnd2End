import contextlib,hashlib,json,shutil,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
repo=Path.cwd()
nb=json.loads((repo/'experiments/master_study/paired_kaggle/wsl/Scheduler_WSL_Ranking_VLM.ipynb').read_text())
code=next(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code' and '06_frozen_sqgan_payload_check' in ''.join(c['source']))
prefix=code.split('    # Verify one image, then five more.')[0]
prefix=prefix.replace('    from Codec.integration.sqgan_payload import SQGANDigitalCodec\n','')
source=(repo/'experiments/master_study/runs/paired-launch-20261102-205933/private_inputs/sqgan_flat').resolve()
original=json.loads((source/'bundle_manifest.json').read_text())
mapping=json.loads((source/'scheduler_sqgan_flat_manifest.json').read_text())
by_flat={mapping['files'][row['path']]['file']:row for row in original['files']}
def digest(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
@contextlib.contextmanager
def stage(name):yield
verified=[];real_copy=shutil.copyfile
def verify_copy(src,dst):
    if Path(src).name=='bundle_manifest.json':return real_copy(src,dst)
    row=by_flat[Path(src).name]
    assert Path(src).stat().st_size==row['bytes']
    assert digest(src)==row['sha256'],row['path']
    verified.append(row['path'])
    print('VERIFIED ACTUAL FLAT ENTRY',row['path'],flush=True)
    # Dry copy only: all real input bytes were read/hash-verified; avoid another3GB allocation locally.
    return str(dst)
with tempfile.TemporaryDirectory(prefix='actual-flat-check-') as temp:
    fixture=prefix.replace("Path('/kaggle/input')","Path("+repr(str(source))+")")
    fixture=fixture.replace("'/kaggle/working'",repr(temp))
    with patch('shutil.copyfile',verify_copy),patch('shutil.disk_usage',return_value=SimpleNamespace(free=10*1024**3)):
        env={'Path':Path,'json':json,'digest':digest,'stage':stage}
        exec(compile(fixture,'actual-uploaded-flat-manifest','exec'),env)
    assert len(verified)==len(original['files'])==27
    assert not env['candidates'][0].exists()
out=repo/'experiments/master_study/runs/paired-launch-20261102-205933/actual_flat_ingress_validation.json'
out.write_text(json.dumps({'scope':'Actual uploaded manifest parsed by exact notebook loader prefix; all27 real source files hash-read; large remote copy substituted locally; neural inference not tested','verified_entries':verified,'count':len(verified),'temporary_cleanup':True},indent=2))
print('ACTUAL UPLOADED MANIFEST INGESTION:27 FILES PASS',flush=True)
