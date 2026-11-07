import contextlib,hashlib,json,shutil,tempfile
from pathlib import Path
from unittest.mock import patch
repo=Path.cwd()
nb=json.loads((repo/'experiments/master_study/paired_kaggle/wsl/Scheduler_WSL_Ranking_VLM.ipynb').read_text())
code=next(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code' and '06_frozen_sqgan_payload_check' in ''.join(c['source']))
prefix=code.split('    # Verify one image, then five more.')[0]
prefix=prefix.replace('    from Codec.integration.sqgan_payload import SQGANDigitalCodec\n','')
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
@contextlib.contextmanager
def stage(name):yield
results=[]
with tempfile.TemporaryDirectory(prefix='flat-ingress-check-') as temp:
    root=Path(temp);inputs=root/'input';inputs.mkdir();work=root/'work';work.mkdir()
    package=inputs/'package';package.mkdir()
    (package/'f000_small.bin').write_bytes(b'verified fixture payload')
    manifest={'files':[{'path':'assets/checkpoint/small.bin','bytes':24,'sha256':digest(package/'f000_small.bin')}]}
    # Derive the actual fixture size; no production weights are copied.
    manifest['files'][0]['bytes']=(package/'f000_small.bin').stat().st_size
    (package/'bundle_manifest.json').write_text(json.dumps(manifest))
    mapping={'schema':'scheduler_sqgan_flat_v1','run_id':'20261028T130100Z-eea4f3aa','bundle_manifest':'bundle_manifest.json','files':{'assets/checkpoint/small.bin':'f000_small.bin'}}
    sidecar=package/'scheduler_sqgan_flat_manifest.json';sidecar.write_text(json.dumps(mapping))
    fixture=prefix.replace("Path('/kaggle/input')",repr(inputs)+'' if False else "Path("+repr(str(inputs))+")")
    fixture=fixture.replace("'/kaggle/working'",repr(str(work)))
    fixture=fixture.replace('8b482d80a1adc7dde7a2725461a17346aca49d6628d4f78f88f1619df7448c21',digest(package/'bundle_manifest.json'))
    copied=[];actual_copy=shutil.copyfile
    def copy(src,dst):
        actual_copy(src,dst);copied.append((str(dst),Path(dst).read_bytes()))
    with patch('shutil.copyfile',copy):
        env={'Path':Path,'json':json,'digest':digest,'stage':stage}
        exec(compile(fixture,'flat-input-fixture','exec'),env)
    assert len(copied)==2 and copied[1][1]==b'verified fixture payload'
    assert not env['candidates'][0].exists(), 'Temporary bundle was not cleaned'
    results.append('materializes nested bundle and cleans temporary copy')
    (package/'f000_small.bin').write_bytes(b'x')
    try:exec(compile(fixture,'flat-input-fixture','exec'),{'Path':Path,'json':json,'digest':digest,'stage':stage})
    except ValueError as e:assert 'Incomplete uploaded' in str(e)
    else:raise AssertionError('Truncated input accepted')
    results.append('rejects truncated uploaded member')
    (package/'bundle_manifest.json').write_text('{}')
    try:exec(compile(fixture,'flat-input-fixture','exec'),{'Path':Path,'json':json,'digest':digest,'stage':stage})
    except ValueError as e:assert 'manifest mismatch' in str(e)
    else:raise AssertionError('Changed manifest accepted')
    results.append('rejects changed bundle manifest')
out=repo/'experiments/master_study/runs/paired-launch-20261102-205933/flat_ingress_validation.json'
out.write_text(json.dumps({'scope':'fixture input materialization only; original bundle separately fully hash-audited','checks':results,'passed':len(results)},indent=2))
print('FLAT INPUT CHECKS',len(results),'PASS')
