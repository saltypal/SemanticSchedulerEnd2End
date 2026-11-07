from pathlib import Path
import hashlib,json,tarfile
root=Path('/home/Satya/paired_sqgan_bundle_20261102/SQ-GAN/20261028T130100Z-eea4f3aa')
manifest=json.loads((root/'bundle.json').read_text()); checks=[]
for e in manifest['files']:
    p=root/e['path']; h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    checks.append({'path':e['path'],'size_ok':p.stat().st_size==e['bytes'],'sha_ok':h.hexdigest()==e['sha256']})
tar=tarfile.open('/tmp/scheduler_sqgan_input_20261102/SQ-GAN.tar')
names=set(tar.getnames())
tar_ok=all(('20261028T130100Z-eea4f3aa/'+e['path']) in names for e in manifest['files']) and '20261028T130100Z-eea4f3aa/bundle.json' in names
result={'manifest_file_count':len(checks),'all_files_verified':all(x['size_ok'] and x['sha_ok'] for x in checks),'tar_manifest_paths_present':tar_ok,'bundle_sha256':hashlib.sha256((root/'bundle.json').read_bytes()).hexdigest(),'tar_bytes':Path('/tmp/scheduler_sqgan_input_20261102/SQ-GAN.tar').stat().st_size,'checks':checks}
Path('/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End/experiments/master_study/runs/paired-launch-20261102-205933/sqgan_bundle_integrity.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
