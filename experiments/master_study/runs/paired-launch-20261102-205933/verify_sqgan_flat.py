from pathlib import Path
import hashlib,json
stage=Path('/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End/experiments/master_study/runs/paired-launch-20261102-205933/private_inputs/sqgan_flat')
side=json.loads((stage/'scheduler_sqgan_flat_manifest.json').read_text())
manifest=stage/side['bundle_manifest']
manifest_sha=hashlib.sha256(manifest.read_bytes()).hexdigest()
results=[]
for rel,item in side['files'].items():
    p=stage/item['file']; h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    results.append({'relative_path':rel,'bytes_ok':p.stat().st_size==item['bytes'],'sha_ok':h.hexdigest()==item['sha256']})
result={'schema':side['schema'],'run_id':side['run_id'],'bundle_manifest_sha256':manifest_sha,'bundle_manifest_sha_ok':manifest_sha=='8b482d80a1adc7dde7a2725461a17346aca49d6628d4f78f88f1619df7448c21','count':len(results),'all_files_ok':all(x['bytes_ok'] and x['sha_ok'] for x in results),'files':results}
path=Path('/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End/experiments/master_study/runs/paired-launch-20261102-205933/sqgan_flat_integrity.json')
path.write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='files'}),flush=True)
