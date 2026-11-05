from pathlib import Path
import hashlib,json,tarfile
source=Path('/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticLossless/Semantic_Lossless/Results/team_downloads/2/satyapaladuguplus__semantic-notebook-2/0b680269ce1a4e2c83d9fe5159b3af1c/published_output/models/teammate_2_models/SQ-GAN/20261028T130100Z-eea4f3aa')
manifest=json.loads((source/'bundle.json').read_text()); verified=[]
for item in manifest['files']:
    p=source/item['path']; h=hashlib.sha256()
    if p.stat().st_size!=item['bytes']: raise RuntimeError(f'size mismatch {item["path"]}: {p.stat().st_size} != {item["bytes"]}')
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    if h.hexdigest()!=item['sha256']: raise RuntimeError(f'hash mismatch {item["path"]}')
    verified.append(item['path'])
print('source_manifest_files_verified',len(verified),flush=True)
out=Path('/tmp/scheduler_sqgan_input_20261102/SQ-GAN.tar')
with tarfile.open(out,'w') as archive: archive.add(source,arcname='20261028T130100Z-eea4f3aa',recursive=True)
with tarfile.open(out,'r') as archive:
    last=archive.getmember('20261028T130100Z-eea4f3aa/last.pt')
    if last.size!=2341438033: raise RuntimeError(f'tar last.pt size {last.size}')
    names=archive.getnames()
    if len(names)<28: raise RuntimeError(f'tar member count {len(names)}')
result={'manifest_files_verified':len(verified),'tar_member_count':len(names),'tar_last_pt_bytes':last.size,'tar_bytes':out.stat().st_size,'source_bundle_sha256':hashlib.sha256((source/'bundle.json').read_bytes()).hexdigest(),'tar_valid':True}
Path('/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End/experiments/master_study/runs/paired-launch-20261102-205933/sqgan_bundle_integrity.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
