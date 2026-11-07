import argparse, hashlib, json, pathlib, zipfile
p=argparse.ArgumentParser(); p.add_argument('--campaign', required=True); p.add_argument('--source-zip', required=True); p.add_argument('--expected-source-sha', required=True); p.add_argument('--output', required=True); a=p.parse_args()
campaign=pathlib.Path(a.campaign); source=pathlib.Path(a.source_zip); output=pathlib.Path(a.output)
def sha(data): return hashlib.sha256(data).hexdigest()
source_hash=sha(source.read_bytes())
checks=json.loads((campaign/'checksums.json').read_text(encoding='utf-8'))
check_results={}
for rel, expected in checks.items():
 f=campaign/rel
 check_results[rel]={'exists':f.is_file(),'expected':expected,'actual':sha(f.read_bytes()) if f.is_file() else None,'match':f.is_file() and sha(f.read_bytes())==expected}
manifest=json.loads((campaign/'manifest.json').read_text(encoding='utf-8'))
source_hashes=manifest.get('source_hashes',{})
with zipfile.ZipFile(source) as z:
 names=set(z.namelist()); source_check={}
 for rel, expected in source_hashes.items():
  candidate=rel if rel in names else next((n for n in names if n.endswith('/'+rel)),None)
  actual=sha(z.read(candidate)) if candidate else None
  source_check[rel]={'zip_member':candidate,'expected':expected,'actual':actual,'match':actual==expected}
summary={'checksums_total':len(check_results),'checksums_matched':sum(x['match'] for x in check_results.values()),'checksum_failures':[k for k,v in check_results.items() if not v['match']],'source_zip_sha256':source_hash,'expected_source_sha256':a.expected_source_sha,'source_zip_sha_match':source_hash==a.expected_source_sha,'runtime_source_hashes_total':len(source_check),'runtime_source_hashes_matched':sum(x['match'] for x in source_check.values()),'runtime_source_hash_failures':{k:v for k,v in source_check.items() if not v['match']},'run_id':manifest.get('run_id'),'source_hash_entries':source_hashes,'check_results':check_results}
output.write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k not in ('source_hash_entries','check_results')},indent=2))
