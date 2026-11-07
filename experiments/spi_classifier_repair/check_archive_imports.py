"""Test imports from the uploaded source archive without the canonical checkout."""
from pathlib import Path
import zipfile,subprocess,sys,tempfile,json

def check(package):
 package=Path(package).resolve()
 with tempfile.TemporaryDirectory(prefix='spi_archive_import_') as temp:
  root=Path(temp)
  with zipfile.ZipFile(package/'source.zip') as z:
   assert z.testzip() is None
   for name in z.namelist():
    assert (root/name).resolve().is_relative_to(root)
   z.extractall(root)
  code="import sys,importlib.util;sys.path.insert(0,sys.argv[1]);import parameters.cloud_notebook,parameters.dataset_pipeline;from pathlib import Path;p=Path(sys.argv[1])/'experiments/spi_classifier_repair/cnn_compare.py';s=importlib.util.spec_from_file_location('cnn',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);print('isolated archive imports passed')"
  result=subprocess.run([sys.executable,'-I','-c',code,str(root)],cwd=root,capture_output=True,text=True,timeout=60)
  print(result.stdout);print(result.stderr)
  if result.returncode:raise RuntimeError('exact staged archive import failed')
 return {'isolated_imports':'passed','source_zip':str(package/'source.zip')}

if __name__=='__main__':
 print(json.dumps(check(sys.argv[1]),indent=2))
