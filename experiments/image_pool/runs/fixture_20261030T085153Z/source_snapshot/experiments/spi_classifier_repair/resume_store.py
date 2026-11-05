"""Small atomic, hash-verified stage receipts. Never reuse by path alone."""
import hashlib,json,os
from pathlib import Path

def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda:source.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def implementation_identity(*objects):
    import inspect,importlib.metadata,platform
    return {'source_sha256':hashlib.sha256('\n'.join(inspect.getsource(obj) for obj in objects).encode()).hexdigest(),
        'python':platform.python_version(),'packages':{name:importlib.metadata.version(name) for name in ('torch','numpy','scikit-learn','transformers')}}

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.partial')
    with temp.open('w',encoding='utf8') as target:
        json.dump(value,target,indent=2,allow_nan=False);target.flush();os.fsync(target.fileno())
    temp.replace(path)

class ResumeStore:
    def __init__(self,root):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
    def _receipt(self,name,key):
        if not name.replace('_','').isalnum():raise ValueError('invalid stage name')
        return self.root/name/key/'receipt.json'
    def read(self,name,spec):
        key=fingerprint(spec);receipt=self._receipt(name,key)
        if not receipt.exists():return None
        row=json.loads(receipt.read_text())
        if row.get('fingerprint')!=key or fingerprint(row.get('spec'))!=key or row.get('status')!='complete':return None
        for item in row['artifacts']:
            path=(receipt.parent/item['path']).resolve()
            if not path.is_relative_to(receipt.parent.resolve()):raise ValueError('unsafe receipt artifact')
            if not path.is_file() or file_hash(path)!=item['sha256']:
                raise ValueError('checkpoint integrity failure: '+str(path))
        return row
    def directory(self,name,spec):
        path=self._receipt(name,fingerprint(spec)).parent;path.mkdir(parents=True,exist_ok=True);return path
    def complete(self,name,spec,files,value):
        directory=self.directory(name,spec);artifacts=[]
        for file in files:
            file=Path(file).resolve()
            if not file.is_relative_to(directory):raise ValueError('artifact outside stage checkpoint')
            artifacts.append({'path':file.relative_to(directory).as_posix(),'sha256':file_hash(file),'bytes':file.stat().st_size})
        row={'status':'complete','fingerprint':fingerprint(spec),'spec':spec,'artifacts':artifacts,'value':value}
        atomic_json(directory/'receipt.json',row);return row
