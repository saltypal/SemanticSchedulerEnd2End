"""Create a separate private two-T4 Qwen repair pilot; never overwrite old runs."""
import base64
import io
import json
from pathlib import Path
import zipfile
from scripts.build_decoder_v3_notebook import ROOT, SOURCES, cell


def build():
    source = io.BytesIO()
    with zipfile.ZipFile(source, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCES + ['Scheduler/training/decoder_v3_qwen_repair.py']:
            archive.write(ROOT/name, name)
        for name in ('Archived.owners_before_simplification_20261027.SemParam', 'Evaluation', 'Scheduler/training', 'Scheduler'):
            archive.writestr(name+'/__init__.py', '')
    encoded = base64.b64encode(source.getvalue()).decode()
    setup = f'''import subprocess,sys,base64,zipfile,io,json,hashlib
from pathlib import Path
subprocess.run([sys.executable,'-m','pip','install','-q','transformers==4.57.1','huggingface_hub','zstandard','pyarrow','Pillow'],check=True)
root=Path('/kaggle/working/source');root.mkdir(exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode({encoded!r}))).extractall(root)
sys.path.insert(0,str(root))
'''
    execute = '''from io import BytesIO
import pyarrow as pa
from PIL import Image,ImageDraw
from huggingface_hub import hf_hub_download
from Scheduler.training.decoder_v3_ollama import OllamaQwenReviewer
from Scheduler.training.decoder_v3_qwen_repair import run_diagnostic
path=hf_hub_download('MITLL/LADI-v2-dataset','data/validation/data-00000-of-00040.arrow',repo_type='dataset',revision='5f2dbfe8c466d32edafd1bab847ec5252309acdb')
with pa.memory_map(path,'r') as f: rows=pa.ipc.open_stream(f).read_all().to_pylist()
manifests=list(Path('/kaggle/input').rglob('extraction/manifest.json'))
assert len(manifests)==1,manifests
roles={r['image_id']:r['role'] for r in json.loads(manifests[0].read_text())}
images=[]
for row in rows:
    data=row['image']['bytes'];identifier=hashlib.sha256(data).hexdigest()
    if roles.get(identifier)=='A':
        images.append((identifier,Image.open(BytesIO(data)).convert('RGB'),'development_image'))
    if len(images)==4: break
assert images,'no development images: fail rather than use test'
control=Image.new('RGB',(512,512),'white');draw=ImageDraw.Draw(control)
draw.rectangle((40,80,200,240),fill='red');draw.ellipse((300,300,450,450),fill='blue')
images.insert(0,('synthetic_red_square_blue_circle',control,'vision_transport_control'))
reviewer=OllamaQwenReviewer(log_dir='/kaggle/working/diagnostic')
try: print(run_diagnostic(reviewer,images,'/kaggle/working/diagnostic'))
finally: reviewer.close()
'''
    folder = ROOT/'kaggle/disaster-qwen-repair'
    folder.mkdir(parents=True, exist_ok=True)
    notebook = dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(
        display_name='Python 3', language='python', name='python3')),
        cells=[cell('markdown', '# Qwen repair pilot\nDevelopment images only. No scheduler training. Synthetic control is not disaster evidence.'),
               cell('code', setup), cell('code', execute)])
    (folder/'qwen_diagnostic.ipynb').write_text(json.dumps(notebook))
    metadata = dict(id='hmmmkk/disaster-qwen-repair-pilot', title='Disaster Qwen Repair Pilot',
        code_file='qwen_diagnostic.ipynb', language='python', kernel_type='notebook',
        is_private=True, enable_gpu=True, enable_internet=True,
        dataset_sources=[], competition_sources=[], model_sources=[],
        kernel_sources=['hmmmkk/disaster-eight-field-ollama'])
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
    print(folder)


if __name__ == '__main__':
    build()
