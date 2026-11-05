"""Package the bounded v2 teacher notebook; building never submits it."""
from pathlib import Path
import argparse, ast, json
import nbformat
from .build_spi_notebook import build as build_base
from .artifacts import write_json


def build(output,tag,input_dataset,server_seconds=3600):
    output=build_base(output,tag,input_dataset,server_seconds)
    path=Path(output)/'notebook.ipynb'
    notebook=nbformat.read(path,as_version=4)
    cells=[cell for cell in notebook.cells if cell.cell_type=='code']
    prefix='CELL_SOURCES='
    header, _ = cells[0].source.split('\n',1)
    if not header.startswith(prefix):raise ValueError('unexpected notebook source ledger')
    sources=ast.literal_eval(header[len(prefix):])
    original='from parameters.spi_campaign import execute,verify_inputs'
    replacement=('from parameters.spi_final_campaign import execute\n'
                 '    from parameters.spi_campaign import verify_inputs')
    if original not in sources[2]:raise ValueError('SPI execution import not found')
    sources[2]=sources[2].replace(original,replacement)
    call="reviewer,seconds=max(1,int("
    if call not in sources[2]:raise ValueError('SPI campaign invocation not found')
    sources[2]=sources[2].replace(call,"reviewer,enable_scene=True,seconds=max(1,int(")
    pip_end="    from transformers import Qwen3_5ForConditionalGeneration"
    dependencies="""    extra=subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','ftfy'],
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
    print(extra.stdout,flush=True);extra.check_returncode()
    extra=subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','--no-deps',
        'ultralytics==8.3.237','git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6'],
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=300)
    print(extra.stdout,flush=True);extra.check_returncode()
"""
    if pip_end not in sources[1]:raise ValueError('dependency insertion point missing')
    sources[1]=sources[1].replace(pip_end,dependencies+pip_end)
    sources[1]=sources[1].replace("('torch','transformers','huggingface-hub','accelerate','bitsandbytes','numpy')",
        "('torch','transformers','huggingface-hub','accelerate','bitsandbytes','numpy','ultralytics','scikit-learn','Pillow','nbformat')")
    summary_line="    write_json(Path('/kaggle/working/spi_checksums.json')"
    if summary_line not in sources[3]:raise ValueError('summary insertion point missing')
    sources[3]=sources[3].replace(summary_line,
        "    if list(root.rglob('scene_interrupted.json')):files=[p for p in files if '/scene/' not in str(p)]\n"+summary_line)
    for i,source in enumerate(sources):
        compile(source,f'final-spi-cell-{i}','exec')
        cells[i].source=(prefix+repr(sources)+'\n' if i==0 else '')+source
    notebook.cells[0].source='''# Parameter finalization: SPI v2 teacher campaign

Managed Kaggle environment; source archive and cohort hashes are checked before
model loading. GPU1 runs Qwen3.5-9B in NF4 with thinking disabled. GPU0 runs
one YOLO-World detector independently of the text preflight; CLIP
image features are cached. Producer ownership and statuses are recorded.

The 12 text controls are synthetic rubric checks, not visual accuracy. Image
labels remain pseudo-labels, never human urgency truth. The 180-image cohort is
fully development-exposed. Incremental raw outputs and the recovery archive are
under /kaggle/working/results/spi; cell logs and executed notebook under
/kaggle/working/cell_evidence. External retrieval is required for durability.
If the four-class support gate passes, grouped CPU model selection follows the
labels. A separate VLM structural-control and natural-image diagnostic uses
actual scene predictions. No scheduler policy is trained.'''
    nbformat.write(notebook,path)
    package=json.loads((Path(output)/'SPI_PACKAGE.json').read_text())
    package.update(purpose='v2 text controls, image pseudo-labels and GPU0 scene observations',gpu0_use='YOLO-World scene detection',
                   rubric='visible_response_attention_v2',student_training='CPU grouped selection only if support gate passes')
    write_json(Path(output)/'SPI_PACKAGE.json',package)
    return output

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--tag',required=True);parser.add_argument('--input-dataset',required=True)
    parser.add_argument('--server-seconds',type=int,default=3600)
    args=parser.parse_args();print(build(args.output,args.tag,args.input_dataset,args.server_seconds))
