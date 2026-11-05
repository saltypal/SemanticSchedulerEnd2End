"""Package the authorized remaining-grid run using the established notebook setup."""
from pathlib import Path
import argparse
import json
import nbformat as nbf
from .build_annotation_notebook import build as build_pilot
from .artifacts import write_json


def build(output,tag):
    output=build_pilot(output,tag,scope='authorized remaining grids:YOLO12 CLIP8 Qwen8;no scheduler training')
    notebook=nbf.read(output/'notebook.ipynb',as_version=4)
    code=[cell.source for cell in notebook.cells if cell.cell_type=='code']
    # The first line embeds CellRecorder sources. Replace it after rewriting cells.
    setup=code[0].split('\n',1)[1]
    setup=setup.replace('Configuration:GPU0CLIP+YOLOWorld,GPU1Qwen4B;server timeout3600s;internal3000s',
        'Configuration:GPU0YOLO12+CLIP8grids,GPU1Qwen8controlgrids;server3600s;internal3000s')
    dependencies=code[1]
    run=f'''with recorder.cell(2):
    from parameters.cloud_notebook import locate
    from parameters.cloud_model_grid import run
    MANIFEST=locate('data/records.json')
    print('Starting bounded remaining-grid campaign with prior review exclusions',flush=True)
    OUTPUT=run(MANIFEST,Path('/kaggle/working/results/model_grid/{tag}'),ROOT/'parameters/config/grid_control_exclusions.json',seconds=3000)
    print('ACTUAL campaign artifact root',OUTPUT,flush=True)'''
    summary='''with recorder.cell(3):
    from parameters.artifacts import write_json
    print('ACTUAL CAMPAIGN OUTCOME',json.loads((OUTPUT/'status.json').read_text()),flush=True)
    hashes={str(p.relative_to(OUTPUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUTPUT.rglob('*') if p.is_file()}
    write_json(Path('/kaggle/working/artifact_checksums.json'),hashes)
    print('Saved detector raw boxes, parity tests, all grid rows, CLIP scores, control outputs and model settings. No winner automatically selected.')'''
    sources=[setup,dependencies,run,summary]
    cells=[nbf.v4.new_markdown_cell(f'''# parameters remaining approach grids — {tag}

Authorized execution: YOLO12settings,CLIP8settings,Qwen8settings×24controls.
GPU0 YOLO/CLIP;GPU1 Qwen. Prepared private source20261025b is reused. No new
dataset download, scheduler training, or codec inference. Fixed20image detector
cohort is diagnostic; reviewer controls use8new groups excluding48previous groups.
Image-level labels cannot establish box localization quality. Prompt tests score
known omission/contradiction controls rather than optimizing approval.

Managed Kaggle runtime; dependencies/setup and real processor-only check print
actual environment/devices. Server3600s maximum, internal3000s, at most2allocated
GPU-hours. No automatic retry or follow-on run. A failed detector stage is
preserved while independent stages may finish; inspect status=partial explicitly.

Outputs: /kaggle/working/results/model_grid/{tag}/; cell_evidence/ contains
executed notebook, cell logs and status. Raw artifacts flush per image/case.
Runtime and saved-version retrieval must be verified; killed-process buffers
may not all survive. All results remain exploratory, with no selected winner.''')]
    for index,source in enumerate(sources):
        compile(source,'grid_cell_'+str(index),'exec')
        cells.append(nbf.v4.new_code_cell(('CELL_SOURCES='+repr(sources)+'\n' if index==0 else '')+source))
    notebook.cells=cells;nbf.write(notebook,output/'notebook.ipynb')
    metadata=json.loads((output/'kernel-metadata.json').read_text())
    slug='betterscheduler-model-grid-'+tag
    metadata.update(id='satyapaladugu/'+slug,title='Betterscheduler Model Grid '+tag)
    write_json(output/'kernel-metadata.json',metadata)
    manifest=json.loads((output/'pilot_manifest.json').read_text())
    manifest.update(slug='satyapaladugu/'+slug,stage='remaining_model_grids',
        scope='YOLO12 CLIP8 Qwen8; no automatic winner or downstream training')
    write_json(output/'grid_manifest.json',manifest)
    write_json(output/'pilot_manifest.json',{**manifest,'compatibility_alias_for':'grid_manifest.json'})
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--tag',required=True)
    args=parser.parse_args();print(build(args.output,args.tag))
