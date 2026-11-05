"""Run notebook9 dummy walkthrough using this interpreter in a fresh kernel."""
import json
from pathlib import Path
import sys
import tempfile
import time
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


def main():
    root=Path(__file__).resolve().parents[2]
    source=root/'experiments/parameter_finalization/notebooks/Full_dataset_Labeling_and_parameter_caching.ipynb'
    output=root/'experiments/parameter_finalization/runs'/('notebook9_execution_'+time.strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True,exist_ok=False)
    nb=nbformat.read(source,as_version=4)
    def save(**kwargs):
        nbformat.write(nb,output/'executed.ipynb')
    with tempfile.TemporaryDirectory(prefix='semantic_notebook_kernel_') as temporary:
        spec=Path(temporary)/'python3';spec.mkdir()
        (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
            'display_name':'Project interpreter','language':'python'}))
        manager=KernelManager(kernel_name='python3',kernel_spec_manager=KernelSpecManager(kernel_dirs=[temporary]))
        client=NotebookClient(nb,km=manager,timeout=180,resources={'metadata':{'path':str(root)}},
                              on_cell_executed=save)
        try:
            client.execute()
        finally:
            save()
            for i,cell in enumerate(nb.cells):
                if cell.cell_type=='code':
                    text=''.join(o.get('text','') if o.output_type=='stream' else
                        '\n'.join(o.get('traceback',[])) if o.output_type=='error' else '' for o in cell.outputs)
                    (output/f'cell_{i}.log').write_text(text,encoding='utf-8')
    print('Executed dummy notebook saved:',output,flush=True)


if __name__=='__main__':main()
