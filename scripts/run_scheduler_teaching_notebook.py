"""Execute the teaching notebook using this interpreter, preserving partial outputs."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import sys
import traceback

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'notebooks/teaching/semantic_scheduler_walkthrough.ipynb'
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + str(os.getpid())
    base = root / 'results/decoder_context/teaching'
    run = base / run_id
    operation = base / (run_id + '-execution')
    operation.mkdir(parents=True, exist_ok=False)
    kernel_dir = operation / 'kernels' / 'scheduler-lesson'
    kernel_dir.mkdir(parents=True)
    (kernel_dir / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'Local project lesson', 'language': 'python'}), encoding='utf-8')
    manager = KernelManager(kernel_name='scheduler-lesson',
                            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)]))
    notebook = nbformat.read(source, as_version=4)
    partial = operation / 'partial_executed.ipynb'
    def checkpoint(cell, cell_index, **kwargs):
        nbformat.write(notebook, partial)
        print(f'Cell {cell_index} checkpointed', flush=True)
    environment = dict(os.environ, SCHEDULER_REPO=str(root), SCHEDULER_LESSON_RUN=str(run),
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    metadata = {'source': str(source), 'interpreter': sys.executable, 'run': str(run),
                'notebook_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    (operation / 'execution.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2), flush=True)
    try:
        client = NotebookClient(notebook, km=manager, timeout=90, allow_errors=False,
                                on_cell_executed=checkpoint)
        client.execute(cwd=str(root), env=environment)
        nbformat.validate(notebook)
        executed = run / 'semantic_scheduler_walkthrough.executed.ipynb'
        nbformat.write(notebook, executed)
        try:
            from nbconvert import HTMLExporter
            html, _ = HTMLExporter().from_notebook_node(notebook)
            (run / 'semantic_scheduler_walkthrough.html').write_text(html, encoding='utf-8')
        except ImportError:
            print('nbconvert unavailable: executed notebook retained; HTML not generated.', flush=True)
        print(f'COMPLETE: {executed}', flush=True)
    except Exception:
        nbformat.write(notebook, partial)
        (operation / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)


if __name__ == '__main__':
    main()
