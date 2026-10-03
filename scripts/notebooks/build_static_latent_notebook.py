"""Generate a portable, module-backed cached static scheduling notebook."""
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[2]


def build():
    md = nbformat.v4.new_markdown_cell
    code = nbformat.v4.new_code_cell
    notebook = nbformat.v4.new_notebook(cells=[
        md("# Whole image latent scheduler\n\nStatic comparison of six policies. Actions select payloads, not UEs. The default input is an explicitly synthetic functional fixture. Set `LATENT_BATCH` to a real cached batch. No model training or codec encoding occurs in this notebook."),
        md("## Clean source setup\n\nLocal execution uses the current repository. Kaggle/Colab may use a pinned Git clone (`LATENT_REPOSITORY`, `LATENT_COMMIT`) or an attached `static_source.zip`. The reset target is fixed to the disposable runtime directory. Persistent results live outside it. An archive records its exact source digest rather than pretending a dirty snapshot is a pushed commit."),
        code('''import os, sys, json, hashlib, platform, subprocess, shutil, zipfile
from pathlib import Path
is_kaggle = Path('/kaggle/working').exists()
is_colab = 'google.colab' in sys.modules or Path('/content').exists()
runtime = Path('/kaggle/working' if is_kaggle else '/content')
if is_kaggle or is_colab:
    root = runtime / 'static-latent-project'
    if root.exists():
        shutil.rmtree(root)
    archives = list(Path('/kaggle/input').rglob('static_source.zip')) if is_kaggle else []
    archive = os.environ.get('LATENT_SOURCE_ARCHIVE') or (str(archives[0]) if len(archives) == 1 else '')
    if archive:
        root.mkdir()
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                target = (root / member.filename).resolve()
                if not target.is_relative_to(root.resolve()):
                    raise ValueError('Unsafe source archive path')
            bundle.extractall(root)
    else:
        repository = os.environ['LATENT_REPOSITORY']
        commit = os.environ['LATENT_COMMIT']
        subprocess.run(['git', 'clone', repository, str(root)], check=True)
        subprocess.run(['git', '-C', str(root), 'checkout', commit], check=True)
else:
    root = next(path for path in [Path.cwd(), *Path.cwd().parents] if (path / 'PROJECT_MAP.md').exists())
sys.path.insert(0, str(root))
print(json.dumps({'stage':'source_ready', 'root':str(root), 'python':platform.python_version(), 'execution':'cached_cpu'}), flush=True)
'''),
        code('''from INFRA.latent.experiment import compare_batch, source_provenance
batch = Path(os.environ.get('LATENT_BATCH', str(root / 'Scheduler/config/static_latent_demo.json')))
if 'LATENT_BATCH' not in os.environ and is_kaggle:
    matches = list(Path('/kaggle/input').rglob('latent_batch.json'))
    if len(matches) == 1:
        batch = matches[0]
output_root = Path(os.environ.get('LATENT_RESULTS', str(runtime / 'static-latent-results') if is_kaggle or is_colab else str(root / 'results')))
if is_colab and 'LATENT_RESULTS' not in os.environ:
    raise ValueError('Set LATENT_RESULTS to mounted persistent storage on Colab')
if output_root.resolve().is_relative_to(root.resolve()) and (is_kaggle or is_colab):
    raise ValueError('Results cannot live inside a disposable checkout')
print(json.dumps({'stage':'configuration', 'batch':str(batch), 'output_root':str(output_root), **source_provenance(root)}), flush=True)
run_dir = compare_batch(batch, output_root)
'''),
        md("## Metrics and latent order\n\nReference ranking and actual task quality are separate. Null values mean unavailable evidence. Synthetic evidence validates behavior only. Kaggle outputs persist on kernel completion; download them with `kaggle kernels output <owner/slug> -p <destination>`."),
        code('''import pandas as pd
import matplotlib.pyplot as plt
metrics = pd.read_csv(run_dir / 'metrics.csv')
display(metrics[['policy','served','reference_ndcg','predicted_priority_gate_violations','evidence_quality_auc','mean_wait_ms']])
result = json.loads((run_dir / 'semantic.trace.json').read_text())
rows = result['decision_trace']
fig, ax = plt.subplots(figsize=(10, max(3, len(rows) * .35)))
for index, row in enumerate(rows):
    ax.barh(index, row['completed_ms'] - row['now_ms'], left=row['now_ms'])
ax.set_yticks(range(len(rows)), [row['payload_id'] for row in rows])
ax.invert_yaxis()
ax.set_xlabel('Configured service time (ms)')
ax.set_title('Whole-latent semantic transmission order')
fig.tight_layout()
fig.savefig(run_dir / 'semantic_order.png', dpi=160)
plt.show()
display(pd.DataFrame(result['outcomes']))
'''),
        md("## Interpretation\n\nInspect `semantic.trace.json` to explain every decision and each parameter. A successful cached schedule does not prove checkpoint reconstruction or downstream task accuracy. See `docs/STATIC_LATENT_SCHEDULER.md` for methodology and limitations. The existing codec replay accepts the selected image IDs without placing neural code in Scheduler."),
    ])
    notebook.metadata = {'kernelspec': {'display_name':'Python 3', 'language':'python', 'name':'python3'}, 'language_info':{'name':'python','version':'3.12'}}
    nbformat.validate(notebook)
    destination = ROOT / 'Scheduler/notebooks/04_static_latent_scheduler.ipynb'
    nbformat.write(notebook, destination)
    print(destination)


if __name__ == '__main__':
    build()
