"""Build/execute the local evidence report without rerunning training or test."""
from pathlib import Path
import sys

import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager


def build():
    root = Path(__file__).resolve().parents[2]
    run = root/"experiments/spi_classifier_repair/runs/minority_recovery_20261029"
    notebook = run/"SPI_minority_and_label_audit.ipynb"
    if notebook.exists():
        # Preserve the previous executed notebook before updating the same report.
        import shutil
        previous = run/"notebook_execution/executed.ipynb"
        if not previous.exists():
            shutil.copy2(notebook, previous)
    cells = [nbf.v4.new_markdown_cell("""# SPI minority-class and label audit

This is an executed **local diagnostic report**, not a new Kaggle run or final
evaluation. Training already executed in the project `.venv`; this notebook
verifies and visualizes its saved evidence. All former-test targets are excluded
from this recovery study. The existing 300 MEDIC images were reused.

Run from `SemanticSchedulerEnd2End` using `.venv/Scripts/python.exe -m
experiments.spi_classifier_repair.build_recovery_notebook`. The existing project
dependency specification/environment is reused; no new packages are installed.
Actual versions and paths are printed below. Kaggle v20 results remain immutable.
""")]
    code = []
    code.append("""print('SETUP START', flush=True)
import json, sys, platform, time, importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from parameters.artifacts import Run
from experiments.spi_classifier_repair.resume_store import file_hash
ROOT = Path.cwd().resolve()
RUN = ROOT/'experiments/spi_classifier_repair/runs/minority_recovery_20261029'
execution_path = RUN/'notebook_execution'
if execution_path.exists():
    execution_path = RUN/('notebook_execution_'+str(time.time_ns()))
REPORT_RUN = Run(execution_path, {'scope':'saved-result diagnostic; no fitting/test use'})
print('Interpreter:', sys.executable, 'Python:', sys.version, 'Project:', ROOT)
print('Backend: CPU report rendering; no new encoder/VLM forward')
print('Packages:', {p:importlib.metadata.version(p) for p in ['numpy','pandas','matplotlib','torch','transformers','scikit-learn']})
print('Source:', json.loads((RUN/'source_manifest.json').read_text())['commit'])
print('Output:', REPORT_RUN.path, 'SETUP END', flush=True)
""")
    code.append("""with REPORT_RUN.stage('01_verify_inputs'):
    manifest=json.loads((RUN/'training/manifest.json').read_text())
    for key, path in [('features_sha256','comparison/clip_b32/features.npz'),('labels_sha256','labeling/labels.json'),('records_sha256','records.json')]:
        source=ROOT/'experiments/spi_classifier_repair/runs/monitor_v20/compact_outputs_v20/campaign'/path
        assert file_hash(source)==manifest[key], path
    assert file_hash(ROOT/'experiments/spi_classifier_repair/minority_recovery.py')==manifest['script_sha256']
    print('All input/source hashes match the executed recovery run.')
    print('Class counts, Low/Medium/High/Critical:', manifest['class_counts'])
    print('Former test:', manifest['old_test'])
    fig,ax=plt.subplots(figsize=(7,3.5)); names=['Low','Medium','High','Critical']; ix=np.arange(4)
    ax.bar(ix-.18,manifest['class_counts']['train'],.36,label='Training')
    ax.bar(ix+.18,manifest['class_counts']['validation'],.36,label='Development')
    ax.set_xticks(ix,names); ax.set_ylabel('Image count'); ax.legend(); fig.tight_layout()
    fig.savefig(REPORT_RUN.path/'figures/support.png',dpi=150); display(fig); plt.close(fig)
""")
    code.append("""with REPORT_RUN.stage('02_blinded_label_audit'):
    audit=json.loads((RUN/'label_audit/audit_results.json').read_text())
    assert file_hash(RUN/'label_audit/blinded_ai_review.json')==audit['review_sha256']
    table=pd.DataFrame(audit['per_teacher_class']).T
    display(table)
    print('Balanced48 training-image panel, AI versus AI; not a teacher-error estimate.')
    print('No labels automatically changed:', not audit['automatic_relabeling'])
    print('Original AI-review mistakes and corrections:', RUN/'label_audit/reviewer_corrections.json')
    fig,ax=plt.subplots(figsize=(7,3.5));ax.bar(table.index,table['outside_reviewer_plausible_range'])
    ax.set_ylim(0,12);ax.set_ylabel('Flagged cases out of12 reviewed');ax.set_title('Diagnostic disagreement, not ground-truth errors')
    fig.tight_layout();fig.savefig(REPORT_RUN.path/'figures/audit_flags.png',dpi=150);display(fig);plt.close(fig)
""")
    code.append("""with REPORT_RUN.stage('03_training_only_recovery'):
    results=json.loads((RUN/'training/results.json').read_text())
    table=pd.DataFrame([{'family':r['family'], 'training_nested_OOF_macroF1':r['nested_training_oof']['macro_f1'],
        'development_macroF1':r['validation_descriptive_only']['macro_f1'],
        'development_Medium_recall':r['validation_descriptive_only']['recall'][1],
        'development_Critical_recall':r['validation_descriptive_only']['recall'][3],
        'head_only_CPU_p50_ms':r['head_only_cpu_ms']['p50'],
        'selected_config':str(r['selected_on_training']['config'])} for r in results])
    display(table);table.to_csv(REPORT_RUN.path/'metrics/recovery_comparison.csv',index=False)
    selected=json.loads((RUN/'training/selection_frozen.json').read_text())
    print('Frozen training-only family:',selected['family'])
    print('Winning OOF score is family-selection-exposed; development is historically exposed.')
    print('No new final-test performance claim. Prior-correction final tau0 equals RF control.')
    fig,axes=plt.subplots(1,2,figsize=(11,3.8));names=['Low','Medium','High','Critical'];ix=np.arange(4)
    for j,r in enumerate(results):
        axes[0].bar(ix+(j-1)*.23,r['nested_training_oof']['recall'],.23,label=r['family'])
        axes[1].bar(ix+(j-1)*.23,r['validation_descriptive_only']['recall'],.23,label=r['family'])
    for ax,title in zip(axes,['Training nested OOF (selection-exposed)','Development (descriptive only)']):
        ax.set_xticks(ix,names);ax.set_ylim(0,1);ax.axhline(.7,color='red',ls='--',label='readiness threshold');ax.set_title(title);ax.set_ylabel('Recall')
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(REPORT_RUN.path/'figures/recovery_recall.png',dpi=150);display(fig);plt.close(fig)
""")
    code.append("""with REPORT_RUN.stage('04_runtime_and_label_protocol'):
    parity=json.loads((RUN/'training/image_runtime_check.json').read_text())
    print('Ten real training images through encoder+head; max probability parity error:',parity['max_parity_error'])
    print('Actual bin checkpoint bytes:',parity['checkpoint_verification']['bytes'])
    print('Matches pinned upstream LFS SHA:',parity['checkpoint_verification']['local_sha256']==parity['checkpoint_verification']['hf_lfs_sha256'])
    from experiments.spi_classifier_repair.evidence_priority import PROMPT_SHA256,PROTOCOL
    print('Evidence-first teacher candidate:',PROTOCOL,PROMPT_SHA256)
    print('Candidate prompt/parser implemented and unit-tested; NOT yet run as a VLM or accepted as a labeling teacher.')
    receipt=json.loads((RUN/'training/bundle/readiness.json').read_text())
    assert receipt['runtime_promotion_allowed'] is False
    print('Research bundle:',RUN/'training/bundle','Runtime promotion:',receipt['runtime_promotion_allowed'])
    print('No full parameter cache launched. Unknown remains unavailable, not a fabricated Low label.')
""")
    headings = ["Actual environment and execution scope", "Verify inputs and class support",
                "Review label adequacy without automatic relabeling", "Test a bounded training correction",
                "Verify image runtime and preserve the release boundary"]
    for heading, body in zip(headings, code):
        cells.extend([nbf.v4.new_markdown_cell('## '+heading), nbf.v4.new_code_cell(body)])
    cells.append(nbf.v4.new_markdown_cell("""## Decision

The current SPI classifier remains unready. The training correction prevents a
zero-recall candidate from silently winning on average score, but does not solve
minority recognition. The AI audit flags label problems and also exposes reviewer
fallibility. Class-balanced disagreement is neither human truth nor calibrated
teacher error. Do not clear the diagnostic gate or relabel images from agreement.

Keep four classes and the current parameter/header contract. Before another final
training run: verify evidence-first labels on resolved image-level cases, reserve
adequately supported unexposed groups, and then refit the fixed compact head.
Current300 cannot be made a fresh final test by reshuffling them.

The prior-correction hypothesis was motivated by [Menon et al., ICLR2021](https://research.google/pubs/long-tail-learning-via-logit-adjustment/).
It was tested on an already balanced forest and was **not retained** in the final
fit; no paper-level guarantee transfers to that forest experiment.

Reproduce fitting in a **new output directory** with `python -m
experiments.spi_classifier_repair.minority_recovery --features <verified CLIP
features.npz> --records <v20 records.json> --labels <v20 labels.json> --output
<new directory>`. This notebook reuses saved fitting results; it does not refit.
"""))
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec":{"name":"python3","display_name":"Project .venv Python","language":"python"},
        "language_info":{"name":"python"}})
    nbf.validate(nb)
    nbf.write(nb, notebook)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(nb, km=manager, timeout=180,
                            resources={"metadata":{"path":str(root)}})
    try:
        client.execute()
    finally:
        nbf.write(nb, notebook)
    print("EXECUTED NOTEBOOK", notebook, flush=True)
    return notebook


if __name__ == "__main__":
    build()
