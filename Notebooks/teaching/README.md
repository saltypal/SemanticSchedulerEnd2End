# Local scheduler teaching notebook

Start with `semantic_scheduler_walkthrough.ipynb`. It imports the current receiver
modules and explains each operation with an invented four-message fixture. It is
an instructional experiment, not a publication benchmark. The last section reads
previous real SwinJSCC artifacts without repeating inference.

From the SemanticSchedulerBuild repository root, using PowerShell:

```powershell
& '..\SemanticSchedulerEnd2End\.venv\Scripts\python.exe' scripts/run_scheduler_teaching_notebook.py
```

The runner uses the invoking interpreter as the kernel, without a global kernel
registration or package installation. Every execution creates a unique directory
under `results/decoder_context/teaching/`. The notebook creates its own run
directory, so do not pre-create `SCHEDULER_LESSON_RUN` if running manually.

Artifacts in a successful run:

- `semantic_scheduler_walkthrough.executed.ipynb`: all executed cells and rich outputs.
- `semantic_scheduler_walkthrough.html`: browser-readable export when nbconvert exists.
- `manifest.json`, `config.json`, `validation.json`: actual environment, source identity and scope.
- `logs/`: incremental text output for each computational cell.
- `metrics/`: received headers, decisions, coverage and historical codec provenance.
- `figures/`: fixture illustrations, coverage graph and historical codec comparison.
- `models/fixture_ridge.json`: fitted teaching model; never use as a production model.

The sibling `<run-id>-execution` directory retains the input notebook hash, local
kernel specification, per-cell checkpoint notebook and any error traceback. To
recover from failure, inspect that partial notebook and error, correct the cause,
then start a fresh run. There is no optimizer checkpoint or expensive training to
resume here. No network service or Jupyter server needs to be launched by the runner.

Regenerate the editable notebook from its reviewed source with:

```powershell
& '..\SemanticSchedulerEnd2End\.venv\Scripts\python.exe' scripts/build_scheduler_teaching_notebook.py
```

The builder replaces the editable notebook; preserve any personal edits before
regenerating it. Executed runs and historical outputs are never overwritten.
