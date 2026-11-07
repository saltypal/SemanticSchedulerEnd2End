# Main study orchestration

The canonical paired-run controller is `paired_runtime.py`. `master_suite.py`
retains the full-suite driver. Reporting and merge logic live in
`summarize_saved.py` and `merge_paired_runs.py`.

The active notebook builders are:

```sh
python -m Scheduler.training.study.build_paired_notebooks
python -m Scheduler.training.study.build_teacher_recovery_notebook
```

They write to `Notebooks/SchedulerSuite`. They do not submit Kaggle jobs.
Existing account identities and input attachments remain unchanged. Run outputs
remain under their configured `experiments/` hierarchy. Old notebook files and
their published version identities remain historical evidence.
