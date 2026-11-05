# Mainstream scheduler promotion

31 reusable source modules now belong to Scheduler/training/stream_suite and Scheduler/training/study. The existing Evaluation environment and Scheduler inference/model code remain their owners. Compatibility entry points preserve old Python imports and file commands.

Three rebuilt notebook sources live under Notebooks/SchedulerSuite. Existing Kaggle owners, slugs and input attachments are preserved. No notebook was submitted. Previous notebooks and result folders remain unchanged.

Validation: 111 focused tests passed with no skips. Six command checks passed from fresh extracted paired source capsules outside the checkout. All 31 modules parsed, and all three generated notebooks compiled. See validation.json and the review for scope. No training, final test access, model inference or data downloads occurred.

Commands from the repository root in the existing scheduler environment:

```sh
python -m Scheduler.training.stream_suite.train_real_suite --help
python -m Scheduler.training.stream_suite.ue_rate_study --help
python -m Scheduler.training.study.build_paired_notebooks
python -m pytest Scheduler/tests/test_mainstream_entrypoints.py Scheduler/tests/test_scene_learning_system.py experiments/master_study/tests experiments/stream_scheduler/tests -q
```

The three new pyproject console entry points require refreshing the editable installation before they appear in an already installed environment. Module and script entry points were exercised without reinstalling dependencies.
