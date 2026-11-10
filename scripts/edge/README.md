# scripts/edge: Kaggle notebook builder and ops

Config: `config/edge/kaggle.yaml` (usernames, per-account `config_dir`, `runner: wsl`).
Never create a new kernel identity per run: slug is `edge-<step_id>`; pushes add versions.

Build a thin notebook (clone main, pip install -e ., run entrypoint, write manifest):

    python -m scripts.edge.build_notebook --step e01 --entrypoint Evaluation.edge.run \
        --title "Edge e01" --account win -- --config config/edge/sim.yaml

Writes `Notebooks/Edge/e01.ipynb` and `Notebooks/Edge/e01.kernel-metadata.json`.

Operate (put `--dry-run` BEFORE the subcommand to print commands only):

    python -m scripts.edge.kaggle_ops --dry-run push --account win --step e01
    python -m scripts.edge.kaggle_ops status --account wsl --step e01
    python -m scripts.edge.kaggle_ops wait --account win --step e01 --timeout 7200 --poll 60
    python -m scripts.edge.kaggle_ops pull --account win --step e01 --out results/edge/e01
    python -m scripts.edge.kaggle_ops list --account win

Successful push/pull appends a row to `parameters/edge_kaggle_runs.md`.
Push stages files under `Notebooks/Edge/_stage/<account>/<step>`.
