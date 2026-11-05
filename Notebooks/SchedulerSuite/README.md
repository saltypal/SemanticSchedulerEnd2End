# Main scheduler notebooks

`paired/wsl` and `paired/windows` contain the current canonical account notebook
sources. `recovery` contains the retained targeted recovery notebook. Builders
live under `Scheduler/training/study`.

These files replace editable-source ownership under `experiments/master_study`.
Historical notebooks remain unchanged. New notebooks embed the canonical source
packages and retain the existing Kaggle owner and slug. No launch occurs merely
by generating them.

`snapshot_template.ipynb` supplies the inherited capsule inventory to the builder.
It is a build template, not an alternate notebook to submit. `inputs/attachment.json`
records the existing input attachment metadata; it contains no dataset payload.
