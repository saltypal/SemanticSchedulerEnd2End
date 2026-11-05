# SQ-GAN — teammate 2

Run: `20261028T130100Z-eea4f3aa`. Training/evaluation status is in the accompanying JSON files.

Zip this run folder or the whole teammate folder for Drive/OneDrive. Keep each run separate. `last.pt` is the final saved checkpoint; `best.pt`, if present, is the development-selected checkpoint. These files include optimizer/progress state where the trainer supports it, but exact training resume is not implemented.

Use the matching repository revision and model dependencies recorded in `environment.json`/`source.json`. Paths in `portable_config.json` are relative to this folder. Dataset inputs are not bundled. Some author code and frozen teacher configurations are fetched from their pinned upstream releases. Missing constructor assets: none recorded.

After installing the recorded model dependencies, restore the model with:

```python
from src.sharing import load_bundle
model, options = load_bundle("path/to/this/run", device="cuda:0")
```
Some author implementations require CUDA even for inference.
