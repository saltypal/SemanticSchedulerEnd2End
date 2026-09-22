# Stage 1C semantic parameter-codebook notebook

`01_train_parameter_codebook_local.ipynb` is a local engineering demonstration, not a research training pipeline. It requires at least ten SHA-256-distinct training images under `data/parameter_codebook/train`, rejects any hash overlap with `Test/images`, and saves a `demo_only=true` semantic parameter-codebook artifact. The current local demonstration corpus is the 24-image Kodak suite downloaded from `r0k.us`.

The runtime codebook uses a pre-modulation SwinJSCC encoder representation, deterministic pooling/projection, and nearest-codeword lookup. CLIP is used only offline to create the teacher targets; runtime codebook extraction must not import or call CLIP.

Regenerate the notebook from its repository source with:

```powershell
python scripts\build_stage1b_notebooks.py
```

Run the same training flow without Jupyter with:

```powershell
.\.venv\Scripts\python.exe scripts\train_parameter_codebook.py --device cpu
```
