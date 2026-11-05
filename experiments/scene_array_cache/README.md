# Scene-array caching

The frozen specification is [PLAN.md](PLAN.md). Open [Scene_Array_Caching.ipynb](Scene_Array_Caching.ipynb).
The notebook is prepared, not submitted to Kaggle. No active SPI run is modified.

From the canonical repository root using its existing environment:

```powershell
.venv/Scripts/python.exe -m pytest experiments/scene_array_cache/test_scene_arrays.py -q
.venv/Scripts/python.exe experiments/scene_array_cache/validate_notebook.py
```

The second command executes all notebook cells on generated fixtures only. The executed notebook,
logs, numerical evidence and figures are saved in `runs/fixture_validation_<UTC>/`.
It does not measure YOLO accuracy or real inference time.

Kaggle executes the default `real` mode, imports its managed torch/CUDA and installs the specified
Ultralytics/CLIP setup dependencies into that kernel. It processes the embedded, pinned 4,000-image
cohort. A fresh-kernel real run must pass the ten-image model preflight before bulk caching.

Optional environment variables:

| Name | Meaning |
|---|---|
| `SCENE_ARRAY_MODE` | `fixture` or `real` |
| `SCENE_ARRAY_RUN_ID` | Output run ID; UTC default |
| `SCENE_ARRAY_RESUME_DIR` | Existing matching run directory |
| `SCENE_ARRAY_DEVICE` | Bulk inference backend; Kaggle `cuda:0`, local `cpu` |
| `SCENE_ARRAY_IMAGE_ROOT` | Local real dataset root containing `data_image/` |
| `SCENE_ARRAY_WEIGHTS` | Verified local YOLO checkpoint |
| `SCENE_ARRAY_SECONDS` | Bulk session budget, default 14,400 seconds |

Scheduler handoff: use `scheduler_arrays.npz`'s `wire_features` with its IDs, success mask,
vocabulary/producer identity and documented feature layout. Counts are already log-transformed.
Join SPI, relevance, novelty and their masks separately. No reference labels are runtime inputs.
This is a 369-byte scene block; it does not fit the previous 35-byte header.

Vocabulary v2 contains all80standardCOCOclasses plus13sceneextensions (93total). See ALL_93_CLASSES.json. Scheduler vector374features; schema/weights expecting old130inputs are incompatible.
