# SQ-GAN codec v1 fine-tune: hand-off

Status: scaffold, never run on Kaggle. Expect to fix small things on the first run (see "If it fails").

## Where things are
- Notebook: `Notebooks/Edge/edge-sqgan-finetune-v1.ipynb` (+ `.kernel-metadata.json`, slug `satyapaladuguplus/edge-sqgan-finetune-v1`).
- The only file you edit: `config/edge/sqgan_finetune.yaml` (commit it to `main`; the notebook clones latest `main`).
- Class map for the segmentation teacher: `config/edge/seg_classes.yaml`. Rows marked `# DECISION:` need your call
  (water replaces Cityscapes `rider`, boat replaces `train`, where tents/rubble/signs go, ...).
- Also decide in the YAML (all marked `# DECISION:`): image sources, `small_image_policy`, `epochs`, teacher revision pin.

## Inputs (attach as notebook datasets; already listed in the metadata)
- `satyapaladuguplus/scheduler-dataset-v1-images` (has `index.jsonl`, `images/...`)
- `satyapaladuguplus/scheduler-dataset-v1-labels` (has `labels_raw.jsonl`, later `splits_v1.json`)
- `satyapaladuguplus/semantic-lossless-private-sqgan-source`, `...-sqgan-assets`, `semantic-scheduler-sqgan-bundle-20261102`.
Roots are auto-discovered under `/kaggle/input` (3 levels). Override with `--images-root` / `--labels-root`.

## Run
1. Open the notebook on Kaggle, accelerator GPU T4 x2, Internet on. Run all.
2. Cell "Trainer source" prints whether Semantic_Lossless came from the pinned GitHub commit or a Kaggle fallback. Read it.
3. Until `splits_v1.json` exists the run stops with a refusal. For a trial only, add `--allow-provisional-splits` to the run
   cell. It prints a loud warning; do not use that bundle for results. Only train-split images are read; val/test never.
4. Output: `/kaggle/working/runs/edge-sqgan-v1/bundle/` (bundle.json, last.pt, assets/checkpoint/, portable_config.json).
   The run ends by verifying the bundle (size + sha256 of every file).

## Upload
Download the `bundle` folder from the notebook output. Create a Kaggle dataset named exactly
`satyapaladuguplus/scheduler-sqgan-v1` containing that folder as `bundle/` (keep it a folder; do not rename files).

## Register in `config/edge/codec.yaml`
1. On any machine: `python -m Codec.edge.bundle_check <path>/bundle --deep --print-codec-yaml`
2. Paste the printed lines into `codecs.v1`; set `bundle_subdir: "bundle"` and keep `kaggle_dataset: satyapaladuguplus/scheduler-sqgan-v1`.
3. Confirm `last_pt_sha256` is the 64-hex value printed; commit to `main`.
4. Tell the pipeline owner: `Codec/integration/sqgan_payload.py` must apply `Codec/edge/seg_map.project_logits` in
   `semantic_map` for bundles that contain `seg_classes.yaml` (the teacher emits 150 ADE channels, the codec needs 20).

## If it fails
- Clone error: repo private -> attach a dataset containing `src/sqgan_train.py` (the banner says so).
- "Parents smaller than native crop": the trainer needs images of at least 256x512; use `small_image_policy: upscale`.
- Teacher label mismatch: the ADE name table differs from the model's `id2label`; fix `seg_classes.yaml`.
- Out of time: lower `train.epochs` or raise nothing past `max_seconds` 36000 (12 h session limit).
