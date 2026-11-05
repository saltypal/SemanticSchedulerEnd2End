# Encoder alternatives and Jev boundary

Jev is TypeSafe's typed decision model, not JEPA. Official image docs https://docs.typesafe.ai/images and release https://typesafe.ai/blog/jev-1-14 describe experimental image input. Choice/Score could support a direct SPI assessment, but schema-correct output does not establish scene-label correctness or calibrated urgency. No free authorized endpoint, local weights or architecture reproducibility has been verified here. No Jev API calls or image uploads were made.

Frozen feature candidates:
- Existing local SigLIP base patch16: first zero-download diagnostic. Image-text alignment could support relevance as well as SPI if separately validated.
- SigLIP2 base patch16: official model https://huggingface.co/google/siglip2-base-patch16-224 describes extended training for semantic understanding/localization/dense features. Not yet downloaded/tested.
- DINOv2: official https://github.com/facebookresearch/dinov2 provides visual-only features with linear probing. Tests a distinct representation; it cannot directly replace CLIP's text-alignment role.
- I-JEPA: optional later visual-only frozen probe, not training from scratch. Do not confuse it with Jev.

Current experiment frozen_probe.py uses the same300 verified images/labels, a frozen local SigLIP encoder, and matched balanced-linear head selection for both CLIP and SigLIP. Three C values, three training-only grouped folds; evaluate existing96 development rows once per chosen head. Full-frame pad224 and model-specific normalization preserve field of view. CPU-only, no model download or VLM rerun. Exact local environment differs from earlier cloud; compare models within this run, not absolute scores from different selection environments. No production/header/KB change or automatic promotion.
