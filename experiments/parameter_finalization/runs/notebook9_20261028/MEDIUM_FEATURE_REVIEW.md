# Runtime feature re-extraction review

Bounded assessment-only recheck, 2026-10-28. **Clearance: no consequential blocker found in the inspected feature correction.** No source edits, GPU/model runs, downloads, or test execution performed by reviewer. Test source was inspected. Mutable source hashes below identify this reviewed state; packaging must include it.

The correction creates a genuinely new image-feature cache, rather than relabeling the old signature or asserting numerical equivalence. `extract_runtime_features` validates image content and grouped splits, enforces batch ownership, finite nonzero vectors and one producer signature, then atomically publishes a new NPZ with the exact record IDs/content hashes. Existing input features remain untouched. No native urgency labels, post text, fitting targets, or future sender memory enter extraction. Mission text is the fixed visual preflight mission; MinimalVisionProvider obtains its image vectors independently of mission similarity.

Notebook cell 4 routes CONFIG.features_path to this fresh NPZ before label export and grouped student fitting; final parameter caching constructs the same MinimalVisionProvider implementation with the same checkpoint/device in the unchanged runtime. Its classifier prediction still enforces the newly trained feature signature. The preflight compares the completed new cache signature back to the already observed fresh producer signature. Wrapper mismatch handling now warns and delegates to that explicit recomputation path; it does not mutate the old cache or disable the final classifier guard. The input manifest/profile check continues to validate the immutable old input bundle; it is not used to stamp the new feature identity.

Resource scope remains one existing 300-image cohort and the already selected CLIP model. The correction adds at most one complete CLIP extraction pass when signatures differ, before expensive teacher labeling; no new model, dataset, training sweep or external service is introduced. No measured cost or numerical agreement claim is established by this review. Native-image identity and split boundaries are preserved. The normal grouped SPI support/recall/export gates still control parameter caching.

Limits: a mid-extraction failure leaves the new output directory and any partial file, without a completed feature NPZ or provenance; the enclosing notebook recorder must preserve the traceback. Automatic resume is not added. Actual cloud preflight, fresh-cache numerical behavior, final source packaging and GPU execution remain operational validation, not findings cleared by static review. No equivalence between the old and new embeddings is claimed.

## Reviewed source identity

```json
{
  "parameters/dataset_pipeline.py": "45de4609bd167a7e0ae86ac800078ec1f6973e56be13bdcdff2875a87312ab15",
  "parameters/minimal_vision.py": "9bf4a63e311f5f891929e0c194d34f081d9921945599fc86f726e093ced77542",
  "experiments/parameter_finalization/notebooks/Full_dataset_Labeling_and_parameter_caching.ipynb": "afc61e31ca1dcfb9b5355705bcf117d9d5785ddc76c16ad8da24aaf9002bb494",
  "experiments/parameter_finalization/runs/notebook9_20261028/prepare_submission.py": "4b7519a6568fb497c141cb18e45e96b247a6f29f0f55108e940b22941ba789bb"
}
```
