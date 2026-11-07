# Qwen9B review-only results — 20261026i

The experiment completed all 30 planned reviews. It reused checksum-pinned h3
images and parameter JSONs; no image extraction or 4B inference was repeated.
This is development evidence on six previously exposed source parents.

## Execution

- Kaggle version 1: https://www.kaggle.com/code/satyapaladugu/betterscheduler-qwen9b-review-20261026i
- Experiment status: complete; worker return code 0; internal elapsed 625.36 s.
- Qwen3.5-9B, NF4/FP16 compute, Transformers 5.3.0, on GPU1 of two assigned T4s.
- Thinking OFF verified for all 30 responses; all 30 valid JSON; no memory failure.
- Maximum input 5,159 tokens; maximum measured allocated GPU memory 11.811 GiB.
- Largest serialized real JSON was reviewed first and completed successfully.
- Source ZIP:`6cd5e8abc3d99113eff24b53a7b601bf0bc08db6687eae2e5caf08828b01a566`.

The last-position-logit setting and 9B-only runtime completed this workload.
This does not isolate which change explains the earlier 4B memory failure.

## Independent recomputation

The analysis independently recomputed every candidate audit, expected verdict
and saved model metric. All 12 reused image hashes matched. Runtime bundle checks
found no forbidden evaluator-label keys. Model placement, NF4 activation, finite
logits,pixel limits and thinking-off receipts passed.

| Cohort | Valid reviews | Both verdicts correct | Interpretation |
|---|---:|---:|---|
| Constructed controls |24/24|16/24|Qwen is not a reliable automatic auditor yet |
| Actual extracted JSONs |6/6|4/6|Two correct absence representations were wrongly rejected |

Control consistency balanced accuracy: 0.7083; sufficiency balanced accuracy: 0.7500.
All 12 inconsistent controls were rejected. However, 7/12 consistent controls were
also rejected. For sufficiency, all 16 insufficient controls were rejected, but only
4/8 sufficient controls were accepted. Genuine missing conditions were rejected
in 4/4 cases; recoverable absence was accepted in 0/2 cases.

The predefined engineering gate therefore **failed**. Thresholds were not changed
after observing the results. Balanced accuracy on actual cases is undefined:
all six have a positive expected verdict under the two-fact reference contract.

## What the two actual-case errors mean

`actual-2` and `actual-3` explicitly contain:

```json
{
  "claims": {
    "building.present": false,
    "building.affected_or_greater": false
  },
  "claim_availability": {
    "building.present": {"available": true, "reason": null},
    "building.affected_or_greater": {"available": true, "reason": null}
  }
}
```

Both native annotation values are also false. Qwen nevertheless said the required
claims were missing because buildings were not detected. It confused an explicit
negative prediction with unavailable evidence. The independent comparison correctly
retains these as represented negative claims; it does not infer absence merely
from missing detector boxes.

Control errors additionally show missed availability masks, omission confused
with contradiction, and a response claiming that false contradicts false. The
raw reasons are preserved. The 4/6 acceptance rate must not be reported as parameter
accuracy: all six match the two supported native building facts,while Qwen makes
two false rejections. This does not validate the other scene claims or scalar
parameter constructs.

## Decision

The review now runs end to end within the budget, including real parameter JSONs.
The remaining demonstrated problem is reviewer reasoning reliability. Keep the
independent native-label comparison and deterministic availability/decoding checks
as primary evidence. Do not change parameter predictions merely to obtain Qwen
approval, or promote 9B as ground truth. A further prompt/claim-format experiment
must retain these failures and test both positive and negative controls.

Recomputed evidence:
`results/ops/20261026_semantic_validation/verified_i/verification.json`.
Runtime outputs:
`results/ops/20261025T2356Z/review_only_package/retrieved/results/semantic_validation/20261026i/`.
Operational integrity and retrieval receipts are tracked in KAGGLE_RUNS.md.
The downloaded experiment-output manifest verifies 69/69 entries with no missing
or mismatched output artifacts. The downloaded source manifest matches the submitted
archive; all 69 archive entries were verified separately. Runtime Python source
files were not individually downloaded by the selected retrieval filter.
