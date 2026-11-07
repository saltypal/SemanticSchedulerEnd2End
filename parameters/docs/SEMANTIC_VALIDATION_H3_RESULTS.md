# Balanced parameter-validation pilot: verified findings

Run: `20261026h3`, private Kaggle notebook version 1, 26 October 2026 IST.
Source ZIP: `4963e4baf2cf051fb7632881a7d0ab199e3697704446321a463c059f75b2c2d0`.
Methods and diagram: [SEMANTIC_VALIDATION_V1.md](SEMANTIC_VALIDATION_V1.md).

## Outcome

The extraction experiment completed. The reviewer comparison is **partial**.
Qwen3.5-9B ran on GPU1 with thinking disabled and real NF4 modules activated,
but neither reviewer is qualified as an automatic semantic validator from these
results. No additional GPU run was launched after this bounded attempt.

Kaggle reported COMPLETE; the experiment correctly reported PARTIAL. Worker exit
code zero means the worker saved its outcome, not that its scientific gate passed.
Internal experiment time was 604.89 seconds. Two Tesla T4 GPUs were assigned.

## Extraction evidence

Twelve official LADI test-source images were balanced across absent buildings,
present buildings without an annotated affected condition, and affected buildings.
These are development observations with unknown event/site independence.

| Check | Result | Interpretation |
|---|---|---|
| Unique images / causal observations | 12 / 15 | Includes independent UE histories and exact repeats |
| Native building facts correct | 24/24 | Presence and affected-or-greater only |
| Seven runtime parameter families available | 15/15 observations each | Availability does not establish accuracy |
| Exact-repeat novelty / independent cold start | Passed | Does not verify natural cross-view novelty |
| Trusted time / geography | Unavailable | No fabricated metadata |
| Extraction time | 27.57 s inside extraction | Worker and setup overhead recorded separately |

CLIP, YOLO-World and the native classifier ran on GPU0. Spatial/complexity
results have no independent localization truth in this panel. Rarity remains
predicted concept frequency, not importance. These data do not establish that
the representation fully describes scenes or supports good scheduling.

## Reviewer evidence

Both models received the same planned 24 controls and six actual extracted JSONs.
The controls deliberately contain omissions and incorrect values. Full 12-parent
extraction and 48-control case files are preserved independently of this budget subset.

| Configuration | Controls attempted / valid | Actual JSONs attempted / valid | Outcome |
|---|---|---|---|
| Qwen3-VL-4B FP16, Transformers 4.57.6 | 24 / 24 | 1 / 0 | All control answers approved; then GPU OOM |
| Qwen3.5-9B NF4, Transformers 5.3.0 | 3 / 3 | 0 / 0 | Runtime projection exceeded remaining budget |

4B answered both booleans true on every control. Its consistency and sufficiency
balanced accuracies were each 0.50; joint correctness was 8/24. It rejected none
of the genuinely missing conditions. This is a substantive reviewer failure even
though its JSON format was valid. On the first real JSON it attempted a further
3.08 GiB allocation with only 2.13 GiB free and stopped. Five actual cases were
never attempted; they are not five wrong model judgments.

9B completed the three predefined technical cases with finite logits, valid JSON,
358 activated 4-bit linear modules, all parameters on `cuda:1`, and a recorded
model footprint of 7,765,103,400 bytes. Mean case time was 13.89 s. The predefined
1.2-margin estimate required 450.01 s for the remaining 27 cases versus 441.78 s
available, so the worker stopped without changing its gate after seeing outcomes.

Only three cases support a paired model comparison: each model was jointly correct
on one. 4B approved all three; 9B rejected all three. The reasons include obvious
logical mistakes: 9B claimed that a false prediction contradicted an identical false
reference, and treated an omission as a factual contradiction. Three observations
cannot establish model superiority or stable error rates. No actual JSON received
a successful VLM review in this run.

## Thinking-off verification

`enable_thinking=False` is recorded in settings, model provenance and every
completed review. The Qwen3.5 template rendered a closed empty thinking block;
generated responses contained direct JSON with no thinking tags. The 9B run used
the native Torch fallback because FLA and causal-conv1d were absent.

The log warning that `enable_thinking` is ignored by the processor's modality
arguments does not mean the template flag was ignored: inspected Transformers
5.3.0 renders custom template kwargs before passing remaining kwargs to the
image/text processor. This is supported by the saved template receipts and direct
responses. Source: [pinned ProcessorMixin implementation](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/processing_utils.py).

## Verification and decision

The frozen production checkpoint passed 80 tests. Independent analysis recomputed
saved audits and metrics, verified all 12 image hashes, checked runtime-label
separation, model placement, pixel limits, thinking receipts and activated NF4.
The results verifier was corrected to recover partial reviews without final files
and to keep control/actual and common-attempted comparisons separate. A copied
interrupted-run fixture passed; its original artifacts were unchanged.

Retrieval verified 83/83 global and 82/82 run checksum entries, with zero missing
or mismatched files. The executed notebook, four cell logs and all 12 images are
present. The initial Windows output-encoding error is preserved alongside its
UTF-8 retrieval correction. Receipts are tracked in `KAGGLE_RUNS.md`.
Recomputed evidence: `results/ops/20261026_semantic_validation/verified_h3/verification.json`.

Retain deterministic availability/ontology checks and independent native-label
comparison as the primary evidence. Do not promote either VLM or select parameters
by maximizing its approvals. The current narrow scene-claim success does not
validate every scalar parameter. Before another GPU comparison, measure real-JSON
prompt size and peak memory first, include an actual representation in technical
preflight, and test a short explicit fact-comparison prompt on positive and negative
controls. A new prompt is a new development experiment; archived results stay intact.
The next large parameter or scheduler study remains deferred until these failures
are resolved under a separately bounded run.
