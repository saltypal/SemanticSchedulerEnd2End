# Proposed two-GPU parameter/VLM pilot

25October2026. Status: methodology and model recommendation,not a launched notebook
or measured VLM comparison. User requested extraction methods be explained before
implementation. The follow-up explicitly defines this as exploratory experimentation:
document failures,revise parameters/methods from evidence,and do not claim a universal
best set or optimize the extractor simply to win reviewer approval.

## 1. Current execution state

The existing BetterScheduler preparation notebook reports COMPLETE; the local
automatic downstream launcher is stopped. Bounded account/status audit found no
confirmed running BetterScheduler job to cancel. Extract/select slugs are absent
from account search and their direct status is denied,not a verified running run.
No notebook was deleted or cancellation session ID guessed. Stop-check receipts:
`../results/ops/20261025T105121Z/stop_check/`. Mamba status separately reported ERROR.
No replacement notebook has been pushed in this proposal.

## 2. Model choice and evidence

Recommend `Qwen/Qwen3-VL-4B-Instruct` as primary reviewer. Use an instruction-tuned
non-thinking model for short bounded binary JSON reviews. `google/gemma-3-4b-it`
is a credible fallback/comparison candidate if legitimate weight access is available.
Do not assume a previous Qwen failure was model incompetence without its error/logs.

Published benchmarks,not results measured by this project:

| Benchmark as reported | Qwen3-VL-4B-Instruct | Gemma3 4B IT |
|---|---:|---:|
| MMMU validation |67.4|48.8|
| MathVista mini/testmini |73.7|50.0|
| HallusionBench |57.6|Not found in inspected IT table|
| RealWorldQA |70.9|Not found in inspected IT table|
| BLINK |65.8|Not found in inspected IT table|

Qwen values were read from the official model-card multimodal benchmark image
at https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-VL/qwen3vl_4b_8b_vl_instruct.jpg
(preserved in evidence/). Gemma values are from Google's model card:
https://ai.google.dev/gemma/docs/core/model_card_3 . Do not mix Gemma pretrained
(PT) results with instruction-tuned (IT) results or Qwen4B with8B. Preserve exact
benchmark versions/prompts/settings when recoverable. Cross-provider tables are
not a controlled head-to-head experiment and do not prove reviewer reliability.

Why select Qwen4B provisionally:published small-model visual performance,relevant
multi-image support,public Apache2.0 checkpoint,and a plausible single16GB GPU
footprint at bounded input sizes. Gemma weights require acceptance/access under
its model terms on Hugging Face; access must be verified,not bypassed. Larger
Qwen models are not justified before the4B reliability/cost pilot.

Four billion16-bit weights are roughly8GB decimal before vision/KV/activation and
framework overhead; this is a capacity estimate,not measured VRAM. Actual total
parameter count/storage,input resolution,sequence length and GPU allocations are
measured in preflight. Do not claim it fits from this arithmetic alone.

Journal wording before experiments: "We selected a candidate using published
visual benchmarks and hardware constraints,then assessed its reliability on
annotation-supported controls." After experiments add actual configuration,
precision,quality/format failures,latency and memory. Do not write a measured
accuracy/speed advantage over Gemma unless both were actually evaluated identically.

Sources:
- https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/raw/main/README.md
- https://ai.google.dev/gemma/docs/core/model_card_3
- https://huggingface.co/google/gemma-3-4b-it

## 3. Requested GPU layout supersedes the previous DataParallel plan

```text
GPU0: CLIP + one YOLO-World producer -> parameter JSON
                    bounded,versioned queue
GPU1: Qwen3-VL-4B-Instruct -> binary review JSON + explanation
CPU: causal KB,geometry,rarity,metadata,serialization,logging and evaluation
```

This is role partitioning,not torch.nn.DataParallel. It supersedes the earlier
plan to scatter each model across both GPUs. Bind reviewer explicitly to cuda:1,
producer to cuda:0; do not use automatic placement that spills reviewer ontoGPU0.
Require two actual assigned GPUs and record every model's parameter device map,
allocation/peak memory,work counts and timing. Two16GB devices are not pooled
into one32GB device. Use a bounded queue so producers cannot exhaust RAM/disk
while reviewer is slower. Independent workers own their models; CPU owner serializes
sender KB updates. No concurrent mutation of the same KB.

For a dualT4 runtime,start with FP16 and a supported PyTorch attention path; do
not assume nativeBF16 or require FlashAttention2. Use bounded image pixels,
short prompts,max256new output tokens and batch1 first. Any quantization fallback
changes the experiment configuration and needs the same reviewer controls. Freeze
the managed Kaggle torch/CUDA stack; pin compatible Transformers/producer versions.

## 4. Exact initial extraction methods proposed for review

### UE task profiles and context relevance

The user explicitly requests UE-specific purposes and KBs. Luna High drafted six
profiles; main review made the everyday task measurable,narrowed annotation claims,
and made the two building-UE mission definitions exactly identical. The runtime
registry is `config/ue_task_profiles.json`,loaded by `ue_profiles.py`. These are
operator-defined experimental tasks,not labels inferred from an image or priority
weights. Device type is provenance only and does not determine importance.

| UE | Simulated equipment | Assigned purpose |
|---|---|---|
| ue01 | Inspection drone A | Document visible building damage |
| ue02 | Inspection drone B | Same building-damage task,independent history |
| ue03 | Access-survey drone | Document visible road obstructions |
| ue04 | Vehicle camera | Document visible physical road damage |
| ue05 | Personal phone | Inventory visible household furniture/appliances |
| ue06 | Field phone | Collect informative visual incident documentation |

Six UEs,five unique mission definitions. The same building prompt/mission version
is shared byue01/ue02. Their KBs are separate,so R should agree for identical
image/task while N may differ. No mission includes the protected structural-flooding
task. Do not infer a disaster-negative from an everyday phone image or vice versa.

Context relevance means alignment of visible content with the UE's assigned task:

`R_raw(I,q_u) = cosine(normalize(CLIP_image(I)),normalize(CLIP_text(q_u)))`.

`q_u` is the short visual task clause recorded in mission.conditions,not the UE
name/device type and not a task selected after seeing the image. Preserve the
original operator paragraph too. Cache each task embedding with task text,mission
version,tokenizer/model hashes and context limit. Reject oversized text rather
than silently truncate. The current single-clause profile makes max aggregation
identical to this one cosine; compound tasks require a separately specified rule.

Record raw cosine in[-1,1]. If the existing bundle uses `(R_raw+1)/2`,preserve that
mapping explicitly; neither is a calibrated relevance probability. High scores
mean model alignment,not priority or urgency. Compare tasks per-profile before
making cross-task claims because prompt baselines may differ. No automatic
"drone>phone" rule or fixed semantic importance weights.

Validation controls:

1. Cross the same eligible images with all five task texts using cached image
   features. Do not assign only matching pictures to each UE and then claim
   relevance discrimination. Include relevant,irrelevant and hard-confusion cases
   only where references establish the relation; unknown cells remain unknown.
2. Building evidence versus household inventory is an easy contrast where supported;
   road damage versus blockage is a harder condition/relation contrast. A blocked
   road is not necessarily damaged,and an intact-looking image is not an explicitly
   annotated negative without support. Keep multiclass/multilabel overlap.
3. Replaying the same image/task under different UE histories must not alter R.
   The same image with another task may change R. Test frozen paraphrases within
   development roles and a shuffled-task control; avoid final-set prompt tuning.
4. Give Qwen the applicable operator paragraph,short task text and definitions
   with the image/JSON. History is supplied only for history-dependent checks.
   Hide equipment labels and priority claims from the judge to reduce device bias.
5. Native-reference support:building conditions use compatible LADI/RescueNet
   definitions; blockage uses an exact supported road-blocked annotation,not merely
   LADI roads_damage; road damage uses documented damage labels without inventing
   surface-specific causes; household presence uses covered COCO categories,not
   a universal everyday/no-disaster label; MEDIC informative supports only its
   declared informativeness task,not universal humanitarian usefulness.

Each UE owns a KB keyed by `(ue_id,mission_version)` with64prior observations.
Mission reassignment starts a new version-specific KB in the pilot; do not
silently mix histories or overwrite the old state. Visual embeddings can be
cached globally,but concept counts/retrieval histories remain UE-local. Log image
identity separately from observation/message identity. Query before committing
the current observation. Sender KB is not receiver delivered history.

Initial series use the six assigned UE bindings without claiming an actual radio
deployment. Preserve a chronological per-UE observation manifest and explicit
synthetic-order status. Include same mission/different history and different
mission/same image controls. UE IDs and dataset IDs never enter learned semantic
arrays; the actual mission text/version,not an equipment stereotype,defines R.

Profiles/KB isolation and registry plumbing are locally testable without vision
inference. They do not establish that CLIP/Qwen agrees with human relevance.
The new VLM notebook must bundle this registry and its hash. The old uploaded
campaign is not updated or resumed by these profile additions.

### Producer methods

One small YOLO-Worldv2 checkpoint (`yolov8s-worldv2.pt` candidate,available in the
Ultralytics implementation) and existing CLIP ViT-B/32. Pin upstream checkpoint
hash and implementation; do not download or install them while merely proposing.
Ultralytics documents migration from the official YOLO-World weights:
https://docs.ultralytics.com/models/yolo-world/ . Record implementation/model license
and do not transfer published FPS/AP to our data or hardware.

| Parameter/evidence | Initial extraction | Interpretation and check |
|---|---|---|
| Context relevance | Normalized CLIP image versus explicit short mission-clause text embeddings; retain per-clause cosine and a declared any-of scalar baseline | Alignment,not calibrated probability. Compound AND/negation is not silently reduced to max similarity |
| Scene/entity/condition evidence | YOLO-World classes/boxes; fixed CLIP place/event/condition prompt similarities | Overlapping predicted evidence,not a certified incident/distress classifier. A domain-specific existing LADI head may be a separately exposed comparator,not hidden primary truth |
| Novelty | Sender KB of normalized embeddings; nearest-cosine appearance distance; separately log newly detected concept changes | Primary appearance proxy,not proof of new asset/event. Query before commit; exact-repeat/new-fact controls and relevant prior images supplied to reviewer |
| Scarcity | Per-concept predicted image-presence counts in last64sender observations with Laplace-smoothed Bernoulli frequency and surprisal | Defined history/population,support and cold start. Rare is not automatically important. This requires extending current condition-probability-only rarity path; do not silently claim it is already implemented |
| Spatial prominence | After canonical labels,NMS and coordinate clipping:target-box area/image area,target count and union target-box coverage | Box extent,not physical proximity or visible mask area. Thresholds have declared development selection; not portable confidence probabilities |
| Scene complexity | Relevant entity count and relevant coverage,class diversity; all-object count/entropy retained as diagnostic | Information-content proxies whose validity is tested; no arbitrary count-to-priority formula |
| Reliability | Preserve per-object/condition score,method and calibration status; unavailable for claims with no valid probability | Do not average scores into generic image importance or call CLIP cosine a probability |
| Timestamp/deadline | Capture time,current trusted time,operator allowed delay; deterministic age/deadline/slack | Missing in static cohorts unless trustworthy metadata exists; synthetic timing separately labeled |
| Geographic/source | Trusted coordinates against explicit mission region; declared source capability | Unavailable without real metadata. Dataset/source identity not an importance feature |
| Task importance | Preserve oldIonly as historical baseline; report supported conditions/mission relevance separately | No new hand-weighted priority law and no operational urgency inferred from objects |

Do not change extraction methods after seeing final reviewer results to maximize
yes frequency. Method revision requires a new version and held development controls.
If an extra condition or relationship is necessary,test that specific addition
instead of adding a fleet of models or more scalar formulas.

## 5. Pilot gates: reviewer must work before full extraction

### A. Technical preflight

Load processor/model onGPU1; verify image tensors are present and their hashes and
pixel-token shapes differ for distinct images. Run a tiny known-content check.
Verify actualGPU device,dtype,attention path,generation termination and raw output.
Preserve stderr/warnings,not merely an API-success flag. Batch1,one/two images
first; no broad extraction if OOM/format/device/vision-input tests fail.

### B. Reviewer discrimination controls

Use24clear,independently annotated development images,with one reference-compatible
JSON view and one deliberately contradicted view each:48binary review cases.
These are reference-constructed controls,not claims that our extractor is already
correct. Keep unavailable fields explicit and do not invent factual positives.

Readiness correction:48cases test consistency only. Add a third,factually
compatible but incomplete view for each image:72review cases total. Freeze a
per-case required-fact checklist before Qwen sees the case. Expected binary pairs:
complete+compatible=(true,true),incomplete+compatible=(true,false),contradicted=
(false,false) under the conjunctive adequacy rule below. This adds sufficiency
controls,not extra models or datasets. Cases without independently supportable
required facts do not enter the reference-graded sufficiency panel.

The rubric contains three small pieces:assessable qualitative claims,required
mission facts,and numeric/history fields verified by code. Correctness means all
assessed assertions are supported. Sufficiency means those assertions are supported
AND every required fact for this case is represented. Qwen receives the criterion
names/rules but not the expected facts/answers; the answer-key instance is evaluator
only. Never define enough as "describes everything visible" or add a rich external
caption to a header-only view. A scalar cannot be credited with a missing category
merely because the reviewer can see it in the image.

Example rubric families,refined/frozen per annotated case:building-damage task
requires supported building presence/condition;blockage requires supported road
obstruction;household inventory requires the declared covered item information;
incident-informativeness requires only native/reference-compatible criteria.
Unsupported specific damage levels,physical identity and urgency are excluded.
Label coverage must be sufficient to judge the chosen required facts,not assumed
complete because a dataset has some labels.

Primary review mode is visible_parameter_json:Qwen sees the actual candidate JSON,
image,mission and definitions,as requested by the user. Hide evaluator references,
expected verdicts,control names,CLIP diagnostic scores and policy identity. Do not
claim parameter values are hidden in this mode. An optional diagnostic
blinded_atomic_claims mode removes numeric parameter values/producer confidences
to investigate anchoring on a bounded panel. Keep the modes explicitly tagged and
their estimates separate;do not silently mix their protocols.

Examples of controlled mismatch:wrong entity,overstated target extent,known wrong
condition where native labels support the contrast. Include same JSON paired
with distinct images whose compatible labels differ,so a text-only approval
strategy is exposed. Match record length/format to avoid trivial typo detection.

Qwen sees image,actual proposed JSON,fixed definitions and mission; it returns
`parameters_consistent` and `representation_sufficient` booleans plus reason and
field-level binary checks. Numeric geometry/frequency/time correctness is checked
in code; the VLM tests qualitative consistency. Only assess context/history fields
when the required inputs are present. An input-sufficiency/skip flag is logged
outside binary verdicts rather than pretending an unknowable field is false.

Do not ask Qwen to certify an exactCLIP cosine,relevance probability,rarity count,
calibration or KB computation from pixels. Verify calculations from frozen inputs
in code and validate predicted relevance against compatible reference labels.
If R is reviewed qualitatively,freeze its text interpretation during development,
do not choose thresholds to make model approval higher,and state that this is
model-assessed plausibility rather than numerical calibration.

For controls with a defined binary reference,report confusion matrix,balanced
accuracy,true-compatible acceptance,true-incompatible rejection,invalid-output
fraction and paired image sensitivity. An all-yes or all-no reviewer fails.

Provisional go/no-go engineering gate,not a journal significance threshold:
>=95%valid JSON on72cases,>=80%balanced correctness,>=75%each reference class,
plus compatible direction on a prespecified image-swap control panel. Freeze
these gates before execution and report uncertainty/raw counts. Borderline results
require a new independently supported control panel,not retries until passage.
Passing clear controls qualifies an exploratory reviewer,not reliable judgment
of all subtle conditions or all domains.

Report the two endpoints separately:consistency balanced accuracy uses its48true/
24false cases;conjunctive adequacy uses24true/48false cases. Also require rejection
of incompleteness on at least18of24incomplete cases before calling the sufficiency
review operational. Balance by reference class,not raw72-case accuracy. Retain
per-image paired results and intervals;these engineering gates are not proof of
generalization. Freeze before execution,not after a result is seen.

### C. Parameter pilot after reviewer gate

Use at most100initial parent images and12six-observation series. Inspect real
parameter distributions/missingness,source cases,binary reviewer disagreement
and deterministic reference errors. Preserve all-no outcomes:they may indicate
reviewer/prompt failure,incorrect extraction or insufficient representation.
Classify causes before altering parameters. Parameter validity is not defined as
the largest model yes rate.

If Gemma access is available and Qwen fails the semantic controls,run the same
bounded controls on Gemma3 4B before deciding. Do not switch because a different
reviewer is easier to satisfy. Resource/access failure and visual-quality failure
are different causes; log them separately.

### D. Bounded scale only after review of pilot evidence

Expand to200,then1,000images if useful new evidence and time permit. The maximum
remains10,000parents/20GBcurated downloads/8aggregateGPUhours across stages. Count
both allocated GPUs:8aggregateGPUhours permit at most4wallhours of a twoGPU job,
including setup/idle/model comparisons. Stop on time budget,invalid JSON bursts,
missing vision input or degenerate verdicts; preserve partial outputs.

"As many as possible" means within these limits after the pilot,not a compulsory
10k-image run. Reuse prepared images; no duplicate20GB source acquisition. A
continuation must use a matching source/config/data snapshot and a recorded
remaining budget. Failed controls do not trigger automatic full-data launch.

## 6. Results,revision and paper records

Remaining implementation gates before an actual notebook launch:

- YOLO-World adapter,Qwen loading/generation/JSON wrapper and role-partitioned
  notebook are not yet wired. Unit/profile checks do not certify GPU inference.
- Current SenderMemory stores embeddings/probabilities only. Add a joined immutable
  observation sidecar (image/observation/source/mission IDs,concept-presence validity,
  preprocessing/model hashes) for prior-image review and detected-concept rarity.
  Reconstruct history from logged causally available observations,not future frames.
- Freeze canonical object aliases/groups (for example TV/television and vehicle
  subcategories) and task/reference mappings. Exact label-string matching must not
  silently zero valid objects or turn a generic damage label into blockage truth.
- Freeze per-case required facts,control roles and missing-input/skip handling.
  Supply all inputs needed for each graded field;unavailable references are not
  counted as reviewer mistakes. Record invalid output separately from yes/no.
- Verify source access/support and remaining budget before scaling. First run
  one/two-image technical checks,then72reviewer controls,then<=100image extraction.

Persist model-choice sources/benchmark table,checkpoint hashes,actual runtime,
input resolutions,prompt/schema versions,per-case raw+parsed outputs,time/VRAM,
reviewer controls,parameter JSON,history traces and independent reference metrics.
Luna High performs bounded metric/trace recomputation; main owns fixes and synthesis.

Use a versioned failure ledger:

- Reviewer defect:missing pixels,format,always-no/yes,prompt anchoring,unsupported
  exact numerical expectations,claim-judgment error.
- Producer defect:wrong object/condition,missed small target,duplicate boxes,
  confidence misinterpretation or invalid coordinates.
- Representation defect:correct evidence discarded,ambiguous scalar semantics,
  missing condition/relation/history information.
- Reference gap:unsupported class/correspondence/metadata,not a proven model fault.

Retain/revise/unresolved decisions follow these causes. Examples:fix a unit or
missingness bug before inventing a parameter; compare task counts with raw entropy
when clutter drives complexity; add a bounded condition item if similar scalar
headers hide different annotated conditions. Keep V0results,change one mechanism
in V1,reuse paired development cases and confirm on untouched groups. No claim of
universal best parameters from this exploratory pilot or from Qwen agreement.
