# Disaster Decoder V3: receiver-side ordering with compact evidence

**Research report — 23 October 2026.** Kaggle training version 1 completed in 470.9175 seconds. The 480-file package and 39-row comparison passed integrity checks, and all 37,440 exported-model episode orders were independently reproduced. The final captured-output notebook version 2 also completed in 297.4164 seconds, matched the original metrics and saved three executed cells, a full table and two embedded figures. Conclusions remain bounded to this controlled observation study.

## Question and scope

The study asks whether compact task-conditioned evidence in a digital header can help a receiver order a set of already-arrived, complete latent jobs. The candidate ranker sees parsed header fields, not images. The experiment does not schedule partial fragments, alter a codec, control a real radio/MAC, or establish reconstruction-quality improvement. It also does not establish emergency priority or support life-safety decisions.

Real LADI images and frozen-model evidence fed constructed UE/queue/repeat scenarios. The benchmark replayed message identities and semantic headers, not actual encoded latent bytes or codec inference. The implemented receiver wrapper accepts opaque complete-payload references; codec execution remained a separate integration gate.

## Header and evidence fields

The version-3 header is a 32-byte digital control record: `>BBHIIQ8B` prefix followed by big-endian CRC32 over the first 28 bytes. It carries version, availability mask, task-profile ID, message token, transmitter-memory version, capture time, and eight quantized scores. Missing fields encode as zero bytes but remain unavailable under the mask; missing is not a measured zero.

| Field | Meaning and extraction rule | Main caveat |
|---|---|---|
| R | Task alignment: mapped cosine similarity `(cos+1)/2` from compatible embeddings or supplied relevance. | Similarity is not truth or utility. |
| S | Semantic rarity/self-information over supported eligible concepts in the transmitter-local observation history. | Requires supported concept evidence and history; it is not global rarity. |
| Q | Confidence in the task profile's critical claim, `max(q, 1-q)`, only with reliable held-out B-role temperature calibration. | Confidence is not correctness; otherwise unavailable. |
| I | Task importance from ordered native LADI logits, sigmoid probabilities, descending isotonic projection, and profile weights. | LADI-small is a reference predictor, not ground truth. |
| F | Trusted freshness `exp(-max(0, now-capture)/tau)`; default `tau=30,000 ms`. | Untrusted or invalid timestamps remain unavailable. |
| P | Largest retained detector area for the profile target, divided by image area. | Detector evidence depends on thresholding, clipping, NMS, and detector success. |
| C | Detection-scene proxy: half normalized log-count plus half normalized class entropy over retained building/vehicle/person/tree/bridge detections. | Not a calibrated measure of cognitive load or damage severity. |
| N | Local embedding novelty `(1-max cosine)/2` against compatible recent transmitter observations. | First observation scores 1; missing/incompatible history is unavailable. |

Task profiles are building damage (1001), road damage (1002), and structural flooding (1003). Their native-label weights and eligible concepts are defined authoritatively in `SemParam/decoder_parameters.py`; the profile coefficients are also recorded in `PROTOCOL.json`. Transmitter history has bounded local capacity and a query-before-commit sequence so the current observation does not leak into its own history-derived score.

The wire contract includes CRC, profile and availability validation, stable token/version/time fields, and parse-before-rank behavior. It is a study header contract, not a measured radio packet format or an accounting of latent payload bits/channel uses.

## Pipeline and implementation boundaries

1. A pinned LADI manifest is validated and source-image IDs are deduplicated. Conflicting labels for identical image bytes are excluded as a group; records are retained for audit.
2. Frozen producers generate image-only candidate evidence: CLIP ViT-B/32 for alignment embeddings, LADI-small as a reference native-label signal, and Grounding DINO-tiny for object boxes. Offline Qwen3-VL-4B generates a separately marked silver proposal subset; its outputs are not substituted into the main ranker's features or reference labels.
3. `SemParam/decoder_parameters.py` creates task/history-aware values and serializes the header. Image producers and cost accounting are in `SemParam/decoder_extractors.py` and `decoder_costs.py`.
4. `Scheduler/decoder_scheduler.py` provides FIFO, round-robin, random, importance-heuristic and learned/model-backed ranking policy interfaces. Policies select only among complete jobs; queue lifecycle remains outside the policy.
5. `INFRA/learning/decoder_v3_study.py` owns split/episode construction, ranker fitting, comparison, and replay. Extraction orchestration is in `decoder_v3_extract.py`; the Ollama reviewer wrapper is `decoder_v3_ollama.py`.

Receiver policy inputs are the eight parsed score values plus eight availability indicators (16 features). Raw images, labels, and latent payload bytes are not receiver features. This separation supports a controlled comparison but does not constitute a deployable device implementation.

## Dataset, labels, and split handling

The pinned source revision is `5f2dbfe8c466d32edafd1bab847ec5252309acdb` (LADI validation/test download). Source split counts are validation 890 and test 1,049. Four source rows represent two exact-byte image IDs with conflicting native labels; those IDs are excluded, not majority-voted or relabeled. Consistent duplicate images are extracted once.

The Stage A audit reports 1,939 manifest rows and 1,936 unique manifest IDs, with four conflict rows excluded, leaving 1,934 retained unique images. Role counts after exclusion/deduplication are A=445, B=178, C=267, test=1,044. The extraction summary's test role count 1,045 is pre-deduplication; the artifact audit's unique test count is 1,044. Group-disjoint validation roles and feature/label ID equality passed audit. Test was extracted as part of the artifact, but must remain held out for evaluation until analysis choices are frozen.

The roles have these protocol uses: A supplies scheduler training queries and imputation; B fits only parameter calibration; C selects scheduler hyperparameters, stopping epochs and methods; test evaluates frozen choices. No scheduler selection uses B or test. Controlled coverage is a sampling design, not verified coverage of all facts or incidents. Different photos do not prove different physical incidents.

Native source labels and Qwen's 32 valid schema proposals are different evidence tiers. Qwen annotations are silver and must be evaluated separately; they are not substituted for native labels or treated as independent ground truth. The LADI classifier output is a candidate signal and reference, not a gold label.

## Training and comparison plan

The benchmark ran 13 methods total: 12 baselines plus main CAHR. All methods used the same fixed 128 A-role queries per task, not a newly drawn set each epoch. Neural training allowed up to 60 epochs with patience 10. The L2 grid was `[1e-5, 1e-4, 1e-3]`; tree leaves were `[7, 15, 31]`, maximum 200 iterations, learning rate 0.05. C selected the task-specific deployment candidate. The reported comparison scores below are held-out descriptive results; test data were not used for method selection.

After all 13 methods finished on core seed 0, building-damage seed 1 and 2 were run for CAHR and the C-selected coverage-tree comparator. Trees were explicitly rerun with those seeds. The nine retrained building-damage ablations removed each of the eight fields in turn and jointly removed `S` and `N`; all files passed the subsequent artifact audit.

CAHR minus the seed-matched tree comparator was -0.0654 at seed 1 (pool CI [-0.1529, 0.0192]) and -0.0936 at seed 2 ([-0.1761, -0.0177]). These initializations did not rescue a superiority claim. No test-best seed was selected.

The ablation evidence was not uniformly favorable to all eight fields: removing I reduced A10 by 0.1365; removing N reduced it by 0.0209, but jointly removing S and N changed it by +0.0047 with a CI spanning zero. Removing F changed nothing, as F was unavailable in all natural-image queues. Nine ablation comparisons are exploratory diagnostics without multiplicity correction; they do not prove every field is necessary.

Calibration availability also mattered: Q was unavailable for roads because B contained only six positive critical-claim examples, below the declared ten-positive gate. Building and flooding each had ten such positives. F was unavailable for every task; S was available for approximately 64%, 76%, and 82% of candidate occurrences for building, road, and flooding. An eight-field implementation is not evidence that eight independently useful features were measured on this dataset.

### Exact methods and learning targets

| Method | Training objective or decision rule |
|---|---|
| FIFO | Arrival time and stable tie order |
| Round robin | Rotation among source queues |
| Seeded random | Reproducible identity-based random ordering |
| Importance | Parsed `I` only |
| Fixed formula | `I * (1 + lambda * N)`, lambda selected on C from 0, .25, .5, 1 |
| Priority classifier | Cross-entropy on native task utility categories; expected category utility orders jobs |
| Pointwise regressor | MSE against normalized native utility |
| Linear RankNet | Pairwise logistic loss on unequal per-observation utilities |
| Neural RankNet | Same ordinary pairwise objective, 16-32-16-1 MLP |
| Linear coverage ranker | Exact coverage-swap-weighted pairwise logistic loss |
| Neural coverage ranker (CAHR) | Same coverage objective, 1,089-parameter MLP |
| LambdaMART | Standard LightGBM lambdarank objective |
| Coverage-gradient trees | Custom detached coverage-pair gradients and positive diagonal Hessian approximation |

The scorer receives `[R,S,Q,I,F,P,C,N,R_available,...,N_available]`. Imputation medians are fitted on A; wholly unavailable fields use .5 with a cleared availability flag. Available parameters are first quantized to the 32-byte header and parsed back before training. Scores are ordering values, not calibrated urgency probabilities.

For a queue with evidence-unit identities `E_i`, utility `w_e`, and total offered unique utility `W`, coverage after the kth service is `C_k = sum(w_e for unique units covered by positions 1..k) / W`. Primary `A10 = mean(C_1,...,C_10)`. Zero-utility queues are undefined and counted, not assigned artificial perfect scores. An exact swap computes `delta = A10(swapped_order) - A10(current_order)`. Its absolute value weights the pairwise loss; its sign defines the preferred order. Weights are normalized within each queue and detached from differentiation, and an explicit L2 penalty is added. A queue-independent scalar scorer cannot recover relations absent from the header, even with this training target.

The main objective uses exact-coverage swaps and weighted pairwise logistic ranking loss. For a candidate pair `(i,j)` with target order `y_ij ∈ {-1,+1}` and learned scores `s_i,s_j`, the pairwise component is `w_ij log(1 + exp(-y_ij(s_i-s_j)))`; exact-coverage swaps compare the task-level utility before and after a proposed ordering change while preserving candidate coverage. The exact implementation and weights in the training code/protocol govern execution; this description does not imply training completion.

## Measured comparison

The published `comparison.csv` has 39 rows: all 13 methods for each task. Entries below are the actual A10 values, rounded to four decimals; the source CSV retains full precision. Each task offered 32 disjoint source pools and 960 evaluation episodes. Of those pools, 24 building, 10 road and 32 flooding pools contributed defined A10. There were 376, 752, and 96 undefined-A10 episodes respectively; zero-utility episodes remained undefined, not assigned a score. The pool, not generated episode, is the statistical unit. These are positive-offer-conditional metrics, not an accuracy assessment of all negative observations.

| Method | Building damage A10 | Road damage A10 | Structural flooding A10 |
|---|---:|---:|---:|
| FIFO | 0.2269 | 0.2732 | 0.2509 |
| Round robin | 0.2462 | 0.2497 | 0.2543 |
| Random | 0.2405 | 0.2412 | 0.2521 |
| Importance (I) | 0.7580 | 0.5953 | 0.6331 |
| Fixed formula | 0.7578 | 0.5953 | 0.6306 |
| Classifier | 0.6923 | 0.4820 | 0.6392 |
| Regressor | 0.6582 | 0.4725 | 0.6299 |
| Linear RankNet | 0.6714 | 0.3818 | 0.5630 |
| Neural RankNet | 0.7024 | 0.5270 | 0.6439 |
| Linear coverage ranker | 0.6712 | 0.4226 | 0.5403 |
| Neural coverage ranker (CAHR) | 0.7634 | 0.5301 | 0.6317 |
| LambdaMART | 0.7530 | 0.4843 | 0.6218 |
| Coverage-gradient trees | 0.7778 | 0.5585 | 0.6050 |

C-selected deployment choices, kept separate from the held-out comparison, were coverage-gradient trees for building damage, the pointwise regressor for road damage, and coverage-gradient trees for structural flooding. A10 is not uniformly won by CAHR: it is below the C-selected method on building and above it on road and flooding, while the simple `I` baseline exceeds CAHR on road and flooding. The results do not support universal superiority.

Paired source-pool bootstrap comparisons of CAHR minus each task's C-selected candidate yielded building `-0.014396` (95% CI `[-0.036446, 0.006614]`, 24 pools), road `+0.057567` (`[0.000000, 0.159633]`, 10 pools), and flooding `+0.026629` (`[-0.018008, 0.073700]`, 32 pools). The intervals include zero for building and flooding; the road lower bound is zero at the reported precision. These comparisons do not establish a uniform or generally robust advantage. Pools are disjoint source pools, not independent disaster events.

The evaluation also reports coverage@5/@10, linear-utility-gain nDCG@10 with `1/log2(rank+1)` discount, high-utility recall@10, redundancy@10, and streaming diagnostics. Pure scheduler median/p95 latency is saved separately from scheduler-plus-metric timing; use `scheduler_latency.json` for inference-cost claims. Streaming uses abstract service ticks, not codec or radio milliseconds.

## Evaluation and analysis

The published comparison preserves the shared query bank and matched candidate pools. Do not select methods using the held-out table: model selection was performed on C. Report all 13 methods, task-separated results, paired uncertainty using source-pool resampling, protocol-level undefined counts, and split provenance. Do not treat 960 generated episodes per task as independent statistical observations.

- Static episodes contain 32 messages from four UEs, with all jobs ready at time zero and equal abstract service cost. The five fixed strata are no injected repeat, within-UE 25%/50% repeats, and cross-UE 25%/50% repeats. Six randomizations are generated per stratum and disjoint 32-source-image evaluation pool.
- Repeated copies of an original image share one controlled observation identity. Their utility counts only once for coverage. Different source images are not assumed to establish different physical facts beyond this controlled observation benchmark.
- Supporting metrics are coverage at 5/10, importance nDCG@10, highest-utility unique-observation recall@10, repeated-observation fraction@10, mean/p95 waiting time, service counts per UE, and inference latency. nDCG can reward redundant high-importance copies and is not the primary metric.
- Repetitions are averaged within strata, then strata within source pools. Comparisons use 10,000 paired bootstrap resamples of the disjoint source pools. Generated queues are not counted as independent disaster events. Task results remain separate; any macro-average gives tasks equal weight.
- Streaming replay uses one abstract service tick per complete job, finite traces at loads .5 and 1.2, and a no-backlog control. Completion precedes arrivals and dispatch at each decision time. Queues drain; future arrivals are invisible. The no-backlog control must preserve arrival order for every policy. These ticks are not measured codec or radio milliseconds.
- Natural-image freshness stays unavailable without a trusted capture timestamp. Its implementation is exercised with explicit synthetic-clock tests, not presented as a learned real-world temporal effect.

The Stage A run did measure registered-operation inference counts: CLIP 8.81762304 GFLOPs, LADI 32.6971392 GFLOPs, DINO shape-dependent 665.607–746.230 GFLOPs. Eight actual device cost profiles were recorded, four per Tesla T4. These figures are operation-registry counts rather than comprehensive hardware FLOPs or latency results. Qwen's full FLOPs were not measured; partial analytic text estimates must not be compared to the image-model figures. Two T4s and a local Ollama process are confirmed by the audited artifact, not a statement about Kaggle training hardware.

| Frozen producer | Registered arithmetic per image | Median measured forward time on T4 |
|---|---:|---:|
| CLIP image branch | 8.818 GFLOPs | 5.96 ms |
| LADI reference classifier | 32.697 GFLOPs | 14.92 ms |
| Grounding DINO | 665.607–746.230 GFLOPs, shape-dependent | 490.89 ms |

Timing medians summarize 24 warm forward measurements per producer across eight image profiles and two workers. Preprocessing, host/device transfers, image loading, model loading and unsupported arithmetic are excluded. Three fixed CLIP task prompts are embedded once per worker, not once per image. Qwen ran offline after the frozen extractors were unloaded, using Ollama across both GPUs; it is not a runtime scheduling dependency. These are server-GPU measurements, not measurements on a phone/UE.

Grounding DINO dominates the measured extraction cost. The I-only policy can in principle avoid the CLIP/detector branches, while the full eight-field producer used them to calculate R/N and P/C. The benchmark does not demonstrate that this extra extraction cost is justified for every task. CPU parameter/history math and header serialization were also timed separately on cached real evidence; those Windows-host timings must not be added to T4 forward timings as if they were a measured end-to-end device latency.

### Qwen annotation usefulness correction (23 October follow-up)

Direct inspection of all 32 saved Qwen assessments found **zero assessable cases
for each of the three tasks**. All responses were syntactically valid but abstained
on building damage, road damage and structural flooding. Thus the annotation pilot
completed technically but did not produce usable semantic annotations. These outputs
were not training targets. The valid-output count must not be interpreted as annotation
quality or successful semantic evaluation. The failure cause has not been established.

### Remaining scientific limitations

- The benchmark uses real images but constructed source assignment, ordering and repetitions. It has no independently reviewed physical-asset correspondence or emergency preference labels. Coverage is not a casualty/rescue objective.
- Each benchmark UE contributes eight observations and starts with empty history. A 64-observation implementation and eviction tests do not establish long-lived, multi-session semantic-history performance.
- Cross-UE duplication cannot always be identified from eight scalars without an asset identity or embedding. The cross-UE strata expose this information limit; they do not eliminate it.
- Only 11 official-test images have the road critical-positive label; road uncertainty is accordingly wide. The 32 source pools are not known independent disasters. Event-level uncertainty and external-domain generalization are not established.
- Calibration is estimated from sparse positives. A completed temperature fit is not proof that Q is well calibrated in every operating condition; Brier/ECE/bin counts are reported in each task's calibration audit.
- The primary neural model is not consistently stronger than simple semantic importance or trees. Additional seeds also expose initialization sensitivity. No test-driven new architecture or test-best seed was substituted.
- Freshness, true deadlines, variable actual decoder costs, fragmentation, radio errors and reconstruction evidence are not empirically validated here. Finite overloaded queues drain after arrivals stop; this does not guarantee bounded latency or prevent starvation in an endless overloaded stream.
- The fixed shared 128-query training bank is smaller than a fresh-queue-per-epoch augmentation campaign. It matches training workloads across neural/tree competitors, but limits workload diversity; the choice was recorded before the test run.
- The original 24-hour target was not met. Completion status is based on the saved runs/audits, not on that target. No important legacy files, datasets or teammate codec files were removed, and no commit/push was performed.

## Verified checkpoint and remaining gates

Stage A run `results/disaster_decoder_v3/20261022_stageA_v3` passed `artifact_audit.json`: 1,934 unique finite feature rows and labels match; embeddings have dimension 512; all three image-model statuses succeeded for all 1,934 IDs; role groups are disjoint; 8 cost profiles are present; 32 Qwen proposals pass schema; two conflicting IDs are absent; and the source snapshot/artifact hashes match. The artifact includes source and environment provenance.

Kaggle private kernel `hmmmkk/disaster-eight-field-scheduler`, version 1, completed in 470.9174869 seconds. Stage B outputs are under `results/disaster_decoder_v3/20261023_stageB_v1/full/`. Each task has C selection records, 13-method comparison and streaming outputs. Additional seeds and all nine retrained ablations belong to building damage only. The run records `codec_evaluation=false` and `test_used_for_selection=false`. Its package audit verified all 480 selected files, seven source hashes, JSON finiteness and all 12 compressed trace files (1,205,010 valid rows). Post-analysis reproduced all 37,440 method/episode orders from loaded artifacts and checked serialized-header agreement, seeds/ablations and streaming integrity. See `results/disaster_decoder_v3/20261023_analysis_v1/audit.json`.

The `INFRA/learning/decoder_v3_study.py` source hash matches the published source. The combined current suite passed 53 tests in 48.45 seconds, including receiver ingress/dispatch, resumed-training parity, three-task orchestration and report aggregation checks. Final notebook version 2 reused completed checkpoints without test retuning and matched all 39 primary quality rows within 1e-12. Its embedded PNGs match the downloaded figure files byte-for-byte, its HTML table has 39 data rows, and no error outputs are present.

Figure 1 in the original run used a simple episode average. The final report-only correction averages repetitions within strata and strata within source pools, matching primary A10 exactly. Plotted competitors are fixed FIFO/importance/CAHR/ordinary neural RankNet/C-selected baseline, not the test-best curves. No predictions or tabulated metrics changed. Original and corrected artifacts are preserved separately.

A Graphify AST-only refresh was attempted but could not run because `graphify` is unavailable; no package was installed for it. This documentation-tool limitation does not affect scheduler execution. The prototype gates are complete; journal readiness, real codec integration, external/event-level validation and independent semantic-fact labels are not established.

## Reproducibility and artifact provenance

The run and artifact files are indexed in [ARTIFACT_INDEX.md](ARTIFACT_INDEX.md). The audited Stage A artifact reports source commit `02580b0921265ac0f0c30177d6ea60e7f8e6e368`, but also records a dirty source state. Therefore the commit alone does not identify the executed code; use the recorded source-file SHA-256 snapshot and uploaded source ZIP digest. Reproduction also requires the pinned dataset revision, frozen-model revisions, runtime/environment, artifact hashes, and the same resolved protocol. The extraction cache is not a substitute for the eventual ranker checkpoint/results.

## Related work and novelty boundary

LADI V2 is the source dataset/model context, not a basis for claiming a new semantic-priority construct ([Scheele, Picchione, and Liu, 2024](https://arxiv.org/abs/2406.02780)). Pairwise learning-to-rank has established foundations, including RankNet/LambdaRank/LambdaMART ([Burges, 2010](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/)). Novelty/diversity ranking is also established ([Clarke et al., SIGIR 2008](https://plg.uwaterloo.ca/~gvcormac/novelty.pdf)); image-text alignment representations are established in CLIP ([Radford et al., ICML 2021](https://proceedings.mlr.press/v139/radford21a.html)). This work should not claim novelty merely for combining semantic scores with rankers, adding a learned model, or including a novelty term. Any contribution claim must be narrowed to a demonstrated protocol/engineering result and supported by the completed matched experiment and a separate literature review.
