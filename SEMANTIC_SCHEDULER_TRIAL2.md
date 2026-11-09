# Semantic-aware AI scheduler | Trial 2
Updated 8 November 2026. Research and design record.

Read this file before resuming. Keep branch provenance, scope decisions, measured results, paper links, and the next decision here. Mark new results as measured only after running their checks.


## Remaining work in execution order

- [ ] Test 1. Prepare LADI v2 same-SPI pairs and freeze grouped calibration, training, development and holdout splits.
- [ ] Test 1. Smoke-test the pinned Qwen and Nemotron on one T4 each. Run both image orders, check the qualification thresholds below, and save their judgments with model and prompt provenance.
- [ ] Test 1. Accept explicitly VLM-labeled manifests in the trainer. Extract SQ-GAN features for the cohort and train the eight-byte RankNet scorer on stable consensus training pairs.
- [ ] Test 1. On untouched development pairs, compare with SPI-first plus arrival order and use both models to check whether cited evidence survives decoding. Report these as model-judged results. Close Test 1 before advancing.
- [ ] Test 2. Wire real SPI and trained scene bytes into pack(), validate with unpack() before queue admission, and verify a real image through the complete sender-to-decoder path.
- [ ] Test 2. Implement and test SPI tier, scene score, earliest known deadline for equal scene scores, then arrival order. Use arrival order within the tier when scene scores are unavailable. Supply deadlines through receiver queue metadata.
- [ ] Test 2. Run matched receiver comparisons and check the Critical dispatch guardrail. Run the separate uplink prioritization simulation under the same link budgets across policies.
- [ ] Test 3. Verify access, hash and splits for the private 6,000-image archive. Scale the qualified teacher on training pairs and compare masked PPO with matched supervised RankNet and the Test 2 winner on the locked Test 3 cohort.

Read this checklist and the active decisions below before each implementation session. Record each completed check, measured result and blocker in this file.

## Active review decision — Qwen and Nemotron

This decision supersedes the human-review prerequisite in the earlier design, runbook and execution history below. Qwen2.5-VL-7B-Instruct and Nemotron Nano VL 8B are the reviewers from Test 1 onward. Their outputs are model-generated judgments. Tests 1–3 use those judgments as proxies for scene usefulness and evidence retention.

1. **Test 1.** Move the pinned Qwen and Nemotron two-image NF4/FP16 smoke tests into Test 1. Verify each candidate can run on one T4. Freeze event, flight and near-duplicate groups across calibration, training, Test 1 development, and the separate locked Test 2 and Test 3 cohorts before fitting or choosing a teacher. Build same-SPI-tier LADI v2 pairs. Hide SPI values, arrival order, filenames and scheduler scores from the reviewers. Ask each model to grade visible impact 0–2, choose A, B, tie or uncertain, and cite visible evidence. Run both image orders and map answers back to image IDs.
2. **Qualify the reviewers.** Require each candidate to reach 90% mapped order consistency among pairs with two valid responses and at least 50% stable strict coverage across all calibration pairs. Require at least 80% agreement on the pairs where both give stable strict choices and report that overlap and its size. This replaces agreement against human labels. Audit identical-image controls and sharpness traps, and log abstentions, malformed outputs and disagreements. Choose the model with the most stable strict judgments on calibration, breaking ties by lower median latency. Use it as the primary labeler and the other as a cross-check. Keep only jointly supported strict choices for the initial fit. If the pilot lacks two qualifying candidates or enough consensus pairs to fit and evaluate, record the shortfall and revisit the rubric or cohort before advancing.
3. **Close Test 1.** Extend the manifest loader to accept explicitly VLM-labeled manifests with model revision, prompt version, group/split IDs and both orientations. Preserve support for the existing manifest format and keep VLM provenance distinct. Train RankNet on training pairs only. Compare against SPI-first with recorded FIFO ties on untouched development pairs. Ask the two models to check source and decoded images for the cited evidence and report retention and disagreement. Advance only when the codec checks and review-based development gates pass. Report preference agreement and evidence retention as model-judged metrics.
4. **Tests 2 and 3.** Keep the matched receiver and uplink replays and the Critical dispatch guardrail. Use frozen VLM judgments for semantic evaluation on the locked cohorts, with the same proxy interpretation. In Test 3, verify the private 6,000-image archive and scale the already qualified teacher to training pairs. Compare masked PPO with matched teacher-supervised RankNet and the Test 2 winner. Keep the locked Test 3 cohort out of teacher selection, training and tuning.


## Decision

SPI sets the priority tier. Within that tier, decode the image with the most useful visible disaster evidence first. A less sharp image showing definite, consequential flood impact should beat a sharp image with little disaster evidence. If the impact itself cannot be established from the image, mark the comparison uncertain. Image sharpness is relevant only insofar as it affects what can be established.

The trial uses only SPI and scene evidence for scoring. Remove COCO, YOLO, dedicated road-obstruction and structure readers, separate relevance inputs, novelty, duplicate cues, and sender/receiver history from the new design. The receiver still maintains its ready queue and lifecycle state. Older code and results serve as historical records.

## Branch provenance

Fetched these refs from saltypal/SemanticSchedulerEnd2End on 8 November 2026.

| Branch | Commit |
| --- | --- |
| main | 102831994f18925bb1fa0732c166ea57ba3b9198 |
| semantic-aware-ai-scheduler | 43aa658b7e91f5ab574f3aba920454add918c085 |
| scheduler_improvement_trials | c03a5d5070b3496e90457533c4626508f836fda4 |

The local worktree at sem_schedluerAI_trial2 is checked out on sem_schedluerAI_trial2 from the committed improvements tip. The original branch is its ancestor. The existing checkout contains uncommitted road/footprint changes and was left intact. The local branch has no upstream.

## Original scheduler audit

The default delay suite is priority-aware and delay-optimized. The reference tier sets a waiting budget b of 2, 4, 8 or 16 simulation time units. The penalty for waiting w is w/b + max(w/b - 1, 0). It stops increasing at dispatch and includes undispatched offers up to the episode horizon. These budgets are experiment settings.

Task D builds a target from the extra penalty of postponing an item by one time unit. Choice models use cross-entropy, regression uses Huber loss, RankNet uses pairwise logistic loss, and LambdaMART consumes quantized target grades through LambdaRank. PPO and Double DQN use the negative increase in episode cost in this arm. Equal-tier, equal-wait candidates receive the same target regardless of their scene evidence. The priority-classifier arm uses tier labels, which can teach the class but cannot resolve scene usefulness within a tier. The environment also has Task E, which targets uncovered exact-image facts, and Task P, which requires teacher labels. Trial 2 does not use exact-image coverage or the delay target to train its scene scorer.

Source checks are in Evaluation/semantic_learning_env.py, Scheduler/training/semantic_learning_models.py, Scheduler/training/semantic_learning_rl.py and Scheduler/spi_policy.py. The SPI selector already supports a semantic secondary score, deterministic FIFO ties and missing-score handling. Reuse that boundary.

The archived no-COCO trial took mean application overhead including CRC from 110.45 B to 84.16 B across 6,000 cached records. It pruned transmitted COCO slots while keeping the original sender prompts and a compatibility feature layout. In the 15-episode replay, SPI-first reached 67/75 Critical targets. Frozen no-COCO LambdaMART reached 38/75 and a reduced-feature refit reached 55/75. Those were silver tiers with synthetic service. The refit had a slightly lower aggregate waiting penalty than SPI-first (3.367 vs. 3.374), so the aggregate concealed a worse Critical outcome. The measured native SQ-GAN replays provide a service-time and packet test harness. Their failed road-reader trials provide no evidence for or against a new pre-mask, whole-scene head.

See docs/scheduler_improvement_trials/no_coco_full_header.md and docs/scheduler_improvement_trials/native_sqgan_extension.md.

The fetched main and original scheduler tips have identical trees for Codec, Evaluation, Scheduler, parameters and Notebooks/SchedulerSuite. Their differences outside those paths include repository documentation, archived run/report files, and the Kodak benchmark notebook. No separate main scheduler patch was required.

## SQ-GAN and wire design

1. Keep the frozen, verified SQ-GAN checkpoint, RGB preprocessing, existing 20-channel SegFormer conditioning, and SQ01 payload format. Configure the pinned source and bundle explicitly because the new worktree has a different relative layout. The existing encode(image, semantic=None) returns bytes. Design an optional encode_with_features path that returns the same bytes and the spatial image-encoder grid immediately before masking and VQ. Confirm the actual tensor shape and call count with the loaded bundle. The handoff's 256 x 16 x 32 is an expected shape. Use an explicit return from the shared encode path and keep the grid on the UE only for the current image.
2. Start with global mean and max pooling of the frozen grid followed by a small learned projection into eight bounded scene values. Mean pooling captures broad scene context and max pooling gives localized evidence a route into the descriptor. Train the projection against scene preference labels through the RankNet scorer. Use fake uint8 quantization in training and exact uint8 in evaluation. Keep a one-value projection as the size and complexity ablation. Defer the handoff's spatial segmentation head and 2 x 4 grid unless the compact head fails specifically on small visible evidence.
3. Propose 36 bytes of application overhead with CRC. The fixed prefix contains version (1 B), producer profile (2 B), SPI/scene availability flags (1 B), source/session/sequence (4 B each), payload length (4 B), four quantized SPI probabilities (4 B), and eight scene values (8 B). Four CRC bytes follow the payload and cover prefix plus payload. Use network byte order. Bind each profile to the SPI producer, checkpoint, preprocessing, projection and quantization. Normalize available SPI probabilities on receive. Distinguish available zero scene values from missing scene evidence. Reject invalid version/profile, length, flags or CRC before admitting a complete message. Use receiver arrival time and the source/session/sequence tuple for bookkeeping. These IDs never enter scores.
4. The 36 B target is a proposed format, including framing and CRC at the application layer. Against the earlier 84.16 B mean it would save about 48 B (57%) per message. That estimate combines scene replacement with a smaller envelope that drops sender timestamps, wait budget, relevance and novelty. Measure both complete formats over identical messages and report scene block bytes, application overhead, SQ01 payload bytes and transport overhead separately. Keep old packet decoding available for historical replay.
5. The SQ-GAN paper establishes task-conditioned compression. It does not establish that the frozen pre-mask grid encodes disaster impact, or that SQ01 reconstruction preserves the evidence a human needs. Test both. Verify byte-identical payloads and identical reconstructions on the same backend, with one encoder call. Then independently review paired originals and reconstructions for retention of the concrete evidence that determined priority. Stop at SPI-first if the compact representation cannot rank that evidence or the reconstruction cannot convey it. Changing the codec rate or retraining the codec is outside this design pass.

## Policy and supervision

Use the existing SPI selector. For each ready queue, select the highest available predicted SPI class, then sort that class by the scene score. Reuse the existing whole-snapshot FIFO fallback for unavailable SPI and within-class FIFO fallback for unavailable scene scores. Break equal scene scores within the highest SPI tier by earliest known deadline when every candidate in the tie has a deadline. If any deadline is missing or the deadlines tie, use receiver arrival order and stable sequence/identity. Validate tier mapping, including Medium and Low/Normal naming, against the producer profile. If a registered producer lacks a supported tier or valid probabilities, record that limitation and keep its unsupported cases out of four-tier performance claims. Wait, service time, IDs and past images stay out of the ranker inputs and targets.

Start with RankNet using a shared linear score over the compact descriptor and logistic loss on ordered pairs. It learns the exact choice we care about and adds little receiver work. Compare it against SPI-first/FIFO, the existing delay-based suite as a historical comparator, and LambdaMART trained on the same frozen, quantized descriptors and reviewed scene grades. For a fair ranker comparison, also refit the linear RankNet scorer on those same frozen descriptors using pairs derived from those grades. Group LambdaMART rows by eligible same-tier candidate set, skip equal-grade groups, and report extra direct pair labels as a separate supervision condition. Keep the simpler scorer when the more complex one cannot show a held-out improvement.

Use existing LADI v2 records with verified event provenance for the first whole-scene aerial pilot. Published tags help find positives and hard negatives, including flood/debris examples. They are not priority labels. Collect independent image-level review with three grades.

| Grade | Reviewer instruction |
| --- | --- |
| 0 | No visible disaster impact useful for assessment |
| 1 | Definite impact with limited apparent extent or assessment value |
| 2 | Definite consequential impact whose nature or extent materially improves assessment |

Reviewers cite what they can see. Use uncertain for unassessable evidence and tie for equally useful scenes. Exclude uncertain and tied labels from strict pair training and report their counts. Include direct challenge pairs where the sharper image shows less consequential impact. Separate event, source/flight and near-duplicate families before generating pairs or caching features. Exclude unresolved-event images from held-out claims. Keep reviewed test labels and reconstruction judgments independent of SPI, YOLO and the VLM.

A separate VLM can label training pairs offline. Give it the two source images, one disaster-assessment rubric and an instruction to cite visible evidence. Hide SPI outputs, scheduler scores, filenames and any history. Ask for A, B, tie or uncertain. Repeat with swapped image order. For the pilot, require at least 90% order consistency and 80% agreement on independently reviewed strict calibration pairs, with at least 50% coverage of those pairs. Check negatives, unclear evidence and sharper-but-less-useful images before accepting labels. Record the model revision, prompt, split, evidence and rejections. Count model labels as silver supervision and compare human-only training with human plus qualified VLM pairs on the same independent test.

Training RankNet on VLM preferences is supervised learning from AI feedback. True RLAIF would train a policy in rollouts using an AI-derived reward. Keep it out of the first build. Revisit the existing masked PPO arm only if matched replay identifies a multi-step advantage beyond the static scorer. The research design is to warm-start from RankNet, restrict actions to the highest SPI tier, and use frozen teacher-derived semantic utility earned only on successful completion before the fixed episode horizon. Give the supervised control the same teacher data and compare on independent labels. Qualifying a VLM on image pairs does not qualify it as a reward for complete schedules.

## Simulator and acceptance

- Reuse the nonpreemptive receiver and the measured codec replay harness. Admit only complete validated messages. Let new messages arrive during service, dispatch exactly one ready job at a time, and record each completion/failure once. Remove historical context, exact-repeat coverage and cached labels from the new policy snapshot. Keep reference tiers, usefulness grades and service times on the evaluator side. Expose only deadlines available to the receiver as queue metadata for tie-breaking. Use the same fixed deadline data in matched replays and keep deadlines out of RankNet inputs. The 36-byte packet has no deadline field.
- Replay identical offered jobs, payloads, service durations, failure schedules and horizons across policies. Use measured per-image codec timings on named hardware with synthetic arrivals under low load, contention and burst overload. Keep legacy synthetic traces as regression controls. Label both scenarios as simulations.
- Compare FIFO, SPI-first, SPI plus RankNet, and SPI plus matched LambdaMART. Report strict same-tier pair agreement and nDCG@3 on shared candidate sets with independent grades. Report independently graded useful evidence delivered by the locked episode horizon, by policy. Keep class-conditioned waiting, dispatch-target misses, completion misses, unfinished jobs and p95 wait alongside these metrics. Define the Critical guardrail before opening the holdout. The proposed pilot gate is at most a 2-percentage-point increase in Critical dispatch-target miss rate versus SPI-first, using the upper paired 95% interval bound. Use event-level paired intervals for semantic results and require a positive lower bound on the primary pair-agreement gain over SPI-first. Require a positive useful-evidence completion gain in contested replay before promoting the learned policy. Small or inconclusive cohorts remain development evidence.
- Round-trip the packet and test zero/unavailable features, corrupt or truncated packets, unknown profiles, FIFO ties, tier protection, concurrent arrivals, decode failure and horizon truncation. Measure UE preprocessing and scene overhead, peak memory, bytes and receiver selection latency on the same hardware. Audit the new sender and receiver path for imports or model loads involving YOLO/COCO, history or the retired readers.

## Reading list

AlphaXiv discovery and PDF review used the following papers. These supply design precedents. Trial 2 has no measured outcome yet.

| Paper | What to study for Trial 2 |
| --- | --- |
| [SQ-GAN: Semantic Image Communications Using Masked Vector Quantization](https://www.alphaxiv.org/abs/2502.09520) | Follow the image encoder, adaptive mask and VQ boundaries. This is the basis for exporting a pre-mask grid while preserving the codec. |
| [Transmit What You Need](https://www.alphaxiv.org/abs/2412.13646) | See why useful visual metadata depends on the downstream task. It motivates a small descriptor trained for disaster assessment. |
| [When Semantic Communication Meets Queueing](https://www.alphaxiv.org/abs/2605.05514) | See how learned-image transmission studies queue delay and downstream task fidelity together. Its action is a transmission rate choice. Our action is which complete message to decode. |
| [LADI v2](https://www.alphaxiv.org/abs/2406.02780) | Read the multi-event aerial image collection and annotation procedure. Use the tags for cohort construction and collect separate scene preference labels. |
| [From RankNet to LambdaRank to LambdaMART](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf) | Compare a direct pairwise loss with boosted trees and nDCG-weighted learning. The original technical report is from Microsoft Research. |
| [LLaVA-Critic](https://www.alphaxiv.org/abs/2410.02712) | Read how a visual critic generates pairwise judgments and supports preference learning. Use its design as a starting point for judge calibration. |
| [RLAIF-V](https://www.alphaxiv.org/abs/2405.17220) | Study image-grounded feedback and hallucination controls for generating silver preferences. |
| [RLAIF vs. RLHF](https://www.alphaxiv.org/abs/2309.00267) | Compare a reward model fitted on AI preferences with direct AI rewards. The reported tasks are language tasks. |
| [Tuning Large Multimodal Models for Videos using RLAIF](https://www.alphaxiv.org/abs/2402.03746) | See how grounded visual context informs a reward model and how they compare against supervised training. The experiments align video-language outputs. |

The handoff's RescueNet paper was also checked through AlphaXiv. It describes pixel annotations from Hurricane Michael. Keep it as a reference for a future localization diagnostic. The pilot uses whole-scene, multi-event review and omits dedicated road and structure readers. [RescueNet](https://www.alphaxiv.org/abs/2202.12361)

## Next decision

The first implementation gate, when authorized, is whether the pinned SQ-GAN encode path can return pre-mask features with deterministic payload parity, and whether those features preserve enough scene signal to beat SPI-first on an independently reviewed development set. Keep SPI-first as the active baseline until that gate and the matched replay pass.

## Execution runbook and gate log — 8 November 2026

Run one OpenCode session per gate with MiMo V2.6 Flash through the requested OpenCode Go route. Read AGENTS.md and this file at the start of each gate. Review the diff and measured evidence before advancing.

1. Test 1. Verify pinned source and weights. Export pre-mask spatial features in one encoder pass with byte and pixel parity. Train the mean/max-pooled eight-byte RankNet projection on independently reviewed same-tier scenes. Lock event, flight and duplicate-disjoint Test 2 and Test 3 cohorts. Validate the 36-byte SPI/scene packet, pair agreement over SPI-first, and evidence retention after decoding.
2. Test 2. Add the opt-in spi_scene_only independent-stream replay. Compare eligible rankers on matched steady and bursty loads 0.5, 0.9 and 1.2. Apply the paired preference, useful completion and Critical miss gates above. Compare earliest-deadline and FIFO tie-breaks with the same ranker and traces. Cover a later arrival with an earlier deadline, equal or missing deadlines, and higher SPI tier protection. Run the separate four-arm sender/receiver FIFO versus semantic uplink comparison with fixed byte-budget traces. Verify zero-delay unlimited-capacity parity. Carry a qualifying uplink policy into a separate 5G-LENA study.
3. Test 3. Verify Mini_dataset_6000.tar under experiments/image_pool/runs/ in the private Kaggle version 5 outputs. Archive SHA-256 is 0f8c6186b09dcf6ab87911df642612919377157bb1585082bed1c93b1506fe2c. Smoke-test NF4/FP16 Nemotron Nano VL 8B at 437f4e28b989cc2d9a16b6767cc930cdf48797ff and Qwen2.5-VL-7B-Instruct at cc594898137f460bfe9f0759e9844b3ce807cfb5 on one T4 each, sequentially. Qualify against the same reviewed and swapped pairs at 90% consistency, 80% accepted agreement and 50% coverage. Choose the most correct accepted judgments, then lower median latency. Run the selected frozen teacher on cuda:0 and the trainer on cuda:1 in a private T4 x2 session. Compare masked PPO, warm-started from RankNet, against matched teacher-supervised RankNet and the Test 2 winner. Apply the same Critical guardrail.

Dataset source. https://www.kaggle.com/code/satyapaladuguplus/semantic-image-pool-scene-cache-20261030/5/output

Test 1 launch preflight. OpenCode rejected opencode-go/mimo-v2.6-flash as unavailable. Its auth list showed no authenticated integrations. A no-edit smoke prompt returned OK on opencode/mimo-v2.6-flash-free. Awaiting route selection. Located the supplied SQ-GAN ZIP at 20261028T130100Z-eea4f3aa.zip. No Test 1 implementation or acceptance result yet.
Checkpoint preflight. Streamed last.pt directly from the supplied ZIP and verified SHA-256 706396402d86e84276393ce3fe39f0b0b1b545ff024c0ecae3b8ab72d22f3d6d against the native pilot record. This verifies the checkpoint bytes. The full bundle, source and runtime parity remain separate checks.
Bundle integrity preflight passed. All 27 manifest-listed ZIP members matched their declared byte sizes and SHA-256 hashes. Source qualification, codec parity and reviewed-scene acceptance remain pending. Searches found historical reviewer tooling but did not confirm a reviewed same-tier pair manifest for Trial 2.
User approved opencode/mimo-v2.6-flash-free for Test 1.

## Test 1 gate results — 8 November 2026

Scope honored: Test 1 only. No commit, no stage, no push, no credentials read, no Test 2/Test 3 work started, legacy Evaluation packet paths untouched. Engineering work below is complete and verified; the reviewed-data acceptance gate stays blocked exactly as stated in its own section.

### Staging (outside the repository)

Extracted the supplied ZIP to `TEMP/opencode/sqgan_stage`. Re-verified all 27 bundle members against `bundle.json` (27/27 exact byte size and SHA-256). Built a staged source root there (copy of the teammate package `src/`, `sources.lock.json`, and the pinned `frapez1/SQ-GAN` @ `0e5cdd7c0240b89749dad684645c53f783ac24cc` copy found under `SemanticLossless/Models/author_releases/SQ-GAN/`); `source_receipt.json` was written only after all 31 pinned files matched the verified bundle `source.json` hashes. The external teammate source at `the Semantic_Lossless checkout` was imported read-only and not edited. Only dependency change: `einops 0.8.2` installed locally (the masker module imports `rearrange`; nothing else was installed).

### Single-pass pre-mask export — real parity smoke: PASSED

Receipt: `docs/scheduler_improvement_trials/trial2_prefeature_parity.json`. Two real LADI test images, local CPU, torch 2.12.1+cpu, 2 threads, bundle run `20261028T130100Z-eea4f3aa`, checkpoint SHA `706396402d86e84276393ce3fe39f0b0b1b545ff024c0ecae3b8ab72d22f3d6d`, source commit `0e5cdd7c…`.

| Check | ladi test_row_0259 (flooding) | ladi test_row_0015 (buildings) |
| --- | --- | --- |
| Baseline encode deterministic (two runs) | true | true |
| `encode_with_features` payload byte-identical to baseline | true | true |
| Encoder calls during export | 1 | 1 |
| Reconstruction pixel-identical (torch.equal) | true | true |
| Feature shape / dtype | [1, 256, 16, 32] float32 | [1, 256, 16, 32] float32 |
| Payload bytes | 464 | 464 |

The observed grid matches the handoff's expected 256 x 16 x 32 and is confirmed with the loaded bundle. Timings: codec load 32,698 ms; teacher 37,536 ms first call then 637 ms; encode 12,074–13,545 ms; feature export 12,393–12,814 ms; decode 36,040–40,850 ms. Float32 grids for both images cached outside the repo (`sqgan_stage\grids`, 524,416 B each, all finite) as training-path inputs. Implementation: `Codec/integration/sqgan_prefeatures.py` (forward-hook capture around the shared `codec.encode`, one encoder call, grid kept UE-local) plus `Codec/integration/prefeature_parity_smoke.py`; a delegating `SQGANDigitalCodec.encode_with_features` method was added to `sqgan_payload.py` during the gate by another actor and preserved.

### 36-byte SPI/scene packet: implemented and tested

`parameters/spi_scene_packet.py` (the `parameters/` owner was materialized from HEAD mid-gate; see below). Layout per design: 32-byte big-endian prefix (version 1 B, producer profile 2 B, flags 1 B, source/session/sequence 4 B each, payload length 4 B, four quantized SPI probabilities 4 B, eight scene values 8 B) + payload + 4-byte CRC32 over prefix+payload → exactly 36 B overhead. Rejects invalid version/profile/length/flags/CRC before admission; profile registry binds SPI producer, checkpoint, preprocessing, projection and quantization (profile 1 → pinned checkpoint `70639640…`); SPI probabilities normalize on receive; the scene-availability flag distinguishes available zero values from missing evidence; IDs are bookkeeping only. 11 tests in `parameters/tests/test_spi_scene_packet.py` cover round trip, byte order, corruption/truncation, flag-byte inconsistencies and registry binding — all pass.

### Pooled eight-value RankNet path: implemented and tested

`Scheduler/training/scene_descriptor.py`: global mean+max pooling of the frozen grid (256-ch → 512-d), learned sigmoid projection to eight bounded scene values (one-value ablation supported), fake-uint8 quantization in training and exact uint8 in evaluation, shared linear RankNet scorer with pairwise logistic loss. The strict loader accepts only manifests declaring `review_kind=independent_scene_pair_review` with named reviewers and grades 0/1/2, excludes and counts uncertain/tie/cross-tier/equal-grade pairs, and rejects native road-label records. Inputs are pooled grid values only — no wait, service time, identity or history. The CLI exits 2 with an explicit blocked message when the reviewed manifest or grid cache is absent; it never invents labels. 12 tests in `Scheduler/tests/test_scene_descriptor.py` pass (fixtures are pytest-temp parsing/mechanics data only, never persisted or reported as review data).

Verification summary: focused rerun 28/28 passed (11 packet + 12 scene descriptor + 5 legacy packet/transport). Full configured suite: 243 passed, 5 failed, 5 skipped, 2 collection errors — every failure/error is pre-existing or environmental in untouched files: `experiments/` and `Notebooks/` paths are excluded by this worktree's sparse checkout, one fixture compares LF vs CRLF bytes, one fixture hash mismatch, and the two collection errors import `Archived/owners_before_simplification_20261027`, which does not exist at `c03a5d5` (`git cat-file -e` confirms). No Test 1 file appears in any failure.

### Reviewed scene pairs: still unavailable — acceptance gate blocked

A three-root search (this checkout, `Documents\Codex`, and the `image_semcom` tree) found **no** completed independent human same-tier scene-pair review: the only pair-preference files are header-only templates (`context_preference_template.csv`, `context_preference_template_v2.csv`; protocol says "No human priority judgments have been collected" and "Zero independent serve-first labels have been collected"); `ladi60_semantic_scheduler_v1.csv` is `review_status=pending` with empty reviewer and native `roads_damage` tags; the only stored swapped-order VLM judgments (`teacher_pair_audit.md`) are silver, `qualified=false`, aspect 0–3 on non-disaster images; historical road/footprint receipts (`strong_reader_receipts/*`, `road_region_receipts/*`, `footprint_*`) are explicitly excluded. No labels were fabricated and no historical road labels were substituted. Consequently the two review-dependent Test 1 acceptance checks — pair agreement over SPI-first and evidence retention after decoding — remain unrun, and the event/flight/duplicate-disjoint Test 2 and Test 3 cohort lock waits on the same review.

### Worktree notes

HEAD unchanged at `c03a5d5`; nothing staged or committed. New files: `parameters/__init__.py`, `parameters/spi_scene_packet.py`, `parameters/tests/test_spi_scene_packet.py`, `Codec/integration/sqgan_prefeatures.py`, `Codec/integration/prefeature_parity_smoke.py`, `Scheduler/training/scene_descriptor.py`, `Scheduler/tests/test_scene_descriptor.py`, this receipt `docs/scheduler_improvement_trials/trial2_prefeature_parity.json`. During the gate an external actor (20:36–20:37) updated the sparse-checkout patterns to include `/parameters/`, materialized the tracked `parameters/` package from HEAD (git-clean; this also replaced the placeholder `parameters/__init__.py` written earlier in this gate with the HEAD package init), added the `encode_with_features` delegation in `Codec/integration/sqgan_payload.py`, and added a thread-id guard in `sqgan_prefeatures.py`; all were preserved and verified in the final test runs.

### Next action

Collect the independent same-tier 0/1/2 human scene-pair review fresh on LADI v2 candidates using the existing empty templates (nothing local can be relabeled into this role), then run `python -m Scheduler.training.scene_descriptor --manifest <reviewed.json> --grids <grids-dir>` and evaluate strict same-tier pair agreement over SPI-first plus evidence retention after decoding to close the Test 1 acceptance gate. Cache grids for the reviewed cohort with `prefeature_parity_smoke --grids-out`. Only after that gate and the cohort lock does Test 2 begin.

### Codex verification of Test 1

OpenCode MiMo V2.6 Flash Free implemented the pre-mask capture, 36-byte packet and eight-value RankNet training path. Codex exposed the codec method, scoped temporary capture to the calling thread, and included the original parameters package in the sparse checkout without replacing its initializer.

Engineering checks. The focused packet, RankNet fixture and legacy packet tests passed 28 checks. OpenCode verified pinned-source hashes for 31 files. Two real LADI images passed CPU parity in docs/scheduler_improvement_trials/trial2_prefeature_parity.json. A separate Codex run on test_row_0009_flooding_any.jpg also passed. Each encode exported a finite 1 x 256 x 16 x 32 float32 grid with one encoder call, an identical 464-byte SQ01 payload and pixel-identical reconstruction. These are codec parity checks.

Full-suite collection was blocked in two legacy modules by the absent Archived.owners_before_simplification_20261027 package. Test 1 semantic acceptance remains blocked until independently reviewed same-tier pairs and grouped cohort assignments are available. RankNet fixture tests do not establish improvement over SPI-first or scene evidence retention. Tests 2 and 3 have not started.

### Execution resumption and Kaggle access

Re-read the active Qwen/Nemotron decision and resumed Test 1 preparation with OpenCode MiMo V2.6 Flash Free. Windows reports Intel Iris Xe and CPU-only Torch. Refreshed Kaggle OAuth interactively and verified authenticated CLI access by listing the active account's notebooks. The executing username is hmmkk20. Listing files in the private version-5 archive notebook still returned HTTP 403. The owner needs to grant that account access for Test 3. Public LADI v2 remains available for the Test 1 cohort. The local 24-image historical provenance receipt identifies candidate events for six images only and cannot establish all required splits.
Public Test 1 source check. The publisher S3 bucket serves the v2a train labels CSV at https://ladi.s3.us-west-2.amazonaws.com/Labels/v2/ladi_v2a_labels_train_resized.csv and the image metadata CSV at https://ladi.s3.us-west-2.amazonaws.com/Labels/ladi_images_metadata.csv. The label CSV supplies source URLs and native tags. The metadata supplies timestamps, GPS and UUIDs. Neither CSV exposes verified event and flight columns. The grouped cohort is not locked.

### Public LADI reviewer hardware pilot

Submitted version 1 of the private hmmkk20/semantic-scheduler-trial2-vlm-smoke Kaggle job. The job uses public LADI v2 source images, confines both sequential model smoke tests to the first T4, pins both model revisions, checks NF4 layers, and records peak GPU memory plus two answers in opposite image orders. This is hardware preflight only. The two images are not a qualified same-SPI cohort and outputs cannot be used as training labels. Local launch source is under the current TEMP directory in trial2-vlm-smoke-20261108.

Hardware pilot v1 completed. Qwen answered both orders on one Tesla T4 with 358 NF4 linear layers and 7,119,831,040 bytes peak allocated memory. Its evidence rationale was inconsistent after swapping. Nemotron failed during pinned config initialization because no attention implementation was set. Submitted private job version 2 with explicit eager attention for Nemotron only. The raw version 1 receipt is retained locally.

Trainer preparation. The new VLM review loader validates pinned reviewer IDs, prompt version, split-disjoint event, flight and duplicate groups, same SPI tier and mapped forward/reverse choices. It checks the calibration thresholds and returns only stable strict training consensus. Existing manifest support is preserved. Twenty-four focused tests passed. These are software tests, not reviewer qualification.

Public cohort preparation. Downloaded and hashed the publisher v2a resized train, validation and test CSVs. Named event prefixes overlap publisher train and validation, so candidate assignments group those by event. The CPU candidate staging job screens exact and thumbnail duplicates and keeps source URL, digest and available EXIF. Official sortie fields and SPI pair membership still require validation before locking the cohort.

### Follow-up on Nemotron and grouped LADI candidates

Confirmed the pinned Nemotron configuration source assigns attn_implementation and then checks whether it contains flash_attention. Version 1 supplied None through Transformers and failed before checkpoint loading. The explicit eager-attention retry is Kaggle version 2. A queued retry is not evidence that the repair passed.

Added Evaluation/trial2_ladi_candidates.py. The public FEMA image URLs for the selected named events use an event/Source/mission/sortie/file layout. The planner uses the publisher path as provenance, excludes unnamed events, rejects missions crossing Trial 2 splits, and only uses native impact tags for candidate sampling. It emitted 101 candidates across 12 events and 59 mission/sortie path groups. Two runs over the same CSVs produced identical JSON with SHA-256 e49d353201d08f5f515cc36e6e2b5bbdc8b9fbe3f21e2ef4d33762c5f35543dc. The path grouping is inferred from publisher layout and is not a separate authoritative flight manifest. Candidate downloads and duplicate screening are still in progress. The archived SPI classifier is staged to score only development-side candidates once the download manifest is ready. No group lock, VLM qualification or RankNet training result yet.

### Test 1 preparation — VLM metrics and group-disjoint reporting (8 November 2026)

Scope honored by this session: Test 1 preparation only. No cloud login or submission, no commit or stage, no Test 2/Test 3 work, no model inference or training, and no label, grade, group or split invented. A parallel actor worked in the same checkout during the session: `Scheduler/training/scene_vlm_review.py` and its dispatch inside `load_reviewed_pairs`, the `train()` provenance fields, `Scheduler/tests/test_scene_vlm_review.py`, `Evaluation/trial2_ladi_candidates.py`, the local v1 receipt and the sections above are theirs and were preserved. An initial draft VLM manifest format written by this session before the parallel work appeared was removed so exactly one explicit VLM format (`review_kind=vlm_scene_pair_review`, pinned `model_revisions`, `prompt_version`, split/event/flight/near-duplicate IDs, forward and reverse votes) exists; the human `independent_scene_pair_review` format and its loader path are untouched.

Added to `Scheduler/training/scene_descriptor.py`, all non-raising reporters over the explicit VLM format:

- `check_group_disjointness(images, pairs=None)` — cohort-freeze report: malformed or unknown-split image records, event/flight/near-duplicate groups crossing splits, pairs missing an image record, pairs spanning splits, and pair SPI-tier missing/mismatch. Returns per-split and per-dimension counts plus `violations` and `disjoint`. Accepts images merged from several manifests, so the calibration/train/development/test2/test3 freeze can be inspected before fitting. An unresolved group dimension must use an explicit shared sentinel; nothing is filled in.
- `swapped_order_metrics(manifest)` — per pinned reviewer over all calibration pairs: `order_consistency` (mapped verdicts agree across both image orders, denominator = pairs with two valid responses), `stable_strict_coverage` (stable strict choices over all calibration pairs; abstentions, malformed answers, ties and disagreements stay in the denominator), plus joint strict `overlap_pairs` and `overlap_fraction`, `agreement`, disagreement examples, `pairs_by_split` and `pairs_unresolved`. Malformed or missing votes count as invalid and are reported, never fabricated; only structural manifest errors raise.
- `qualification_report(metrics)` — applies the active decision's 90% / 50% / 80% gates per reviewer and jointly, reports the overlap size, and returns one `qualified` verdict. An empty denominator (None) never passes a gate.

Exact measured results (software tests only; these are not reviewer qualification):

- After integration, before the three new tests: `python -m pytest Scheduler/tests/test_scene_descriptor.py Scheduler/tests/test_scene_vlm_review.py -q` → 13 passed in 13.02s (12 original scene-descriptor tests + 1 parallel-actor VLM loader test).
- With the three new tests: same command → 16 passed in 10.53s.
- Final focused rerun: `python -m pytest Scheduler/tests/test_scene_descriptor.py Scheduler/tests/test_scene_vlm_review.py parameters/tests/test_spi_scene_packet.py -q` → 27 passed in 11.13s (11 packet + 12 original scene-descriptor + 1 parallel-actor VLM loader + 3 new). This equals the parallel actor's earlier 24 focused checks plus the 3 added here.
- New tests: `test_swapped_order_metrics_report_pass_case_without_raising`, `test_swapped_order_metrics_record_shortfall_instead_of_raising`, `test_group_disjointness_reports_crossings_without_raising`. Fixtures are pytest-temp mechanics data in the explicit VLM format; they are never persisted as review data and never reported as model judgments.

Not run and still blocked: reviewer smoke/qualification on real same-SPI pairs (the queued Nemotron eager-attention retry is not a passing result), SQ-GAN feature extraction for the cohort, RankNet training on real consensus pairs, the SPI-first development comparison and evidence retention after decoding. The checklist items above stay unchecked. HEAD unchanged at `c03a5d5`; nothing staged or committed.

### 9 November status verification

The provisional public staging manifest contains 99 images after exact and thumbnail duplicate screening. Split counts are 15 calibration, 19 training, 10 development, 25 Test 2 and 30 Test 3. These are staged candidates, not locked same-SPI pairs.

Kaggle smoke version 2 completed at the job level. Explicit eager attention resolved Nemotron's earlier configuration failure. It loaded 355 NF4 linear layers on one Tesla T4, then failed during its first visual inference in the C-RADIO vision encoder layer normalization with RuntimeError expected scalar type Float but found Half. Nemotron has no successful two-image inference result yet. Qwen's prior hardware smoke remains the only successful one. No reviewer qualification, real-label RankNet fit or reconstruction evidence comparison yet.

### 9 November archive lookup and Test 1 resumption

The user-provided Kaggle dataset satyapaladuguplus/mini-spi-training-labels is accessible. Its full file listing contains labels, manifests and scene caches, but no TAR or ZIP. The exact Mini_dataset_6000.tar download returned HTTP 404. The dataset's input_manifest.json says the upstream 6,000-image scene cache did not persist image bytes. The original private notebook's files endpoint still returned HTTP 403. This dataset is not a verified substitute for the pinned image archive.

Resumed public LADI SPI scoring from the 99 staged candidates. Submitted Kaggle smoke version 3 on the existing private notebook. The retry keeps eager attention, excludes the vision encoder and projector from NF4 and runs both in FP32, with input casting at their entry points. The language model still uses NF4. Version 3 is a hardware smoke, not reviewer qualification.

Public LADI SPI scoring completed on 44 provisional images. The scored set contains 15 calibration, 19 training and 10 development images. Generated 30 calibration, 82 training and 8 development same-SPI candidate pairs, excluding pairs from the same mission/sortie path. These are unreviewed candidate pairs. Started pinned SQ-GAN grid extraction for the 29 training and development images and reconstruction for development only. No held-out Test 2 or Test 3 images entered these computations.

### Test 1 reviewer execution continuation

SQ-GAN feature generation completed for 29 training/development images and all 10 development reconstructions. Nemotron v3 still failed: the pinned C-RADIO input conditioner converts its output to BF16 before the FP32 patch embedder. Confirmed the explicit dtype cast in the pinned upstream source. Submitted v4 with the conditioner dtype set to FP32 as well. Prepared a separate calibration run for 30 provisional pairs and two identical-image controls per reviewer, with exact publisher image-hash checks, both orders, a structured evidence prompt and raw outputs. Its Nemotron path also aligns projector output with the language embedding dtype. The calibration job has not been submitted while v4 is active.

Version 4 completed at the notebook level. It passed the C-RADIO patch embedding stage and reached token assembly, where FP32 vision projector output was assigned into FP16 language embeddings. The raw version 4 receipt is retained locally. Version 5 adds a projector output cast to the language embedding dtype and runs both reviewers on the 30 calibration pairs in both orientations, plus two identical-image controls each. Kaggle accepted the submission.

All 99 staged image SHA-256 values were reverified against disk. The cohort audit reported 12 events, 58 publisher path-derived mission/sortie groups, and 99 distinct images after the prior duplicate screen with zero cross-split violations. The local input receipt trial2-public-ladi/trial2-reviewed-cohort-input.json under TEMP has SHA-256 eb5197a454b448e3ac1a09fc7e09f0a43628dbe0841e0dcb2da76a6674f0cad6. The Test 2 and Test 3 cohorts have no SPI scores or reviewer votes in this receipt.

Version 5 was observed RUNNING. Local session 59770 is monitoring completion, downloads the calibration JSON, computes qualification metrics, and submits the prepared 82 training plus 8 development pair run only if both models completed every orientation, all qualification thresholds pass, the cohort check passes, and identical-image controls return tie or uncertain. No RankNet fit will be reported from the calibration answers.

### Test 1 decision — Kaggle calibration v5 completed

Downloaded the completed reviewer_calibration.json from hmmkk20/semantic-scheduler-trial2-vlm-smoke. SHA-256: 9d87a799d098f6eb04fc8b45b9013ebfbe93c7e9942156d1d3e84185e81f5131. Ran trial2-vlm-calibration/evaluate_calibration.py on the receipt and provisional-development-spi.json. Raw outputs and qualification.json are in TEMP/trial2-vlm-calibration-output-v5. The calibration cohort disjointness check passed.

Both models generated all 62 responses on one T4. Nemotron's projector cast resolved the FP32-to-FP16 assignment error. The run reported 225 NF4 layers and 12,061,415,936 peak allocated GPU bytes for Nemotron, and 359 NF4 layers and 7,376,712,192 bytes for Qwen.

Qwen emitted repeated exclamation marks in all 62 responses, with zero valid votes. This is a generation failure and the root cause is unverified. Nemotron produced 60 valid pair responses. It was order-consistent on 7/30 pairs (23.3%) and gave a stable strict preference on 5/30 (16.7%). It chose A and B on its two identical-image controls. Cross-model agreement is undefined with zero overlapping stable strict judgments.

Decision: Test 1 reviewer qualification failed. The required thresholds remain 90% order consistency, 50% stable strict coverage, and 80% cross-model agreement, with tie or uncertain on the identical-image controls. The training/development review run was not submitted by this turn. RankNet was not fitted and the downstream gates remain pending.

### Mini cache photo provenance check

Downloaded the mini Kaggle manifests and verified that all 6,000 scene-cache entries join to pool_manifest.json by image ID and SHA-256. The pool lists 4,000 CrisisMMD, 1,000 Open Images, and 1,000 Places365 photos. The first two sources supply per-photo URLs. Places365 has source identifiers and no direct URLs in this manifest. Downloaded one CrisisMMD photo from its pinned URL and confirmed an exact SHA-256 match. All 3,933 mini training IDs join to selected_manifest.json, which provides original relative image paths. Of the mini training image hashes, 3,887 also appear in the scene pool. The mini cache does not list LADI v2. Reuse the matched CrisisMMD originals where appropriate and source LADI v2 separately. Existing cache files support header replay. Trial 2 review, feature extraction, and reconstruction checks require image pixels, which can be fetched in batches. No new VLM labels were generated in this check.

### Dataset and label scope clarification

The complete Mini cache covers 6,000 images: 4,000 CrisisMMD disaster photos, 1,000 Open Images photos for object diversity, and 1,000 Places365 photos for scene diversity. The project record describes cached scene measurements and Mini SPI classifier predictions across the full pool. Treat these predictions as model outputs. The separate CrisisMMD VLM labeling run recorded 3,933 available labels and 67 unavailable. Those labels are distinct from the full-pool classifier predictions and from Trial 2 same-SPI preferences.

LADI v2 and MEDIC were used in smaller pilots outside the completed Mini cache. MEDIC includes CrisisMMD source material. Check source IDs and hashes before combining them, and account for overlap when forming splits or reporting coverage. TREC-IS, xBD/xView2, RescueNet, FloodNet, BDD100K, and COCO were discussed outside the completed Mini cache. Continue to use the cache for header replay and fetch only the original photos needed for Trial 2 pixel-based review, SQ-GAN features, and reconstruction checks.

### Reviewer diagnostics v6

Submitted a bounded one-T4 diagnostic to the existing private Kaggle notebook. The Qwen run compares the original configuration with a variant that keeps the visual tower unquantized in FP32. It runs the original smoke example and three calibration pairs in both orders plus two identical-image controls. It logs first-step logit finiteness, image grids, generated token IDs, and raw answers. These changes test a numerical hypothesis and do not establish a fix by themselves.

Nemotron v5 returned valid choices on all 60 pair prompts. It chose A 28 times, B 14 times, and tie 18 times across those prompts. It agreed across orders on seven pairs, including two stable ties and five stable strict preferences. The pinned NVIDIA chat code inserts one placeholder per image when none is supplied. Its example explicitly labels each image placeholder. V6 tests explicit A/B placeholders and a separate instruction to grade each slot before choosing. The diagnostic uses calibration examples only. The full qualification thresholds and the requirement for genuine identical-image control answers remain in force.

V6 remained RUNNING at the last status check. The CLI returned no live log text or v6 diagnostic receipt at that check. The diagnostic script SHA-256 is ef67d43978b2d479218aec9cc95b6d44716037bdab12382d581de9022747bcac. In v5, nine pairs retained the same letter after swapping and therefore changed the selected photo. Fourteen switched between a strict preference and a tie. These are observed answer patterns. Their underlying causes still need validation.
