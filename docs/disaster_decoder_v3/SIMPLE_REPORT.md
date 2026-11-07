# Disaster Decoder V3 — simple report

**Final prototype report: 23 October 2026.** Kaggle training, artifact/prediction auditing and the final executed notebook are complete. All 37,440 saved model/episode rankings were reproduced, all 39 comparison rows matched on resume, and 53 local tests passed. The notebook contains the full comparison table and two embedded graphs.

## What is being tested?

Imagine a receiver has already received several complete compressed image representations. Each one has a small header with evidence about the task and image. The experiment asks whether a computer can use those header values to choose a useful order for looking at the already-arrived representations.

This is not a test of sending image fragments, changing the compression method, or controlling a real wireless network. It does not measure reconstruction quality or prove emergency-response benefit.

The images and extracted model features were real. UE assignments, arrival queues and repeated-observation scenarios were constructed. Actual latent payload bytes were not decoded in this benchmark.

## What is in the header?

The 32-byte header carries eight optional clues, called `R, S, Q, I, F, P, C, N`: task match, local semantic rarity, confidence in a critical claim, task importance, trusted freshness, detected target area, scene complexity, and local embedding novelty. The receiver also gets a flag for each value saying whether it is available. An unavailable value is not the same as a measured zero.

These are estimates from models, history, or timestamps—not guaranteed facts. In particular, model confidence is not correctness, a scene-complexity proxy is not severity, and the local novelty value is not proof that an image shows a new incident.

## What has actually finished?

The study used a pinned LADI dataset revision. Two exact image IDs appeared with conflicting source labels, across four source rows. Those records were kept for audit but excluded; they were not relabeled. Repeated images with consistent labels were extracted once. The audited cache has 1,934 unique images: 445 in A, 178 in B, 267 in C, and 1,044 in test. A separate extraction summary reports 1,045 test-role rows before duplicate removal; the audit's 1,044 is the unique count.

CLIP, LADI-small, and DINO extraction succeeded on all 1,934 retained images. The audit also found eight hardware cost profiles across two Tesla T4 GPUs and 32 valid Qwen-generated proposals. Qwen proposals are silver annotations and must be considered separately from source labels. LADI-small is a reference model, not ground truth.

All 13 methods were compared on the shared evaluation pools. The A10 results show mixed outcomes:

| Task | CAHR | Simple I baseline | Method selected on C | C-selected method A10 |
|---|---:|---:|---|---:|
| Building damage | 0.7634 | 0.7580 | Coverage-gradient trees | 0.7778 |
| Road damage | 0.5301 | 0.5953 | Regressor | 0.4725 |
| Structural flooding | 0.6317 | 0.6331 | Coverage-gradient trees | 0.6050 |

The task-specific selection column comes from C; held-out scores did not choose the method. The simple `I` baseline exceeds CAHR on road and flooding. Paired 95% source-pool bootstrap intervals for CAHR minus the C-selected method include zero for building and flooding, and the road comparison is positive at the reported point estimate with a lower bound of zero. This is mixed evidence, not a general win for CAHR. See the [full 13-method table and confidence intervals](RESEARCH_REPORT.md#measured-comparison).

## What was delivered, and what remains outside this prototype?

The run used 128 shared A-role queries per task and the planned training limits. C selected methods; held-out results did not choose a replacement winner. All three tasks have verified streaming outputs. Extra seeds and nine retrained ablations were building-only. The completed notebook matched the original quality results exactly; its graphs use the same source-pool weighting as the table. No model was retuned on test results.

The working modules, trained exports, checkpoints, traces, comparison files, two reports and executed notebook are indexed in the handoff. The teammate's digital codec is still needed to measure actual reconstructed-image evidence and decoder times. The original 24-hour target was not met; the reports do not hide that or call this journal-ready.

Even a successful benchmark would only support claims about ordering complete, already-arrived representations under this experimental setup. It would not show real network deadlines, emergency priority, life-safety value, or better decoded images. See [the research report](RESEARCH_REPORT.md), [protocol](PROTOCOL.json), [journal](IMPLEMENTATION_JOURNAL.md), and [artifact index](ARTIFACT_INDEX.md) for detailed evidence and reproducibility limits.
