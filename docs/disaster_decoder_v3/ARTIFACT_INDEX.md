# Disaster Decoder V3 artifact index

This index records the completed extraction, full training/comparison and verified final executed notebook. Paths are repository-relative unless noted. Historical versions remain preserved; the corrected final notebook is version 2.

| Artifact | What it supports | Limits / handling |
|---|---|---|
| `results/disaster_decoder_v3/20261022_stageA_v3/artifact_audit.json` | Passed integrity audit; IDs/counts, finite feature checks, role disjointness, model statuses, Qwen schema, cost-profile count, hash checks. | Audit is about artifact consistency, not model quality. |
| `results/disaster_decoder_v3/20261022_stageA_v3/extraction/manifest.json` | Source role/split records and conflict/exclusion tracking. | 1,939 rows are not 1,939 unique retained images. |
| `results/disaster_decoder_v3/20261022_stageA_v3/extraction/features.jsonl` | Frozen model features and statuses for 1,934 retained unique images. | Not receiver input; raw image-derived features are used for study preparation only. |
| `results/disaster_decoder_v3/20261022_stageA_v3/extraction/labels.jsonl` | Matched source labels for retained IDs. | Keep labels out of deployed receiver features. |
| `results/disaster_decoder_v3/20261022_stageA_v3/extraction/label_conflicts.json` | Four source rows for two exact-byte IDs with conflicting labels. | Excluded as a group; preserve for audit, do not majority-vote. |
| `results/disaster_decoder_v3/20261022_stageA_v3/extraction/inference_cost_profiles.json` | Eight actual registered-operation cost profiles, four on each GPU. | Registered operation estimates, not complete hardware FLOPs. |
| `results/disaster_decoder_v3/20261022_stageA_v3/extraction/qwen_proposals_ollama_4b.jsonl` and `qwen_status.json` | Local Qwen proposal evidence; 32 schema-valid annotations in audited artifact. | Silver only; evaluate separately from source labels. |
| `results/disaster_decoder_v3/20261022_stageA_v3/source_manifest.json` | Uploaded source-file SHA-256 snapshot and ZIP digest. | Critical because the recorded Git commit had a dirty tree. |
| `results/disaster_decoder_v3/20261022_stageA_v3/environment.txt` and `hardware.json` | Extraction runtime/hardware context. | Extraction environment is not Kaggle training environment. |
| `results/disaster_decoder_v3/20261022_stageA_v3/run_status.json` | Extraction stage completed status and elapsed time. | Does not indicate training completion. |
| `results/disaster_decoder_v3/20261022_stageA_v3/canonical_source/` | Canonical source snapshot for uploaded extraction. | Consult the exact SHA-256 values in the manifest. |
| `results/disaster_decoder_v3/20261022_stageA_v3/extraction/` | Stage A outputs and nested worker/annotation artifacts. | Contains sensitive/high-volume data artifacts; do not redistribute without confirming dataset and project policy. |
| `results/disaster_decoder_v3/20261023_stageB_v1/monitor_snapshot/study/comparison.csv` | Historical first downloaded 39-row comparison. | Workspace copy; the full package below is the primary handoff copy. |
| `results/disaster_decoder_v3/20261023_stageB_v1/full/study/comparison.csv` | Downloaded full study comparison artifact. | Treat paired pool as statistical unit; do not use held-out scores for selection. |
| `results/disaster_decoder_v3/20261023_stageB_v1/full/study/{task}/selection.json` | C-role selected method and candidate details for each task. | Model-selection evidence only; selections are task-specific. |
| `results/disaster_decoder_v3/20261023_stageB_v1/full/study/{task}/evaluation/paired_vs_selected_baseline.json` | Paired source-pool bootstrap differences/intervals between CAHR and the C-selected method. | Pools are not independent disasters; intervals do not establish universal superiority. |
| `results/disaster_decoder_v3/20261023_stageB_v1/full/study/{task}/evaluation/` | Per-task episode metrics, replay decisions, streaming outputs and latency summaries. | Streaming ticks are abstract; not radio/codec milliseconds. |
| `results/disaster_decoder_v3/20261023_stageB_v1/full/study/building_damage/seeds/` and `ablations/` | Two additional seeds and nine retrained field ablations. | Building-only auxiliary experiments, verified by post-analysis. |
| `results/disaster_decoder_v3/20261023_stageB_v1/full/study/data_audit.json` and `status.json` | Dataset role counts, cache hashes, selection boundary, and study completion status. | Independently checked against the downloaded extraction input. |
| `results/disaster_decoder_v3/20261023_stageB_v1/full/run_status.json` | Training job status and elapsed time. | Job completion does not by itself audit every prediction or output. |
| `results/disaster_decoder_v3/20261023_stageB_v1/full/canonical_source/INFRA/learning/decoder_v3_study.py` | Published source snapshot. | Source hash reportedly matches the published source; verify against manifest before reproduction. |
| `INFRA/decoder_queue.py` and its tests | Receiver queue wrapper checks complete job/header/profile/token/arrival and F-update behavior. | Separate integration wrapper; does not change the training benchmark. |
| `kaggle/disaster-decoder-v3-extract/05_disaster_decoder_v3.ipynb` | Extraction notebook source. | Source file is not proof of a completed remote run; artifact audit is. |
| `kaggle/disaster-decoder-v3-train/05_disaster_decoder_v3.ipynb` | Original training notebook source. | Not itself an executed-output artifact. |
| `Scheduler/notebooks/05_disaster_decoder_v3.ipynb` | Actual executed Kaggle final version-2 notebook, 39-row HTML table and two embedded PNGs. | Reused verified extraction and completed checkpoints; no test-driven retuning. |
| `results/disaster_decoder_v3/20261023_stageC_v2/` | Final run status, source manifest, resumed provenance and captured outputs. | Version 2 corrects report-curve aggregation only; quality metrics are unchanged. |
| `results/disaster_decoder_v3/20261023_analysis_v2/` | 37,440 reproduced rankings, all-baseline paired comparisons, oracle diagnostics, fixed-model utility sensitivity, seed/ablation summaries and corrected figures. | Privileged oracles are not deployable policies. Sensitivity does not retrain/select models. |
| `results/disaster_decoder_v3/20261023_analysis_v2/figure1_coverage.png` and `figure2_paired_comparison.png` | Two main report figures. | Repetitions -> strata -> pools; fixed comparator list, matching primary A10. |
| `results/disaster_decoder_v3/20261023_stageB_v1/full/artifact_manifest.json` | All 480 selected primary files, per-file hashes and transfer-recovery record. | Seven generated pycache paths are deliberately excluded. |
| `SemParam/decoder_parameters.py`, `decoder_extractors.py`, `decoder_costs.py` | Header/math, extraction adapters, and cost registry implementations. | Match exact run snapshot before reproducing. |
| `Scheduler/decoder_scheduler.py`, `INFRA/learning/decoder_v3_study.py` | Policies and study/training orchestration. | Smoke validation is not full training evidence. |

## Key counts and provenance

- Pinned source: LADI revision `5f2dbfe8c466d32edafd1bab847ec5252309acdb`; downloaded source splits contain validation=890 and test=1,049 rows.
- Manifest: 1,939 rows and 1,936 unique IDs before the conflict group is excluded. Four rows for two exact IDs have conflicting labels.
- Retained audited IDs: 1,934 total; A=445, B=178, C=267, test=1,044. Extraction-summary test count 1,045 is the pre-deduplication count; use audit unique counts for final reporting.
- Model outcomes: all three frozen image models succeeded on all 1,934 retained IDs; embeddings are 512-dimensional.
- Cost profiles: 8, with 4 on `cuda:0` and 4 on `cuda:1`. Registered-operation inference estimates: CLIP 8.81762304 GFLOPs, LADI 32.6971392 GFLOPs, DINO 665.607–746.230 GFLOPs depending on shape. No comparable full-model Qwen FLOPs were measured.
- Annotation: 32 valid Qwen proposals, silver tier only.
- Audit provenance: Git commit `02580b0921265ac0f0c30177d6ea60e7f8e6e368`, dirty state recorded; source files individually SHA-256 pinned; uploaded source ZIP digest `5e78608516c5949d033a30248ed02d3fb17df4134dbad52fe52513ecca93cae3`.
- Training: private Kaggle kernel `hmmmkk/disaster-eight-field-scheduler`, version 1, completed in 470.9174869 seconds. All 480 selected files passed package audit; all 12 gzip traces parsed to EOF with no invalid rows. All 37,440 exported-model episode rankings were reproduced.
- Results: see [RESEARCH_REPORT.md](RESEARCH_REPORT.md#measured-comparison) for the full 13×3 A10 table and paired CAHR-versus-C-selected intervals. Selections were made on C, not the held-out scores; all methods are shown to avoid cherry-picking.
- Validation: 53 combined local tests passed in 48.45 seconds. The executed notebook passed nbformat validation; its three code cells have real execution counts, two embedded figures match downloaded PNGs byte-for-byte, its HTML table contains all 39 data rows, and no error outputs are present.
- Final notebook: `hmmmkk/disaster-eight-field-final-notebook`, version 2, completed in 297.4164031 seconds. It matched the original 39 quality-metric rows within 1e-12. Notebook SHA-256: `2cb2fae87b14d37c52b37f065d2af872bfb34907d9b7b5d7ba49319ae36a04e0`.

The bounded prototype's validation gates are complete. This does not establish real codec performance, independent emergency/fact ground truth, event-level generalization, comprehensive VLM FLOPs or journal readiness. Original runs and the report-only correction remain distinguishable.
