# Documentation map

Start with root `README.md` and `PROJECT_MAP.md`.

For the finalized ranking loss, benchmark construction and primary endpoint, use [protocol 04](decoder_scheduler_redesign/04_FINAL_BENCHMARK_EVALUATION_AND_LOSS.md), its [JSON contract](decoder_scheduler_redesign/benchmark_protocol_v1.json) and [mathematical reference checks](decoder_scheduler_redesign/reference_checks.json). The reference checks are synthetic arithmetic validation, not trained-model results.

For the proposed Qwen/VLM roles in runtime task interpretation, parameter labeling, transmitter-KB comparison and benchmark assistance, see [guide 05](decoder_scheduler_redesign/05_VLM_INTEGRATION_AND_TASK_CONDITIONING.md). VLM labels and automated judge scores remain distinct from independent reviewed truth.

The latest receiver-side scope is documented in the proposed [high-level research design](decoder_scheduler_redesign/01_HIGH_LEVEL_RESEARCH_DESIGN.md), [low-level implementation/evaluation specification](decoder_scheduler_redesign/02_LOW_LEVEL_IMPLEMENTATION_SPEC.md) and [finalized parameter extraction guide](decoder_scheduler_redesign/03_FINAL_PARAMETER_EXTRACTION_GUIDE.md), dated 22 October 2026. These define the latest user-selected eight parameters, including transmitter-KB novelty. Complete header-plus-payload messages have already arrived; the scheduler chooses their decoding order. These design documents supersede earlier parameter proposals for this new path; they do not describe an implemented runtime migration.

| Topic | Current guide |
|---|---|
| Whole-latent scheduler and limits | `STATIC_LATENT_SCHEDULER.md` |
| Current structure and compatibility | `CODEBASE_CLEANUP.md` |
| Original source migration | `MIGRATION_MAP.md` |
| Notebook roles | `NOTEBOOK_MIGRATION.md` |
| Modular CLI and environments | `REBUILD_GUIDE.md` |
| Earlier UE experiment | `SEMANTIC_SCHEDULER_RUNBOOK.md` |
| Measurement status | `DEMO_CLAIMS.md`, `VALIDATION_REPORT.md` |
| Teammate notebook merge | `MERGE_8073B75.md` |
| Codec consolidation | `MAIN_MERGE_1E44BAA.md` |

Phase 1 specifications, literature surveys, presentations, rendered reports
and earlier plans remain research records at their existing paths. Their
dates/scopes matter: proposed work is not an implemented feature, and an old
measurement is not a newly reproduced result.
