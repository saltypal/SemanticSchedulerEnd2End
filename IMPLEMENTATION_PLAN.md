# Trial 2 reviewer repair

## Scope and baseline

Keep the approved SQ-GAN, SPI, RankNet, and two-reviewer architecture. V9 on one T4 used the pinned Qwen checkpoint, NF4 with FP16, SDPA, and 256 visual tokens per image. All eight answers parsed. Both duplicate controls tied. Two pairs had stable strict preferences. The third chose B both ways. Compare two targeted changes on the same three pairs and two controls before any full calibration.

## Files and responsibilities

- TEMP/trial2-vlm-qwen-rubric-v9/pilot.py is the read-only tested baseline.
- TEMP/trial2-vlm-qwen-order-v10/pilot.py and kernel-metadata.json are the only implementation outputs. The script retains the same checkpoint, environment, image hash checks, T4 limit, model loading, and decoding. Metadata reuses the approved private notebook identity.
- SEMANTIC_SCHEDULER_TRIAL2.md records receipts and decisions. The supervising thread owns journal edits.

## Execution and interfaces

Use the existing pilot entry point and its forward, reverse, and identical cases. Load Qwen once. Evaluate an evidence-first prompt at the 256-token image budget and the same prompt at a 512-token budget. Require each photo to be described separately and both grades to appear before the choice. Instruct abstention when damage is uncertain, and preserve the existing response contract with integer grade_a/grade_b in [0,2], a choice in A/B/tie/uncertain, and a nonempty evidence string. Preserve evidence_string alias validation. Keep raw outputs and original unforced choices. Allow enough output tokens for both descriptions. Store each variant independently in reviewer_order.json, with orientation, choice, raw answer, grades, first-token finiteness and image grids. Write receipts incrementally.

## Validation and gates

Compile the script, confirm metadata targets hmmkk20/semantic-scheduler-trial2-vlm-smoke, and summarize the diff. Test only one T4, at most 16 prompts, no training or other changes. The supervisor submits the job and reviews measured output. A smoke candidate needs eight valid, finite answers, the same mapped verdict on all three swaps including at least one strict winner, and two tie/uncertain controls. Preserve the existing full-cohort 90% consistency, 50% strict coverage, and 80% agreement thresholds and keep both test cohorts untouched. If both variants fail, report which inputs failed and preserve the raw answers. Never force a choice to match its swapped answer.

## Delegation

Luna xhigh implements the bounded script from the v9 temporary snapshot associated with repository HEAD b08f0f5. Modify only the two v10 temporary files. Do not submit Kaggle jobs or modify the repository. Return filenames, diff summary, and compile result.
