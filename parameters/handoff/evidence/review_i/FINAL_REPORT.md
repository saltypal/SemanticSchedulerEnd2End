# Operational report — review-only run 20261026i

## Run and result

- Kaggle version 1 completed: https://www.kaggle.com/code/satyapaladugu/betterscheduler-qwen9b-review-20261026i
- The experiment completed 30/30 reviews: 24 constructed controls and 6 actual parameter JSONs. All 30 rows were complete, valid JSON, finite-preflight-approved, thinking-off verified, and below the generation token limit.
- Controls: 16/24 jointly correct; consistency balanced accuracy 0.7083; representation-sufficiency balanced accuracy 0.7500.
- Actual JSONs: 4/6 jointly correct. Two were falsely rejected because the model interpreted explicit negative claims as missing evidence. These results are development evidence, not a final generalization claim. See [`SEMANTIC_VALIDATION_I_RESULTS.md`](../../../../docs/SEMANTIC_VALIDATION_I_RESULTS.md) for the independent interpretation.
- The run used Qwen3.5-9B NF4 with FP16 compute on GPU1 of two Tesla T4s. Transformers 5.3.0; CUDA 12.8. Processor receipt confirms `enable_thinking=false` and `thinking_disabled_verified=true`. Peak measured GPU1 allocation was 11.811 GiB.

## Retrieved artifacts and integrity

Retrieval root: `retrieved/`. It contains the experiment run, 12 input images, 30 raw review texts, review JSONL/metrics/status, an executed notebook, and four cell logs. All four cells report complete.

- Run artifact checksum manifest: **69/69 verified**, with no missing or mismatched files.
- The remote source manifest matches the manifest embedded in the submitted source archive. **69/69 source archive entries** verify against it; archive SHA-256 is `6cd5e8abc3d99113eff24b53a7b601bf0bc08db6687eae2e5caf08828b01a566`.
- The filtered retrieval did not separately download the source `.py` tree; source verification uses the submitted archive and matching remote manifest. No second download was needed.

Machine-readable final integrity record: `terminal_audit.json`. The initial `terminal_checksums_verified.json` confirms all 69 run-output checksums but marks source-tree files missing because the selective download did not include `.py` files. The follow-up `source_manifest_archive_verified.json` confirms the downloaded source manifest matches the submitted source archive manifest and all 69 archived source hashes match; no second download was needed.

## Retrieval issue

The status monitor observed `COMPLETE` at 00:14:01 UTC, but its first SDK retrieval encountered a Windows `charmap` encoding error while rendering output. The existing download already contained the output checksum manifest and all experiment files. I stopped only the operational monitor process, retained those files, and audited them locally without another download. The error was caused by the monitor's terminal retrieval retry handling; it did not alter the completed Kaggle run or its artifacts.

Push, package, test, status and monitor receipts remain in this directory. The canonical run map is `BetterScheduler/KAGGLE_RUNS.md`.


