# BetterScheduler: measured laptop checkpoint

The bounded CPU pilot is complete. Parameter finalization and the cloud campaign
are not complete. This report points to measured evidence rather than replacing
its immutable run artifacts.

Source run: `results/local/20261025T082051Z`.
Corrected wire replay: `results/replay/20261025T084941Z`.
Independent validation: `results/ops/20261025T084931Z/validation/VALIDATION_REPORT.md`.

## Workload and limits

- 180 parents:100 LADI,80 MEDIC;156 exact/near-duplicate components.
- 9,272,304 new downloaded bytes; no reported image download failures.
- Approximately719 seconds CPU elapsed, below the3600-second cap.
- Actual CLIP extraction68.897s; SigLIP126.814s. These are whole-cohort extraction
  times on this laptop, not decoder service times or deployment latency claims.
- Queues:32 complete-message observations,4 logical senders,K=10.
- Native annotated-observation and exact-copy arms; relevant-only arms lacked
  sufficient independently positive held-out parents.

## Provisional development-selected results

| Producer/task | Full-precision A10 | Selected groups | 8-bit header | 16-bit header |
|---|---:|---|---:|---:|
| CLIP building damage |0.4122|Novelty|56 B|59 B|
| CLIP road damage |0.4189|Relevance,novelty,scene|67 B|80 B|
| SigLIP building damage |0.4200|Relevance,novelty|57 B|61 B|
| SigLIP road damage |0.3231|Relevance,novelty,scene|67 B|80 B|

These configurations were chosen on the same development split used for their
scores. They are optimistic selection results, not an estimate from a locked
final test. Do not select a universal backbone from this table.

The selected CLIP building novelty method uses predicted condition changes as
well as embedding distance. Its result does not establish that appearance-only
novelty is sufficient. Affected-people results are deliberately not emphasized:
only four of the80 MEDIC parents carry positive labels.

## What the corrected deployment checks establish

Cached replay exactly reproduced the six original selected full-precision A10
values. Actual encode/decode of numerical headers then changed some rankings at
8-bit precision. For example, CLIP building A10 changed0.412228→0.408181;
SigLIP road changed approximately0.323→0.26. All six16-bit A10 values matched the
full-precision values. Matching A10 does not imply identical latent payloads or
pixel reconstruction.

The real saved-head extraction CLI passed a repeated-image check: sender memory
version0→1, novelty cold-start1→0, embedding distance0 on the repeat, no positive
condition delta, and two correctly decoded56-byte headers. This checks the
extractor/serialization path, not neural-codec execution.

The initial19-test validation ran in a mutable checkout: unrelated cloud/data
and notebook source changed during the checks. Exact pre/post source hashes and
executed commands are retained by the independent operator. A later unchanged
snapshot run must be identified separately rather than rewriting that history.

## Limits and next decision

No verified event IDs/capture times, no natural cross-view object correspondence,
no localized-producer validation and no codec reconstruction were established.
Native image atoms are distinct annotated observations, not unique real assets.
Unknown annotation fields remain unknown. An unavailable parameter is unresolved,
not empirically rejected.

Before cloud selection: build and validate the historical exposure ledger,
require paired task/arm/queue support, preflight official data, and verify both
GPUs process images with numerical parity. The old Mamba session was manually
stopped by the user, but API receipts still reported RUNNING; no new GPU job has
been started. Current cloud source has not been executed remotely.
