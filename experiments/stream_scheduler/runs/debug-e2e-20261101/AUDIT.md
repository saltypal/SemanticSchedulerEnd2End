# End-to-end scheduler repair audit

## Result

The active local test setup has been executed, confirmed faults corrected, and all14 non-teacher real-cache methods passed a bounded pilot. GPU repair validation is tracked separately; do not infer VLM inference success from CPU grammar tests.

## Confirmed faults and corrections

| Fault | Correction | Acceptance evidence |
|---|---|---|
| All64 original prefix replies were plain text, not valid teacher JSON | Constrain task-specific output tokens; strictly validate parsed objects; preserve invalid historical replies | Teacher contract/parser tests, actual pinned tokenizer reachability/EOS checks; bounded GPU check pending |
| Nullable aspect arrays failed the proposed grammar library | Use fixed named rating objects with string enums, normalize to four integer ratings/unknown | Character grammar and pinned tokenizer tests |
| Insufficient teacher labels aborted the student driver before writing per-model status | Persist every requested model outcome including exact support counts and blockers | Missing-label fixture and real v4 cache replay; no lowered16/8 gate |
| Strict preference evaluation credited exact score ties as correct for B | Require the correct strict sign; a tie is wrong for either strict target | Zero-logit counterexample test |
| Churn reassigned source identities after constructing repeats | Preserve sources and make required initial sources available at time zero | Local/cross repeats jointly tested with churn/bursts; independent review reproduced old fault and checked correction |
| Resuming a producer could invalidate PPO's original warmstart binding | Freeze/hash the initial actor before producer retraining; reuse it | Changed-producer and tamper tests, independent focused recheck |
| Simulation accepted NaN/infinite horizon/clock/budget inputs | Validate finite times, valid tiers, delay budgets, repeat modes and failure probabilities | Boundary/failure tests and empty-episode definedness |
| Invalid portable model shapes/preprocessing could survive loading | Validate architecture keys/shapes, positive normalization scales, class weights and numerical trees before service | Malformed-artifact tests plus actual trained export parity |
| TD-PER shared the VLM uniform-mixture rule | Restore pure normalized TD priorities for TD-PER; retain mixture/importance correction only for VLM guidance | All Q variants optimize, checkpoint/resume and export; formula assessed independently |
| Short pilots used a fixed10k exploration horizon | Scale to half the planned action budget and preserve that schedule on resume | Checkpoint schedule test |
| Supervised/PPO minibatches issued independent neural calls for each state | Batch independent sets while preserving per-state pooling and per-episode losses | Output/gradient/value equivalence tests and real-cache training |
| Taking samples in episode order could exclude later populations | Interleave causal trajectories across offered training episodes | Capped sample test includes every cohort; real pilot uses4/12/50/60/100 UE training configurations |
| RL artifacts lacked the same portable parity gate as supervised artifacts | Verify best-checkpoint NumPy/Torch outputs before saving portable RL models | DQN/PPO fixture and real-cache parity artifacts |
| Child failure could coexist with a superficially completed supervisor | Persist run_outcome and child exits, convert stale running/planned states to interrupted | Supervisor failure/partial tests; per-arm ledgers remain authoritative |
| Earlier profiling claimed one thread without bounding native BLAS | Set/report actual native thread pools as well as Torch threads |14 models,56 shape/timing rows; native pools verified1 |
| Directory relocation broke three historical metric imports | Add narrow compatibility routes to canonical metric modules | Entire original220-test suite passes after test dependencies restored |
| GPU notebook exceeded Kaggle's1MB source limit | Remove embedded large vectors; regenerate only the missing pinned embeddings during bounded validation | Exact HTTP400 diagnosis preserved; corrected notebook525682bytes; remote result tracked below |

## Tests actually executed

- Final active four-root release run: **297 passed,3 skipped**, `validation-release/junit.xml` and `pytest.log`. Earlier intermediate runs remain preserved.
- Additional real serialized-header/controller and tree-boundary checks: **10 passed**, `tests-baseline/final-added-tests.xml`. These include enqueue/CRC/completeness/duplicate identity, nonpreemptive service, concurrent arrivals, success/failure history, foreign/duplicate completion, no retry, empty/future queues, reset and RR with unavailable semantic fields.
- Static-tokenizer reuse and complete-cache no-reload cases are included in the297-test release total; teacher-focused20/20 results are also retained. Constructor profiling measured about3.4seconds/request before reuse and under1ms/callback after a one-time vocabulary build. This is CPU construction overhead, not a claimed end-to-end VLM speedup.
- Independent waiting-delay reference: **11/11 passed**, `validation-final/waiting_reference/`.
- Corrected historical teacher-cache replay: **1 passed**, after retrieving and checksum-verifying actual v4 contexts/requests. It reproduces insufficient qualified preference support rather than accepting a missing-file error as a valid blocker.
- All seven teacher-student variants train from explicitly synthetic qualified-format preference fixtures. This tests ingestion/learning/export; it does not qualify a real teacher.
- Original skips: one requires two CUDA devices, two require external SwinJSCC model artifacts. Codec inference remains outside this task.

Counts from separate targeted reruns are not added repeatedly to inflate the number of distinct tests. No archived code or tests were edited. The legacy receiver-controller and modern387-feature simulator are separately exercised; these tests do not establish a deployed digital codec path.

## Real-cache vertical pilot

`real-cache-pilot/`: archiveSHA verified;6000 records,5952 distinct image hashes, all5952 embeddings available through the verified v4 extension.3915 image-level VLM silver references and8 conflict components are recorded. Final source test remains unopened.

All14 non-teacher methods trained/exported and completed development replay. Scale: five training traces,32 snapshots, two supervised epochs and504 selected actions per RL arm, one seed. This is deliberately a correctness pilot, not the full20k-action experiment or convergence evidence. The three real teacher-dependent arms remain blocked unless qualified supervision is sufficient; synthetic test passes do not override those gates.

## Inference and compute

`systems_profile.json`/`.csv`:14 portable models at queue sizes4/16/64/256,56 timing rows. Five warmups,20 measured repetitions, actual WSL hardware/runtime, one native BLAS/Torch thread. Timings include context construction and NumPy selection, exclude the decoder. FLOPs are only Torch-profiler-supported operators using2ops/MAC; do not label them total system FLOPs. Tree FLOPs are not invented.

## Reproduction and provenance

Use the recorded existing WSL interpreter. `pip-freeze.txt`, `before_manifest.json`, `after_manifest.json`, `changed_sources.json`, `DECISIONS.md` and all JUnit/log files preserve the execution boundary. The local interpreter is Python3.14.4 while the historical project metadata declares3.12; the source-path experiment runner was tested in the recorded environment, not installed as an allegedly validated3.12 package.

```bash
/home/Satya/.venvs/semantic-scheduler/bin/python experiments/stream_scheduler/validate_system.py
```

All actual GPU versions, budgets and retrieved output paths belong in the existing `parameters/KAGGLE_RUNS.md`. V4 is immutable. New GPU validation has a hard1200-second limit on at most two T4s (at most0.667aggregate GPU-hours); it is not full scientific training.

## Remaining limits

- Grammar correctness does not establish visual truth or emergency urgency. Narrow controls remain narrow.
- Existing insufficient real labels are not fabricated or relabeled to force a successful arm.
- Dynamic mixed-image traces remain declared synthetic chronology; natural cross-view correspondence is not inferred from similarity.
- The adaptive second feedback round, full-budget multiseed training and independent final evaluation are still outside the completed evidence.
- No encoder, decoder, SPI classifier, model weights or archived implementation was redesigned.

## Remote validation

V5 terminal: all17 optimization mechanisms passed on the assigned two T4s (16 CUDA learners plus native/export LambdaMART). All32 narrow controls qualified at1.0 and34/39 outputs parsed. It ran out of its bounded allowance before prefix/replay queries. A separate channel test failed collection because the execution bundle omitted transitive Evaluation.Artifacts dependencies. This was a packaging failure, not a successful GPU test. Raw evidence and checksums are in repair-validation-v5/.

The packaging closure is corrected and an isolated extracted-bundle subprocess passed its two CPU channel cases (CUDA case appropriately skipped locally). A bounded targeted v6 follow-up is completing the CUDA channel regression and five remaining format queries. It checks equality of the actual official model/revision/NF4/weight hashes against v5 and explicitly prohibits training/label promotion from these isolated responses. Current results belong in the Kaggle map until verified retrieval is complete.

Large-stream mechanical checks also passed for50/60/100 UEs offering2500/3000/5000 messages with churn/bursts and two reference policies. All relevance values were available; observed non-cold novelty availability and queue conservation are in large_stream_checks.json. This does not validate natural temporal semantics or neural-policy performance at5000 queued jobs.


## Final repair checks

V6 completed the five remaining Qwen format checks with matching model/weight provenance. V7 fixed Evaluation.errors packaging and passed all three existing channel tests on two CUDA devices, exit0, after explicitly isolating pytest from parent configuration. Retrieved validation/JUnit/driver hashes verified. Local full release:297passed/3skipped; CUDA-only skipped case now covered remotely. Two actual-codec artifact tests remain outside this task. These results validate implementation paths, not convergence or emergency-priority accuracy.
