# Bounded teacher/recovery design assessment

Scope: assessment only, no production/test/notebook edits. Inspected current contracts and existing regression coverage; no new GPU inference or end-to-end training executed by this reviewer.

## Recommendation
Proceed with labeled current/history presentation and explicit aspect rubric, while preserving the controls, independent swapped queries, 80% consistency threshold, strict support requirements and reward validation. The presentation diagnosis is plausible and supported by answers describing history; it is not a demonstrated universal cause of Qwen inconsistency. A new pilot must decide.

## Consequential findings
1. Confirmed prompt/consumer mismatch: teacher_pipeline.py tells the candidate teacher to leave aspects unknown, while VisionPrefer needs 16 strict train and8 strict validation aspect preferences. Schema already requires aspect objects. Ask anchored 0–4-or-unknown ratings without forcing unsupported values.
2. Candidate image interleaving can plausibly cause current/history confusion. Render both current candidates prominently, with histories visibly subordinate, and rerender both independent orientations. Do not obtain a passing swap score by reusing or mechanically reversing an earlier answer.
3. The same neutral visual binding repair is needed for prefix/replay Evidence IDs; candidate-only rendering leaves the feedback arms exposed to the original multi-image binding difficulty.
4. Source UE windows are recent arrived observations, not necessarily older than the selected candidate. One current candidate may also occur in the other candidate's window. Preserve actual arrival times, mark current cross-references, and avoid implying successful delivery.
5. Existing source-history status ternary ignores failed_tokens and calls every non-waiting/non-inflight observation processed. Distinguish failed/unknown; receiver processed history contains successful completions only.
6. New layout/prompt/pixels require a fresh teacher output directory and explicit provenance; preserve old failed responses. Cache identity currently binds output grammar/model/weights/requests but not processor settings explicitly.
7. Rate script now bootstraps imports, but source hashing at lines44–46 is cwd-dependent. The production parent uses repo cwd, so that path is valid; normalize to ROOT if claiming arbitrary launch cwd.
8. Existing PairRun.execute('wsl') retrains all ten D models. Targeted recovery must schedule only rate study and teacher-dependent methods in new destinations; never mutate completed checkpoints.

## Required acceptance checks
- Render fixtures with distinct solid-color current/history images: exact source mapping, no cropping, no omitted source image, failed vs successful vs waiting roles, equal current panel area.
- Swap twice restores original mapping/labels; swapping does not alter mission, delay or lifecycle facts; no image features/labels/future facts are injected.
- Prefix renderer visibly links every Evidence ID to the same images and equal-time plans; swapping only schedule labels preserves common evidence.
- Consume actual request/schema format, not only a hand-authored approximation. Unknown aspect ratings remain unknown; strict ratings exercise all seven students through existing fixture tests.
- Fresh output/cache identity rejects old responses and changed renderer/processor settings.
- Full small rate CLI run from the intended sterile runtime, then full declared UE/rate grid; --help alone verifies import only.
- Mandatory teacher controls and8-pair pilot unchanged; all40 pair gate and strict train/validation support enforced. Passing is necessary, not independent urgency truth.
- Recovery ledger retains nine explicit outcomes and rate outcome even on failure; no14-model retraining. Profile CPU rate job separately from GPU teacher/learners.

Residual limitations: a consistent VLM can be consistently wrong; presentation changes are developmental protocol selection. Improved agreement is not a final-test accuracy result. The two feedback arms can still fail prefix/reward/replay-support gates even after candidate qualification. Do not lower gates or manufacture strict preferences to make all arms run.

## Inspected source SHA256

{
  "experiments/stream_scheduler/teacher_pipeline.py": "09da393241d8b544381b0cb15789c228a522740022449fe2bbc4ca620fa6ea31",
  "experiments/stream_scheduler/qwen_teacher.py": "036bb10ece38156fa33185f2125cab01e92e39cd92e9dbad665a20e30e2eb074",
  "experiments/stream_scheduler/train_teacher_rankers.py": "d8ee972d4c50776d7658346988ce4f881526fec8b9b4d551623c257b5800ed97",
  "experiments/stream_scheduler/train_feedback_rl.py": "a08e6b0548e7dc0a1c877a08bad7a31fbe7454b38697cd32a59dc65529198314",
  "experiments/stream_scheduler/ue_rate_study.py": "8b0510e94b98d41d6a08cf1ae61afbe1c8f779c674d7eb2ba5961383951ee1a6",
  "Evaluation/semantic_learning_env.py": "671affafec680012b27eb8ee7149dbc175086cc1a79e46a1abdcc9387858c2b6"
}


## Actual implementation checkpoint

Inspected the new panel renderer, recovery subclass, candidate/prefix query construction, partial-aspect parser/aggregation and visual protocol identity. Executed PYTHONPATH=. python -m pytest experiments/master_study/tests/test_gap_recovery.py -q in the existing WSL environment: **7 passed in9.50s**, no skips. Exact output: assessment-gap-tests.log. These are CPU fixtures/import/contract checks, not new GPU teacher qualification.

No evaluator-label or future-arrival access found in contrast_pair: it reads permitted SPI/relevance/novelty feature values/masks and stable tokens for tie-breaking. First8 training contexts keep the previous candidate rule; other queries are deliberately contrast-conditioned and disclosed. No completed D model is scheduled by TeacherRecovery.execute.

### Remaining findings sent to main

- **P2 resume behavior:** TeacherRecovery initially always regenerates prepared data and omits --resume for existing student/feedback outputs. A resumed run can fail at output mkdir or identity binding. Main should add compatible existence checks/resume flags, or explicitly disable recovery resume. Fresh first launch does not trigger this.
- **P2 wording:** prompt initially says only separate receiver-history row was processed, though recent-window entries can also be explicitly processed. Say only explicit successfully-processed status establishes receipt; recent-window membership alone does not.
- **Methodology change:** partial-aspect aggregation on >=2 common dimensions including mission usefulness preserves unknowns but changes eligibility/target versus original four-dimension complete-case average. Keep explicit adaptation label and support/dimension diagnostics; do not describe all label gates as unchanged.
- **Visual QA:** three recent-history labels occupy~170px tiles; unwrapped17px strings may overlap. Inspect actual dense panels and fit/shorten text as needed.

Current CPU tests support role binding, prefix membership receipts, partial unknown preservation, teacher cache rejection and no completed-D scheduling. They do not yet establish full GPU processor pixel behavior, teacher accuracy, successful nine-arm training, or resume correctness.

Actual-checkpoint source SHA256 (mutable checkout):

{
  "experiments/stream_scheduler/teacher_panels.py": "b62909fd3311de880ec11d53862934382313a65f771c9cef1be62696693e31bb",
  "experiments/master_study/teacher_recovery.py": "c6dc007fdd299659d8fd9ca1d9a5466a172abf32500531487c67e509a6f862ca",
  "experiments/master_study/tests/test_gap_recovery.py": "ae0b763b86ec30dcd5daab4bd02856d93ac193db5a7febde2605f928c16f17a8",
  "experiments/stream_scheduler/teacher_pipeline.py": "0f12fc815a0f0824bf768a14f9b177abecea50ecb4fe8162eaff173c338310be",
  "experiments/stream_scheduler/qwen_teacher.py": "2a7af2b99a8836ded33ad948274d49836e57777f772a8a1ab8fd14fbd038439c",
  "experiments/stream_scheduler/teacher_output.py": "5597e97417c30e1f3189cb51b0c99a25855e29fbae8a05f8d6a4ce21877892a7"
}
