# Paste this assignment into your Codex chat

**Current sequence:** use [PLAN.md](PLAN.md)'s active phases. Begin with the
Phase 0 SPI data audit/classifier; defer the reviewer pilot below to optional
Phase 2. The historical assignment does not authorize skipping Phase 0/1.

The full engineering handoff is [HANDOFF.md](HANDOFF.md). Give Codex the text
below after opening the cloned repository. This is an assignment from the project
owner, not a request to make a successful-looking result.

---

You are taking over the **offline validation slice** of BetterScheduler in the
`SemanticSchedulerEnd2End` repository. Start from branch `new-scheduler`, then
create your own branch `teammate/json-v2-validation`. Do not push to the shared
branch, overwrite experiment outputs, or launch old notebooks automatically.

Read these in order:

1. Root `AGENTS.md`, `PROJECT_MAP.md` and `BetterScheduler/HANDOFF.md`.
2. `BetterScheduler/PLAN.md`, giving priority to the handoff status/current
   objective over older historical launch notes.
3. `BetterScheduler/docs/OFFLINE_VALIDATION_REDESIGN.md`.
4. `BetterScheduler/docs/SCENE_JSON_V2_AND_MODEL_SETTINGS.md`.
5. `BetterScheduler/docs/SEMANTIC_VALIDATION_I_RESULTS.md` and `KAGGLE_RUNS.md`.
6. The modules and tests named in the handoff's ownership and build sequence.

## Goal

Determine whether JSON v2 correctly represents the observable scene evidence
needed for a declared operator mission and whether it supports an operator
action-urgency decision. Decoder priority comes afterward. Qwen is an offline
reviewer under evaluation; it will not run in the decoder scheduler.

Do not replace this with generic image-caption scoring, a policy leaderboard,
more Qwen approvals, or a claim that 12 images prove universal scene understanding.
The schema is domain-independent. The existing producers are not universally
accurate, and the optional LADI classifier is specifically an aerial-domain model.

## First turn: inspect and report, without a GPU launch

- Resolve OS, checkout, branch, HEAD, dirty state, Python interpreter and dependencies.
- Read the handoff's evidence/status table and independently verify its main claims.
- Tell me which artifacts are included in Git and which must be restored.
- Explain the three review routes and exactly what each may see.
- Show the unresolved issues that could change the experiment's interpretation.
- Propose the smallest first local milestone. Do not start a large redesign.

Then continue with authorized local fixes and tests. Use one canonical plan and
keep findings/results in artifacts; never manufacture certainty or expected results.

## Authorized ownership

You own `scene_review.py`, `scene_review_run.py`,
`build_scene_review_notebook.py`, their versioned experiment configs and tests,
plus review-specific documentation and run bookkeeping. Coordinate before
changing the producer algorithms, main JSON schema, mission definitions or
scheduler. You are not alone: preserve other contributors' edits and originals.

## Hard boundaries

- Received messages remain `[ParameterHeader][CompleteLatentPayload]`, queued for
  one decoder. No resource grants, radio scheduling or encoder redesign here.
- No paid APIs. A bounded Kaggle experiment uses the existing account's free
  GPU allocation; current access/quota must be verified rather than assumed.
- No new weights, datasets, cloud run or package upgrade is needed for your first
  local milestone. Public restoration of the 12-image fixture is allowed once
  using the bounded restore command in HANDOFF.md, if it is absent.
- CPU tests must use a project Python 3.12 environment, not a global Python 3.14
  fallback. Preserve the actual interpreter/package/source identity in receipts.
- Unknown, available false, available zero and cold start have different meanings.
  Do not fill missing evidence using an evaluator label or a model guess.
- Independent annotations are reference truth only within their documented scope.
  Qwen/Luna/CLIP agreement is not an independent label.
- Existing LADI labels do not establish emergency urgency. The current fresh/stale
  policy is explicitly synthetic and must stay labeled as such.
- Do not silently feed v2 JSON to the archived v5 reviewer. Preserve old results.

## Required local milestone

1. Restore/verify the fixture where needed; report any fixture-dependent skips.
2. Run the BetterScheduler test suite and examine failures rather than weakening tests.
3. Build the 44-case v2 pilot locally without loading a model.
4. Verify JSON-only has no image/reference answers; image-only has no JSON;
   annotation-assisted auditing is clearly separated.
5. Verify the newly separated policy-following/reference-correctness metrics when
   extracted claims are wrong/missing. Inspect the existing regression and extend
   it where necessary rather than duplicating an already completed correction.
6. Verify both package manifests agree on actual budgets, source ZIP and upstream.
7. Produce `LOCAL_VALIDATION_REPORT.md` with checks, failures, source identity,
   limitations, proposed corrections and a go/no-go recommendation.

## Bounded remote pilot, only after local acceptance

The previous owner proposed one private Qwen3.5-9B NF4 run on two T4 GPUs,
thinking OFF, server1800/internal1650seconds (at most one aggregate GPU-hour).
The current workload is 44 calls across six exposed development parents. It is
not launched yet. Present the concrete package and local evidence to me before
spending GPU quota. Do not infer approval from a historical launch command.

Record actual assigned devices, model/processor/weights, finite logits, input
tokens/pixels, memory, raw outputs and partial results. Test a real full JSON first.
No automatic retry or chained experiment. Notebook COMPLETE is not experiment
success: inspect its own status and retained outputs. Verify persistence/retrieval.

## What success means

An honest, reproducible answer to what the current representation preserves,
what it misses, and what Qwen can reliably judge. Negative results are valid.
Deliver separate factual, encoded-answer, policy-following and reviewer metrics.
State unsupported relation/urgency claims explicitly. Do not train a scheduler
until this evidence is sufficiently clear.

At each milestone explain the outcome, evidence, remaining uncertainty and next
smallest useful action. Ask only for decisions that materially change scope,
resources or external impact. Do not ask repeatedly for already-authorized local work.

---

## What to send back to the project owner

- Branch/commit and concise change summary.
- Test logs and interpreter/source identities.
- Three concrete image/JSON failure examples where supported.
- Per-route/per-arm metrics, including invalid/unattempted counts.
- What the evidence establishes and does not establish.
- The next decision required, if any. No need to make a positive result.
