# Phase 1 implementation plan

Date: 2026-09-27. Immediate deliverable: a self-contained manual-priority image scheduling demonstration for Monday, September 28. Full extraction/RL development follows gated checkpoints; this document does not claim those components exist.

## 1. Baseline first: exact scope

Build one new notebook, `Scheduler/notebooks/03_manual_priority_baseline.ipynb`, backed by reusable Python modules. It must run independently of the archived classifier notebook, pretrained model downloads, SwinJSCC checkpoints, and RL training. Its inputs are images with an explicit manual priority manifest; its outputs are actual simulated queue traces and image-order plots.

Use 24 distinct mixed images, six manually assigned examples per class: 0 Critical, 1 Emergency, 2 Moderate, 3 Normal. These are demonstration labels under a declared task, not dataset-native ground truth or a classifier evaluation. Curate from existing image assets where suitable, supplement with documented public-source imagery when needed, and keep source IDs, hashes, licensing/source links, label author, rationale, and task context. No class inference from filenames or pre-existing event categories. Label text and class color must appear on the notebook contact sheet so the user can review/edit the assignment.

The baseline proves that a scheduler given priorities respects them. It does not prove the system can infer those priorities. Randomized class labels may be used only in stress tests clearly labeled synthetic, never substituted for the manually reviewed demonstration manifest.

## 2. Jobs, service, and algorithms

Introduce image-job contracts under INFRA with job ID, image ID, arrival time, manual priority, optional deadline, and service duration. One job represents a whole image/latent; no UE ID. Preserve the existing UE-shaped contracts for compatibility rather than redefining their meaning globally. Scheduler reads immutable eligible-job snapshots; only INFRA owns clock, admission, queue updates, completion, and expiry.

Baseline service is an explicitly synthetic nonpreemptive single server, default 10 ms/image. It represents following the service queue and is not a measured radio link or decoder runtime. Later codec integration replaces placeholder payload handles with actual representations without letting the policy inspect raw images/latents. Do not report tensor storage as transmitted bits.

Compare five policies with deterministic ties by arrival sequence then stable job ID:

1. FIFO: earliest arrival.
2. Strict priority: lowest class integer, FIFO within class.
3. EDF: earliest finite deadline, FIFO on ties; missing deadlines sort last.
4. Class round robin: cycle the four nonempty class queues, FIFO within each. Label it explicitly; do not reuse the current UE round robin.
5. Priority with maximum-wait safeguard: choose the oldest job waiting at least 100 ms; otherwise strict priority. This is an explicit starvation-control rule, not an image-parameter weighted score.

Do not add proportional fair or EXP/PF to this first experiment: the chosen model has no per-device throughput history or channel rates. Shortest-job-first can be a later comparison once variable measured service demand matters.

Scenarios use the same manifest and exogenous trace for every policy:

- **Ordering sanity:** all 24 images arrive at time zero, equal 10 ms service, no deadlines, sufficient time to drain. Fixed seeded shuffle, seed 42. This isolates ordering; equal throughput is an expected result.
- **Streaming:** repeat the image pool as uniquely identified jobs, 200 arrivals spaced 15 ms apart, 10 ms service; demonstrate idle periods and no service before arrival. Repeats are workload events, not new independent images.
- **Overload:** 200 arrivals spaced 5 ms apart, 10 ms service, a 20-waiting-job buffer excluding the running job, and 60 ms relative deadlines. Use the same deadline across classes initially to isolate priority behavior. Tail-drop on arrival to a full buffer for every policy.
- **Trade-off stress:** same overloaded arrivals with deterministic service durations drawn from 5, 10, and 20 ms independently of class, and relative deadlines drawn from 30, 60, and 120 ms independently of class. Freeze the trace before policy runs.

Process simultaneous events deterministically: finish current service first (completion exactly at deadline succeeds); expire waiting jobs whose deadlines are reached; admit arrivals in manifest/sequence order; select the next job. A running nonpreemptive job completes even if late and is recorded as late, not silently counted as timely. A queued job expires at its deadline; admission overflow is a separate drop reason. The server never sees future jobs in its snapshot. Runs continue until all offered jobs are completed or dropped; keep any finite-horizon exploratory censoring explicit rather than calling pending jobs drops.

## 3. Notebook outputs and reproducibility

For every policy, show the same before-arrival contact sheet and an after-service contact sheet with service rank, class, start time, and waiting time. For stress traces show a clearly labeled first-24-job window, and display dropped images separately with reasons. Render the actual selected order from the trace, not a fresh sort made only for plotting. Include a compact Gantt/service timeline.

Metrics are derived from job records:

- Waiting = service start - arrival; completion delay = completion - arrival. Show mean, median, p95, and maximum per class with sample counts.
- Completed images per simulated second and timely completions per simulated second, using the same configured observation window when comparing rates; also report total drain time separately. No invented bit throughput.
- Deadline misses among offered jobs with deadlines, distinguishing expired-before-service, completed-late, and overflow-dropped. Show per-class offered/completed/timely/dropped counts to avoid survivor bias.
- Jain's index over per-class timely-service fractions for classes with offered jobs, with the exact formula and denominator displayed; all-zero fractions yield unavailable. Always pair fairness with class outcomes because equal service is not the objective of strict priority.
- Priority inversions among co-eligible jobs in the static test and critical/emergency waiting times in dynamic traces. Strict priority cannot be expected to serve a future high-priority arrival before a job already in service.

Show one compact metrics table, per-class delay plots, completion/drop plots, and fairness/tail-wait trade-offs. Use 20 fixed seeds for stress-trace variability; report descriptive ranges or confidence intervals with the replication unit stated. These are simulation replications, not evidence from 20 independent image datasets.

The notebook includes installation/imports, configuration, seed, environment detection, manifest review, execution, plots, and saving. Core simulation/policy/plot helpers live in modules. Use a guarded clean-reset-and-sync setup for an explicitly owned disposable clone only; never delete the user's local working repository. Support local checkout mode without reset and Colab/Kaggle clone mode. Record exact commit, dirty state/source digest, package versions, hardware, seed, manifest hash, and resolved configuration.

Artifacts go to configured persistent results/Scheduler/run-id: manifest, trace, metrics, figures, config, provenance, and executed notebook. Colab uses a mounted persistent destination; Kaggle uses its output directory with a reminder that persistence requires saving a version/export. Local uses a configured path outside any disposable clone. CPU is sufficient for this baseline; GPU/TPU detection does not imply those backends were tested or needed. Before remote notebook execution, the source commit must be pushed; local commits alone are not remotely reproducible.

## 4. Module ownership and acceptance gates

| Part | Inputs -> outputs | Dependencies | Owner/mode | Acceptance |
|---|---|---|---|---|
| Archive | Current pilot -> original-byte archive and Git recovery point | Git, hashes | Codex / Whiteboard | Four files match stored SHA256; old layout recoverable |
| Label manifest | Mixed images + task/rubric -> 24 explicit labels | Image provenance | Codex drafts; user reviews meaning | Four classes represented; no automatic labels; visible rationale |
| Image simulator | Manifest + arrival/service trace -> lifecycle records | INFRA contracts | Codex / Whiteboard | Hand-calculated schedules, conserved jobs/time, deterministic replay |
| Baseline policies | Immutable eligible jobs -> selected ID | Simulator interface | Codex / Whiteboard, explain policy rules | Exact static order; FIFO ties; RR turns; maximum-wait safeguard |
| Notebook/report | Trace + image manifest -> before/after plots and metrics | Simulator/policies | Codex / Whiteboard | Restart-and-run-all on CPU; order plots equal trace; all outputs saved |
| Image extraction | Image + task + past evidence -> structured fields | Pretrained models, SemParam | Teach and Code: discuss core, Codex builds adapters after agreement | Independent field validation; explicit unavailable evidence; no class leakage |
| Codec integration | Image -> SwinJSCC / codebook payload -> scheduled reconstruction | Existing EncDec checkpoints | Preserve methodology; Codex integration | Actual checkpoint round trip; header/payload identity; accurate units |
| Learned priority/policy | Independent labels + queue environment -> predictor/policy | Validated extraction and simulator | Super Code; design before training | Held-out evaluation and matched heuristic comparisons |

Owning tests run first, then affected integration tests. Required baseline scenarios: empty queue; one job; ties; future arrivals; zero offered load; overload; full buffer; missing deadlines; exact deadline completion; late nonpreemptive completion; expired waiting job; starvation safeguard; stable duplicate image IDs with unique job IDs; reset/replay; invalid classes/durations; corrupted/missing image file; and metrics computed from a small hand-checked trace. No new tests are required merely to check unchanged archived file text; hashes cover preservation.

Current repository command for existing regression checks:

```powershell
.\.venv\Scripts\python.exe -m pytest SemParam/tests Scheduler/tests INFRA/tests Test/tests/test_stage1b_semantic_pipeline.py Test/tests/test_contracts_and_registries.py -q
```

After implementation, notebook execution must be validated from a fresh kernel and the resulting figures inspected. Actual checkpoint inference is an additional codec gate, not implied by baseline tests passing.

## 5. Next design checkpoint

The extraction proposal is [PHASE1_EXTRACTION_RESEARCH.md](PHASE1_EXTRACTION_RESEARCH.md). Review independent object/relationship, task relevance, knowledge novelty, and uncertainty fields first. Finalize the bounded header and neural observation only after inspecting real extractor outputs and overhead. Keep classifier training, RL objective/constraints, value-based scheduling, latent-head training, and the meaning of 'JEV' as explicit later design checkpoints. This prevents a supposedly complete RL specification from inventing the user's objective.
