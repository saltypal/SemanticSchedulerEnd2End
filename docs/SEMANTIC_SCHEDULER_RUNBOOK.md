# Semantic whole-latent scheduler runbook

## What is implemented

The scheduler selects the next **whole SwinJSCC latent job** among currently
eligible jobs from five simulated user equipments. `SemParam` extracts CLIP
context relevance (P2) and a cached embedding; `INFRA` recomputes novelty (P3)
against successfully delivered images at each decision. Reviewed P5 task impact,
P6 urgency and P7 evidence sufficiency come from the task rubric and human
annotation. A fixed 24-byte V2 header describes these five values and receiver
history version. The header is a logical sidecar; it has no implemented physical
control channel. `EncDec` encodes once and can replay selected latents through
the real SwinJSCC checkpoint and AWGN channel.

The queue, arrivals, service time, deadlines and channel success are simulated.
No code in this branch allocates 5G resource blocks. The DQN is a small standard
459→64→32→25 next-job policy; PyTorch training lives in `INFRA/learning`, while
`Scheduler` uses exported NumPy weights. FIFO, round robin, privileged fixed
priority, EDF, priority+EDF, SJF, aging, semantic greedy, LADI predicted-label
priority, ordinal classifier and pairwise ranker share the same episode input.
A privileged one-step marginal-utility reference also uses evaluator-only
priority and evidence-group facts. It is not a bound on the globally optimal
schedule and must not be compared as an ordinary deployable policy.

## Current data gate

`data/manifests/ladi60_semantic_scheduler_v1.csv` is a **pending draft** with 60
real, distinct LADI v2 images selected from six multi-label strata. Its 30
two-image evidence groups are automatic similarity *suggestions*. Neither the
pair relationship nor P7 has been human reviewed. The corresponding image files
and contact sheet are in ignored local `results/ladi-curation/v1/`; no LADI image
bytes are committed. Training and evaluation deliberately reject this draft.

Before training, review the contact sheet and each underlying image. Correct any
evidence group that does not represent a defensible related observation; keep
groups of two within one split. For each row, set `evidence_sufficiency` to 0 or
1, `reviewer` to a real reviewer name, and `review_status` to `reviewed`. If
overriding impact, urgency or base priority, record the rationale. Keep the
images in an `images/` folder next to the reviewed CSV, then upload this bundle
as a **private** Kaggle dataset or provide a local path. The validator verifies
every image SHA-256 and the 36/12/12 group-safe split.

For faster review, run `python scripts/build_ladi_review_page.py --manifest
results/ladi-curation/v1/ladi60_semantic_scheduler_v1.csv --output
results/ladi-curation/v1/ladi60_review.html` and open the generated local HTML.
It shows 30 paired cards, asks for both P7 values, requires confirmation of
each related pair and a real reviewer name, then downloads a reviewed CSV.
Move the downloaded CSV beside the original `images/` folder before validating
or packaging. A rejected pair must be regrouped in the draft and reviewed
again; the page never silently approves it.

The class rubric is task-specific supervision, not universal disaster priority.
P5/P6/P7 results must be called **oracle-assisted**. The evaluation notebook
also masks these fields to show the P2/P3-only automatic ablation. Static
class-priority targets cannot teach the supervised classifier/ranker the
decision-time novelty effect; the DQN and semantic heuristic are the dynamic
novelty comparators. All image-disjoint and closed-catalogue results are kept
separate.
The P2/P3-only ablation masks semantic fields P5/P6/P7 at the policy input;
it still exposes simulator deadlines and age. Because deadlines are generated
from the reviewed urgency rubric, this is a feature ablation within the
simulator, not a claim of fully annotation-free end-to-end triage.

With one fixed 1024×1024 SwinJSCC rate for every image, measured complex-channel
uses may be identical. In that case SJF and FIFO have the same scheduling key;
this is an experimental result, not a reason to invent cost variation.

## Local validation

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest SemParam\tests INFRA\tests Scheduler\tests -q
.\.venv\Scripts\python.exe scripts\build_semantic_scheduler_notebooks.py
.\.venv\Scripts\python.exe scripts\prepare_semantic_kaggle.py both
```

To regenerate a fresh pending LADI draft from the official dataset:

```powershell
.\.venv\Scripts\python.exe scripts\curate_ladi60.py --output-dir results\ladi-curation\v1 --source-split validation --pool-per-stratum 10
```

The codec checkpoint exported under Python 3.12/PyTorch 2.10 has a strict
runtime guard. The existing `.venv-models` environment matches this checkpoint;
the general `.venv` does not. A one-image measured smoke run is a legitimate
codec check. Use `EncDec.integration.scheduler_replay.cache_swinjscc_payloads`
and `replay_selected_jobs` with `models/SwinJSCC` and `external/SwinJSCC`.
The codec configuration requires 1024×1024 preprocessing. Latent tensor storage
bytes, payload bits and complex channel uses are distinct fields.

## Kaggle training and evaluation

Only the canonical notebooks under `Scheduler/notebooks/` are edited. The
staging script copies them into `kaggle/train` and `kaggle/evaluate`, discovers
the authenticated username, and writes private `kernel-metadata.json` files.
The first notebook cell removes only its disposable `project-src` checkout,
clones `semantic-aware-ai-scheduler`, fetches the branch, detaches at its latest
remote commit and records its SHA. The evaluation notebook repeats this and
rejects any training artifact produced by a different SHA or manifest hash.

Required inputs:

- `SCHEDULER_REVIEWED_MANIFEST`: CSV with adjacent `images/` folder.
- `SCHEDULER_TRAIN_RUN_DIR` for evaluation: persisted training-output folder.
- For measured training, either a compatible SwinJSCC checkpoint at
  `SCHEDULER_SWIN_MODEL_DIR` or a verified `SCHEDULER_PAYLOAD_COSTS_CSV` from
  checkpoint inference. If the private dataset
  `semantic-scheduler-measured-costs` is attached, the training notebook finds
  its `payload_costs.csv` automatically. `SCHEDULER_PAYLOAD_MODE=simulated_fixed` is a separately
  labelled infrastructure smoke mode.
- `SCHEDULER_RESULTS_DIR` may override `/kaggle/working/outputs` or the local
  persistent default.

After reviewing and packaging the LADI subset, create its private Kaggle
dataset. After committing and pushing the source branch, run:

```powershell
.\.venv\Scripts\python.exe scripts\package_ladi_review_dataset.py --manifest <reviewed.csv> --output-dir <private-ladi-package>
kaggle datasets create -p <private-ladi-package> --private
.\.venv\Scripts\python.exe scripts\package_semantic_payload_costs.py --manifest <reviewed.csv> --costs results/ladi-curation/v1/payload/payload_costs.csv --output-dir <private-cost-package>
kaggle datasets create -p <private-cost-package> --private
.\.venv\Scripts\python.exe scripts\prepare_semantic_kaggle.py train --reviewed-dataset <username>/semantic-scheduler-ladi60-reviewed --payload-cost-dataset <username>/semantic-scheduler-measured-costs
kaggle kernels push -p kaggle/train
kaggle kernels status <username>/semantic-scheduler-train
kaggle kernels logs <username>/semantic-scheduler-train
kaggle kernels output <username>/semantic-scheduler-train -p downloaded/train-run
.\.venv\Scripts\python.exe scripts\package_semantic_train_output.py --run-dir <downloaded-run-dir> --output-dir <private-train-package>
kaggle datasets create -p <private-train-package> --private
.\.venv\Scripts\python.exe scripts\prepare_semantic_kaggle.py evaluate --reviewed-dataset <username>/semantic-scheduler-ladi60-reviewed --train-output-dataset <username>/semantic-scheduler-train-output
kaggle kernels push -p kaggle/evaluate
kaggle kernels status <username>/semantic-scheduler-evaluate
kaggle kernels output <username>/semantic-scheduler-evaluate -p downloaded/evaluation-run
```

Kaggle requires the private reviewed-image dataset and, for notebook 2, the
private persisted training-output dataset to be attached to the kernels. The
staging script fills `dataset_sources` only from the explicit slugs given above;
without them it leaves the list empty and the notebook fails its input gate.

## Faculty walkthrough and interpretation

1. Open the 25-image faculty grid and five-UE arrival table.
2. Inspect one queue snapshot: five parameters, availability, deadline,
   resource units, scheduler score, decision and 24-byte header.
3. Follow a related image pair through successful delivery and the novelty
   recalculation. Failures, drops and expiration never update receiver history.
4. Compare utility-over-time, critical-before-deadline, inversion severity,
   latency, resource units and fairness across exactly shared streams.
5. Show static nDCG/Kendall only for the all-at-once fixture; staggered arrival
   streams use utility, deadline, inversion and fairness metrics.
6. Show SwinJSCC reconstruction only when actual checkpoint replay succeeds.

The baseline and reward share a task rubric. This makes the experiment a
controlled demonstration of a chosen objective, not proof of general
real-world triage performance. If the DQN fails to beat a transparent heuristic,
report that outcome. One T4 is enough for this small network; GPU or CPU
runtime does not turn simulated network outcomes into physical 5G evidence.
The Kaggle metadata requests its T4 accelerator shape; training uses only
`cuda:0` even if the runtime allocates two T4 devices.
