# Semantic whole-latent scheduler runbook

## What is implemented

The scheduler selects the next **whole SwinJSCC latent job** among currently
eligible jobs from five simulated user equipments. `SemParam` extracts CLIP
context relevance (P2) and a cached embedding; `INFRA` recomputes novelty (P3)
against successfully delivered images at each decision. P5 task impact and P6
urgency come from a fixed task rubric. For this prototype P7 is explicitly
defaulted to one, with no human review claim. A fixed 24-byte V2 header describes these five values and receiver
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

## Fixed prototype data

The source draft has 60 distinct LADI v2 images selected from six multi-label
strata, with 30 CLIP-proposed two-image evidence groups. The prototype keeps all
images and pairings exactly as selected. It creates a **separate**
`prototype_fixed` manifest with P7=1 for every image. These pairs and P7 values
are assumptions, not independently reviewed evidence. The source draft and
image files stay unchanged. The validator checks all image SHA-256 hashes,
strata, and the 36/12/12 group-safe split. Results measure a controlled
simulator objective and cannot establish real-world disaster triage quality.

The private Kaggle bundle includes the 60 PNGs, prototype manifest and the
checkpoint-derived `payload_costs.csv`. Its measured channel-use values are
uniform for this fixed SwinJSCC rate; SJF consequently has no meaningful
cost variation to exploit. Do not invent variable costs.

The class rubric is task-specific supervision, not universal disaster priority.
P5/P6/P7 results must be called **prototype-rubric-assisted**. The evaluation cells
also mask these fields to show P2/P3 input sensitivity. Static
class-priority targets cannot teach the supervised classifier/ranker the
decision-time novelty effect; the DQN and semantic heuristic are the dynamic
novelty comparators. All image-disjoint and closed-catalogue results are kept
separate.
The P2/P3-only semantic-greedy comparator uses automatic semantic fields, but
still sees simulator deadlines generated from rubric urgency. Masking inputs to
classifiers, ranker or DQN trained with all five fields is an out-of-distribution
stress test, not a retrained automatic-only model comparison.

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

## Kaggle training and same-run evaluation

Only the canonical notebooks under `Scheduler/notebooks/` are edited. The
staging script copies them into `kaggle/train` and `kaggle/evaluate`, discovers
the authenticated username, and writes private `kernel-metadata.json` files.
The first notebook cell removes only its disposable `project-src` checkout,
clones `semantic-aware-ai-scheduler`, fetches the branch, detaches at its latest
remote commit and records its SHA. Notebook 1 now runs the full policy comparison,
plots, faculty walkthrough and claim ledger immediately after training, saving
all evaluation outputs under the same run directory. Notebook 2 remains available
for later independent reruns and rejects any training artifact produced by a
different SHA or manifest hash.

Required inputs:

- `SCHEDULER_MANIFEST`: prototype CSV with adjacent PNG files; the packaged
  Kaggle dataset supplies the default path.
- `SCHEDULER_TRAIN_RUN_DIR` for evaluation: persisted training-output folder.
- For measured training, the packaged dataset supplies checkpoint-derived
  `payload_costs.csv`. The notebook also accepts `SCHEDULER_PAYLOAD_COSTS_CSV`
  or can measure with a compatible checkpoint at `SCHEDULER_SWIN_MODEL_DIR`.
  `SCHEDULER_PAYLOAD_MODE=simulated_fixed` is a separately labelled smoke mode.
- `SCHEDULER_RESULTS_DIR` may override `/kaggle/working/outputs` or the local
  persistent default.

From the repository root, freeze a separate prototype manifest and package the
existing data. The installed Kaggle CLI creates datasets privately by default.
Authenticate with `kaggle auth login` first if the API reports an auth error.
After committing and pushing the source branch, run:

```powershell
.\.venv\Scripts\python.exe scripts\freeze_ladi60_prototype.py --source results/ladi-curation/v1/ladi60_semantic_scheduler_v1.csv --output results/ladi-curation/v1/ladi60_semantic_scheduler_v1_prototype.csv
.\.venv\Scripts\python.exe scripts\package_ladi_review_dataset.py --manifest results/ladi-curation/v1/ladi60_semantic_scheduler_v1_prototype.csv --costs results/ladi-curation/v1/payload/payload_costs.csv --output-dir results/kaggle-ladi60-prototype --zip
kaggle datasets create -p results/kaggle-ladi60-prototype
.\.venv\Scripts\python.exe scripts\prepare_semantic_kaggle.py train --manifest-dataset <username>/semantic-scheduler-ladi60-prototype
kaggle kernels push -p kaggle/train
kaggle kernels status <username>/semantic-scheduler-train
kaggle kernels logs <username>/semantic-scheduler-train
kaggle kernels output <username>/semantic-scheduler-train -p downloaded/train-run
```

The three DQN seeds each have a 40-minute wall-clock cap, so their combined
training budget is two hours. CLIP extraction, evaluation and Kaggle setup are
additional runtime. The notebook records completed episodes and whether a seed
stopped at its cap; it does not claim that every seed completed 300 episodes.

To run notebook 2 later on the exact trained commit:

```powershell
.\.venv\Scripts\python.exe scripts\package_semantic_train_output.py --run-dir <downloaded-run-dir> --output-dir <private-train-package>
kaggle datasets create -p <private-train-package>
.\.venv\Scripts\python.exe scripts\prepare_semantic_kaggle.py evaluate --manifest-dataset <username>/semantic-scheduler-ladi60-prototype --train-output-dataset <username>/semantic-scheduler-train-output
kaggle kernels push -p kaggle/evaluate
kaggle kernels status <username>/semantic-scheduler-evaluate
kaggle kernels output <username>/semantic-scheduler-evaluate -p downloaded/evaluation-run
```

Kaggle requires the private prototype-image dataset and, for notebook 2, the
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
