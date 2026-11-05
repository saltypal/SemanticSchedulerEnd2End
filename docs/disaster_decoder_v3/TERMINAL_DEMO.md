# Terminal demonstration

From the repository root:

```powershell
python scripts/demo_decoder_scheduler.py
python scripts/compare_original_schedulers.py
python scripts/demo_decoder_scheduler.py --model selected
python scripts/demo_decoder_scheduler.py --task road_damage
```

Default: load the first three held-out image-model cache rows in file order, without
label/score filtering, and repeat the first observation as a fourth message.
Recalculate all eight parameters using causal sender history, serialize/parse the
32-byte headers, run the actual ordinary neural RankNet checkpoint and dispatch
through DecoderQueue. CAHR remains available explicitly through `--model cahr`.
The original eight supported algorithms can be compared with
`python scripts/compare_original_schedulers.py` on the identical headers.
Print each field, missingness, header hex, ranking score, dispatch order and JSON
save location. No model retraining or Kaggle allocation is needed.

This is cached image-evidence replay, not live image-model inference, and it does
not encode/decode latent bytes. The repeated observation gets a new transport token;
its visual novelty becomes zero. That does NOT guarantee the learned ranker places
it last. The observed building example actually orders the repeat ahead of the
original. This shows N is not a hard monotonic constraint, but does not by itself
prove a scheduling error: either copy can provide the first observation. Quality
depends on the other available evidence and the objective. This walkthrough does
not expose native truth labels or compute a reference utility metric. Retain its
output without calling the order a validated semantic failure or success.
Scores are not probabilities or a feature-level explanation of why a decision occurred.

For live extraction on user-provided images (requires model downloads/dependencies):

```powershell
python scripts/demo_decoder_scheduler.py --images "C:\path\image1.jpg" "C:\path\image2.jpg" --device auto
```

Live mode is implemented but not validated on this CPU host in this checkpoint.
The cache replay and actual checkpoint inference are tested. Default outputs are
unique timestamped JSON under `results/disaster_decoder_v3/terminal_demo/`.
`--output` accepts an explicit new path; existing files are never overwritten.

## Qwen checkpoint correction

Inspection of the saved Ollama proposals found 32/32 syntactically valid outputs,
but 0/32 assessable building, road or structural-flooding assessments. Qwen ran on
both T4s but did not deliver useful annotations. These are abstentions, not negative
damage labels, and were not scheduler training targets. Valid JSON is not semantic
success. The earlier handoff's valid-output count alone was insufficient to establish
annotation usefulness. Image preprocessing/prompt behavior and a small visual audit
must be checked before any further Qwen labeling campaign; no cause is assumed here.
