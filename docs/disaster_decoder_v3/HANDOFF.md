# Start here: disaster semantic decoder scheduler

This is the completed bounded prototype and its measured comparison, not a claim
of a universally superior algorithm or journal readiness.

## Open these files

1. `Scheduler/notebooks/05_disaster_decoder_v3.ipynb`: the actual executed final
   Kaggle notebook, with the 39-row table and two embedded comparison figures.
2. `docs/disaster_decoder_v3/SIMPLE_REPORT.md`: the plain-language explanation.
3. `docs/disaster_decoder_v3/RESEARCH_REPORT.md`: methods, all results, confidence
   intervals, extraction costs, limitations and references.
4. `docs/disaster_decoder_v3/RUNTIME_USAGE.md`: actual module/API integration.
5. `docs/disaster_decoder_v3/ARTIFACT_INDEX.md`: saved-run paths and evidence.

The archive manifest identifies the exact source, dataset-feature cache, model
artifacts and notebook hashes. Frozen extraction was performed on Kaggle with two
T4s; Qwen ran through Ollama on both GPUs. Scheduler training completed in the
original Stage B run. The final notebook resumed that completed run and required
the quality metrics to match, without changing methods based on the test results.

## What the system does

UE image evidence + declared task + local history -> eight parameter values ->
32-byte header -> receiver queue -> selected complete digital payload reference.

The fields are relevance, scarcity, reliability, importance, freshness,
prominence, complexity and novelty. Availability flags preserve missing values.
The receiver does not see the image, embedding, native labels or pending payload
content. Task-specific exported models are loaded without importing Torch in the
Scheduler module. INFRA checks completeness, header CRC, transport token, task
profile and arrival eligibility, and owns queue removal.

## What the results say

The study compared 13 methods on three supported disaster-review tasks. CAHR did
not establish consistent superiority over simpler semantic importance or tree
rankers. The validation-selected exported models are coverage trees for building
damage and structural flooding, and a regressor for roads. The road test showed
that importance-only had a higher point score; that test observation was not used
to replace the preselected model.

Nine retrained field ablations and two additional seeds were evaluated for
building damage. Natural-image freshness was unavailable, and road reliability
was withheld because calibration positives were insufficient. These limitations
are recorded rather than replaced with synthetic urgency labels.

## What is not delivered as a claim

- No real codec training or decoded-image-quality improvement: the teammate's
  complete digital codec adapter is still the integration dependency.
- No real MAC/PHY scheduler, deadline guarantee or physical service-time result.
- No emergency/lives-at-risk ground truth or independent physical-fact matching.
- No proof that all eight fields are necessary, or that the proposed loss is
  historically unprecedented. RankNet and diversity-aware ranking already exist.

The simulator used real image-derived evidence and constructed message traces.
It did not pretend opaque observation references were measured latent payloads.

## Reproduction

Use the saved environment/source manifests for the measured Kaggle runs.
`kernel-metadata.json` under `kaggle/disaster-decoder-v3-final` attaches the verified
extraction and primary training outputs. The executed notebook contains its exact
source snapshot. `--stage all` builds a separate from-source notebook template;
it does not overwrite the delivered executed notebook.

The archive's blank package initializers isolate the additive V3 modules from
historical project dependencies. They are not a migration of the user's existing
repository. Legacy files, old notebooks, datasets and codec work remain preserved
in the workspace; no commit/push was made.
