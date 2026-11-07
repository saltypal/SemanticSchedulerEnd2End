# Reconsidering semantic evidence, scene conditioning and dataset diversity

Date: 25 October 2026. Research recommendation for discussion, following the user's
challenge to the eight scalar parameters. No extraction model, header schema,
training result or final benchmark is changed by this note. The existing V3
implementation and its results remain the baseline described in BUILD_STATUS.md.

## Decision and boundary

Recommend testing one shared image representation with overlapping scene/event
outputs, operator mission conditioning, causal transmitter KB evidence, and raw
age/deadline metadata. Add localized measurements only where labels and measured
scheduling value support them. Do not start with one resident YOLO per scene.

The receiver still schedules complete received messages for one nonpreemptive
decoder. All new RGB interpretation happens where the transmitter image is already
available. Receiver inference consumes transmitted summaries and allowed header
history. Evaluator annotation facts remain separate.

The motivating issue is information validity, not classifier size. Neither
"disaster" nor an image embedding distance establishes operational priority.

## Evidence ledger

1. **Incidents1M** (2022 preprint; later journal publication): original full text,
   dataset/model/negative-label sections inspected. It supports 43 incident and
   49 place labels with a shared ResNet-50 and independent prediction branches.
   Section III-D distinguishes 977,088 images with incident-positive labels from
   810,066 images having class-negative labels and no incident-positive labels.
   A category-negative label does not certify that an image is globally benign.
   Adding class-negative training changes incident mAP from 62.83 to 67.19 in its
   study. This supports hard-negative collection, not a scheduler improvement claim.
   Source: https://arxiv.org/html/2201.04236v1

2. **MEDIC**: official schema inspected. Provides informativeness, disaster type,
   humanitarian and damage-severity labels, including non-disaster/non-informative
   categories. Contains CrisisMMD; do not treat them as independent sources.
   Humanitarian categories do not give verified individual distress, rescue urgency
   or natural cross-view object correspondences.
   Source: https://crisisnlp.qcri.org/medic/

3. **LADI**: official overview inspected. Includes class presence/absence and damage
   assessment labels, not a universal emergency-priority scale. Disaster collection
   membership must not be used as an importance label.
   Source: https://github.com/LADI-Dataset/ladi-overview

4. **Personalized Saliency in Task-Oriented Semantic Communications** (2022):
   original full text inspected. Uses user query matching and scene-graph triples
   to condition image information on interests. Supports mission-conditioned
   relevance, not our exact scalar formula or natural-language generalization.
   Its encoder/resource-allocation boundary differs from our receiver scheduler.
   Source: https://arxiv.org/abs/2209.12274

5. **YOLO-World** (2024): original method and offline-vocabulary sections inspected.
   Supports one text-conditioned detector with reusable vocabulary embeddings;
   different scene vocabularies need not mean different visual model weights.
   Its detection benchmarks do not validate distress, medical status or urgency.
   Source: https://arxiv.org/html/2401.17270v3
   Official implementation: https://github.com/AILab-CVC/YOLO-World

6. **Grounding DINO** (2023): original method/evaluation sections inspected. Supports
   category/referring-phrase localization. Useful comparison to YOLO-World, not
   proof that arbitrary phrases are reliably detected in our imagery.
   Source: https://arxiv.org/html/2303.05499v4

7. **CLIP** (2021): original paper/official tokenizer inspected. Provides image-text
   matching and reusable embeddings; the original implementation has a 77-token
   text context. An unrestricted mission paragraph needs explicit clause handling
   or an appropriate alternative model. Similarity is not calibrated probability.
   Sources: https://proceedings.mlr.press/v139/radford21a.html
   https://github.com/openai/CLIP/blob/main/clip/clip.py

8. **On Calibration of Modern Neural Networks** (2017): original abstract/method
   account inspected. Confidence can disagree with observed correctness. Validation
   calibration is useful but does not establish unseen-domain reliability.
   Source: https://proceedings.mlr.press/v70/guo17a.html

9. **DoTA**: official dataset/reference repository inspected by research delegate.
   Video anomaly intervals/classes and tracks may support a later road-domain
   study. Interval labels are not automatically visible accident labels for every
   individual frame. All extracted frames from a clip/event must remain grouped.
   Source: https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly

Literature supports mechanisms, not an optimal combination or novelty claim for
this system. No new models were executed or datasets downloaded for this review.
The Incidents project website timed out; actual sample access remains unverified.
AlphaXiv discovery also returned recent leads not inspected; those are not evidence.

## Resolve the categories before choosing scalars

Use overlapping descriptions rather than one mutually exclusive scene class:

- Environment/place: road, residential area, indoor, forest, river.
- Event/condition: collision, fire, flooding, collapse, blockage; absence/unknown
  only when the model/annotation supports that judgment.
- Entities: people, vehicles, buildings, roads and other mission targets.
- Attributes/relations: building damaged, road blocked, person on roof, where
  labels support extraction. A person box does not establish distress.
- Mission: what the operator wants to learn, where relevant, and when it is needed.

The same image can be a road scene with flooding, people and damaged buildings.
Hard selection of one branch risks discarding valid evidence after a scene error.
Scene tags should initially inform one scorer/applicability mask; no class should
hard-code "all emergencies outrank all ordinary images".

## Recommended treatment of the old fields

| Existing item | Recommendation | Reason / retaining evidence |
|---|---|---|
| R, context relevance | Keep concept; compare text-only similarity with evidence-conditioned relevance | Operator task paragraph is a useful input. Validate relation/negation and task changes; a cosine is not ground truth. |
| N, KB novelty | Keep causal TX KB; compare whole-image retrieval with structured evidence changes | Same/different embeddings do not certify same/new task facts. Sender knowledge and receiver-delivered knowledge differ. |
| S, scarcity | Keep as optional rarity diagnostic/ablation, not automatic priority or viewpoint novelty | Rare irrelevant material can be useless. Define reference population/window/support; let held-out scheduling effects decide. |
| I, weighted task-probability sum | Retire as the proposed primary importance definition; preserve as a baseline | Existing hand-weighted aggregation does not establish emergency utility. Class probabilities remain useful uncertain observations. |
| F, exponential freshness | Replace in proposed design with capture timestamp and operator-specified allowed age/deadline | Default decay constant is not validated for disaster missions. Derive candidate age/slack at receiver; unavailable timestamps remain masked. |
| P, largest target box area | Optional target-specific visibility/scale measurement | Pixel area does not give physical distance or urgency. Small relevant objects can matter more than large backgrounds. |
| C, detector count/entropy | Diagnostic; replace with task-relevant supported entities/facts if useful | Raw scene density can favor crowds/traffic without useful added evidence. |
| Q, max(p,1-p) | Use claim-level reliability/calibration and availability, rather than generic high confidence | Confident absence must not become priority. Unknown differs from irrelevant or benign. |
| Geographic/source context | Optional trusted metadata | Task/region restrictions and event associations may help; no inferred location/trust from visual appearance or sender ID. |

The user rejected the current I formula but also mentioned task importance in the
candidate global list. This note treats that concept as unresolved; it does not
silently reinstall the rejected weighted formula. A common mission outcome is
still needed to train and evaluate any ranking.

## Distinguish novelty, rarity and complementary evidence

Novelty asks whether this observation changes what the KB represents. Rarity asks
how often a concept occurs in a defined reference collection. Complementarity asks
whether another view adds a useful entity, attribute or confirmation beyond what
is already available. Different viewpoint alone is not proof of complementarity.

Examples (conceptual, not measured data):

- Same building, changed lighting, same visible task facts: visual difference but
  possibly negligible new mission evidence.
- Same building, side view revealing a blocked exit: complementary task evidence
  even though building identity is unchanged.
- A novel bird during road-damage inspection: novelty/rarity can be high while
  task relevance is low.
- A confirmed passable road during access-route planning: ordinary appearance can
  still provide useful task evidence.

Use embedding retrieval to propose comparable prior observations. Where feasible,
compare predicted entities/attributes and trusted track/site keys. Do not promote
retrieval similarity into verified natural asset correspondence. Unknown matches
must remain unknown. Retain receiver-relative overlap uncertainty explicitly;
local TX novelty cannot solve duplicate reports from different transmitters.

## Minimal candidate architecture

1. Register an operator mission paragraph plus explicit time/region constraints.
   Preserve the original text and a versioned supported-clause representation.
   Unsupported conditions remain explicit, not guessed. This setup is amortized
   over images; it need not repeat a language-model call per image.
2. Start with one pretrained image-text visual encoder. Reuse its representation
   for task similarity, KB retrieval and small separately supervised multi-label
   incident/place heads. This is a low-cost hypothesis, not a claim that CLIP can
   recognize all damage/person states correctly.
3. Add one optional shared detector for localized target evidence only if it
   improves decisions. Compare current Grounding DINO with YOLO-World under the
   actual target hardware. Their weights/backbones are separate from the global
   encoder unless explicitly redesigned; pretraining does not magically share them.
4. Compute transmitter KB changes query-before-commit and summarize supported
   evidence with masks/reliability. Do not send ground-truth annotations.
5. Send a versioned header with the whole codec payload. Compare scalar summaries
   with a compact structured/descriptor extension if the scalars lose necessary
   distinctions. Current 32-byte wire format remains baseline; do not assume a
   richer message fits 32 bytes before choosing and counting fields.
6. Receiver combines permitted header evidence with successful delivered-header
   history, derives age/deadline slack, and selects one whole job.

For capture time c, allowed age D and receiver time t: age=t-c; absolute deadline
d=c+D. If expected decoder service time is available, completion slack=d-t-E[Tdec].
These are scheduling metadata; a candidate's age is not by itself the receiver's
Age-of-Information process. D is application/operator supplied, not a guessed
output of a scene classifier. A receiver-local queue waiting limit is a different
constraint. No exponential decay or automatic discard is specified here.

The scheduler produces an ordering estimate learned against independent task
evidence outcomes. Do not construct new subjective class weights and then use
those same weights as independent evidence of semantic importance. Different
domain severity scales are not automatically comparable; initially report task
and domain results separately.

## Collection and experiment needed to decide

Use Incidents1M for broad incident/place diversity and category-specific negatives;
MEDIC for annotated informativeness/humanitarian/condition stress; retain LADI,
xBD and RescueNet for building detail/controlled overlap. Flooding can remain a
separately protected scheduler-unseen task. DoTA is a later clip-aware road check.
Do not bulk download before a small schema/image-access audit.

Within each usable source, include positives and annotated hard negatives. Do not
make all ordinary images come from a different source/camera than all incidents.
Unknown class labels remain unknown. Preserve event/parent/duplicate groups across
sources before sampling, cropping or fitting. Actual event/site grouping is needed
where source metadata allow it; absent IDs cannot be invented from filenames.

Use three separate queue regimes: mixed relevant/irrelevant scenes; all-relevant
scenes with differing entities/conditions; repeated and complementary observations.
Vary background prevalence without tuning it to make the learned policy win.
Same-image/multiple-mission comparisons are supported only where annotations
establish each mission's facts. Arbitrary paragraph correctness is not established
by binary building labels. Individual distress/urgency needs its own valid labels;
do not invent those labels from a VLM or person detector.

First compare these information sets under the same policy family and data groups:

- Existing eight-field baseline and direct I.
- Context relevance + KB evidence novelty + timestamp/deadline metadata.
- Previous set + rarity (tests whether S earns its cost).
- Previous set + overlapping incident/place/condition outputs.
- Previous set + localized scene-specific evidence.

Compare one shared scorer with scene conditioning against separate experts only
after the evidence inputs are validated. Include a simple scene/incident filtering
baseline so easy disaster-vs-background discrimination does not masquerade as
advanced semantic ranking. Include FIFO, RR, relevance-only and novelty-only.

Assess extractor labels with per-class AP/recall and false alarms, using annotation
masks. Assess evidence-change detection against supported fact correspondence.
Assess ranking with A_K/coverage@K plus misses/redundancy/failure outcomes, reporting
ordinary-background filtering separately from within-relevant ordering. Use
critical recall only for supported severity/critical labels. Report actual producer
latency/memory and header bytes; source-evidence versus reconstructed-evidence
results remain separate. Use held-out events/groups and preserve a fresh final set.

Retain a candidate only if it adds useful held-out decision value with acceptable
extraction/header cost. Stop expansion if label validity, data leakage or missing
identity/correspondence explains failure. A large classifier cannot resolve those
limits. This note supplies the research decision and next experiment, not a claim
that the proposed model or parameter set has won.

## Follow-up: policy portfolio, ground truth and model-assisted assessment

The user wants FIFO/RR, individual-parameter policies, fixed formulas, learned
utility scoring, explicit ranking objectives, contextual models and conditional
RL retained as candidate families. Ranking is an output/learning formulation:
scorers can produce rankings, and set encoders can be trained with regression or
pairwise losses. Do not count architecture versus loss changes as unrelated
algorithms or claim combinations automatically improve performance.

### What is real versus predicted?

1. Source evidence truth: published image/event/object/condition labels, with
   validity and correspondence provenance. Runtime producers predict some of
   these observations from images; numeric header scores are not evaluator truth.
2. Decision outcome: under a fixed declared mission and evidence accounting rule,
   additional annotated evidence delivered by the selected job. The rule is a
   normative benchmark choice, not a fact magically provided by the dataset.
3. Reconstruction outcome: evidence actually retained in decoded images. Native
   source facts are the reference; codec success alone does not certify retention.
   A frozen predictor/Qwen gives model-assessed retention unless independently
   validated for the applicable source/task. Natural asset identity must not be
   manufactured from image similarity or common incident category.

Example mission: expose distinct blocked-road identities. A and B show the same
blocked road r1, C shows blocked road r2, D is annotated nonblocked for this task.
If correspondence is verified and facts are preserved, both A,C,B,D and C,A,B,D
have A3=5/6; A,B,C,D has A3=2/3. There is no unique mandatory target ordering.
After A, B's normalized marginal gain is 0, C's is 1/2 and D's is 0. B versus D is
a tie for immediate binary coverage, not a negative preference example. If B
reveals a new supported obstruction attribute, the declared fact set must reflect
that. An open road may be useful under a different route-planning mission.

Let F_i be the independently annotated facts offered by source i, H the successful
delivered set and U(H) the union coverage under a declared mission. Pointwise
target is g_i(H)=U(H union {i})-U(H). Shared normalization by total offered utility
is evaluator/training-only; it cancels relative within-queue order and cannot be
silently given to runtime as extra information. Pairwise target uses sign(g_i-g_j)
with ties preserved/omitted for strict losses. Exact sequence references are
bounded to small queues and may have multiple optima; greedy references are not
called exact when multi-fact overlap invalidates that claim.

Here U is U_src for the ideal-source-evidence arm. In the post-codec arm, U_rec
counts only supported facts retained in reconstructions, using the same offered
source denominator. Unknown retention cannot be silently filled from source truth.
Primary post-codec claims require an appropriate independent retention reference;
otherwise retention is model-assessed. Runtime successful-header history is not
ground-truth knowledge of which visual facts survived.

Immediate marginal-gain MSE and pairwise losses are supervised surrogates for a
greedy decision rule; they do not generally optimize the full A_K sequence under
multi-fact overlap. Set/attention processing alone does not add planning. All must
be evaluated on the declared sequence objective and bounded exact references.

### Compatible candidates and fair comparisons

- FIFO, RR, seeded random and historical direct I.
- Each individual parameter with declared direction/missingness handling. Any
  direction/threshold selection happens on development folds, never final labels.
- Fixed formulas as transparent baselines, with equally bounded validation tuning.
- Pointwise Ridge/boosted tree/MLP, targeting additional useful evidence.
- Contextual versions with permitted successful-header history.
- Shared MLP with pairwise RankNet-style loss; compare with MSE on the same MLP.
- Listwise losses only with defensible list/soft/tied targets, not arbitrary forced
  full orders. Plain relevance nDCG can over-reward redundant images.
- Deep Sets or candidate-history attention if explicit contextual features miss
  useful relationships. Compare under matching information and tuning budgets.
- PPO only if annotation-backed exact/lookahead references show meaningful
  planning headroom. Same runtime features, valid action masks and train/test
  groups. Reward C_k/K, discount 1, horizon K matches A_K; final-only or unweighted
  delta reward is a different objective. Oracle planning cannot justify RL on a
  simple singleton-copy benchmark with no planning headroom.

Candidate hybrid mechanisms (ablate components; no asserted winner):

1. Attention/Deep Sets context encoder + shared scorer + pairwise loss.
2. Pointwise scorer + contextual reranking (shortlist only if needed; measure
   exclusion mistakes and compare to ranking the complete ready set).
3. Shared scoring core + scene-conditioned small heads, using scene information
   actually transmitted in headers; no RGB access at receiver.
4. Real deadline handling + semantic ranker + deterministic FIFO tie-break.
   This is a deadline-extension objective; do not invent deadlines or turn an
   unknown timestamp into immediate expiration.
5. Validity/nonfinite-score fallback to a declared baseline. This tests robustness,
   not an automatic semantic gain. All baselines receive identical available info.
6. Supervised contextual ranker initializing a later RL policy, conditional on the
   same planning gate and independent feedback quality.

### Qwen and Luna: useful assistance, not automatic truth

Qwen can create proposed fact descriptions with regions, unknowns and provenance;
compare retrieved source views; propose training preferences; and assess original
versus reconstructed task evidence. Freeze such annotation outputs before
comparing policy traces. Hide policy names, predicted parameters/scores and claimed
ranking quality from judges. Do not let a judge select a new convenient mission
per image. Use the same frozen evidence catalog across all policies.

First assess the judge on independently labeled examples: per-class precision,
recall/false alarms, unknown coverage, domain variation; for pairs, swap position
and include verified exact-repeat/new-attribute cases. Compare model agreement
and errors but do not treat consensus as proof. Sampling an audited small cohort
must be driven by source/task/error support, not a universal arbitrary sample
count. Teacher-generated training labels are pseudo/silver labels; an independent
native-label final set remains primary. A separate judge model reduces one form
of coupling but does not eliminate shared visual or normative errors.

No new human reviewers are available. Therefore individual distress, urgent rescue
priority and unannotated natural cross-view equivalence remain unresolved unless
valid published labels support them. Report Qwen-assessed exploratory results for
those labels, not independently verified operational success. Luna can execute
deterministic checks and bounded evidence audits with actual image access; another
model's opinion or higher effort is not an independent annotation standard.

### Validation route

Audit label construct/availability and producer correctness before training. Use
source groups/events/sites/parents/duplicates and role boundaries; protect known
producer exposure. Train/select models inside development groups; hold a fresh
final cohort. Evaluate mixed relevance, relevant-only difficulty, and controlled
multi-fact overlap separately. Keep source-ideal and post-codec results separate;
reuse the same received payload/noise across every policy.

Primary ordering outcome remains A_K/coverage@K on supported facts, with redundancy,
misses, decoder failures, actual costs and deadline outcomes where defined. Report
critical recall only where annotations justify it. Use per-group paired uncertainty,
not queue repeats as independent events. A policy can be suboptimal without having
one uniquely wrong selected message; compare oracle gaps and ties explicitly.

If headers are identical for semantically different candidates, report ambiguous
observation cases and examine header information before model escalation. Do not
replace the benchmark because FIFO/RR wins a valid comparison.

Relevant primary sources added for this follow-up:

- RankNet original method: https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/
- Set Transformer: https://proceedings.mlr.press/v97/lee19d.html
- Novelty and Diversity in Information Retrieval Evaluation (Clarke et al., 2008),
  original full-text introduction inspected: https://cormack.uwaterloo.ca/novelty.pdf
  Standard relevance-only metrics can reward redundant results; motivates an
  evidence-coverage-aware objective, not a claim our A_K is a new metric.
- When Vision-Language Models Judge Without Seeing (Zou et al., ACL 2026),
  official abstract inspected: https://aclanthology.org/2026.acl-long.703/
  Shows judge informativeness bias and motivates grounding/independent checks.
  No quantitative result is transferred to our scheduler or Qwen setup.

This follow-up records a proposed validation protocol. No new training, Qwen
inference, model judging, annotation, parameter change or experiment was run.

For the later request covering source image sequences, 10/20/50 UEs, different
mission paragraphs and contact-sheet judging, the specific proposed evaluation
protocol and reviewed literature are in
[SEQUENTIAL_BENCHMARK_PROTOCOL.md](SEQUENTIAL_BENCHMARK_PROTOCOL.md). Its dynamic
episode design extends rather than relabels the historical static benchmark.
