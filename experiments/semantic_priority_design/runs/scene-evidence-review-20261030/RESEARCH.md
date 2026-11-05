# Scene evidence for contextual decoder scheduling

30 October 2026. Bounded research finding supporting the canonical master plan;
not a replacement plan, implementation, extraction run or training result.

## Decision

Retain contextual Deep Sets scoring with RankNet as the main candidate family,
the shared MLP as a simpler comparison, Set Transformer as an interaction
comparison, and matched RL/VLM arms. First test whether the permitted scene
representation carries the distinctions demanded of those models. Do not confuse
adding model capacity with adding missing semantic information.

## Confirmed information boundary

Inspected parameters/scene_measurements.py, MEDIC_PARAMETER_HANDOFF.md and
Evaluation/stream_simulator.py. The active compact ingress exposes instances,
category count, largest-box fraction and union coverage. Category names, retained
boxes and scores exist in the provider audit, but that does not authorize their
use as runtime features. The current compact parser rejects extra fields.

These four measurements express quantity and extent, not object identities,
damage states or interactions. Matching values can describe very different
scenes. A model may learn useful correlations with other allowed fields, but
cannot universally recover scene meaning discarded by this representation.

## Inspected primary sources and design implications

| Source | Mechanism actually inspected | Application and limit |
|---|---|---|
| [Personalized Saliency in Task-Oriented Semantic Communications, 2022](https://arxiv.org/abs/2209.12274) | Scene triples, query/user-interest matching, personalized weighting of semantic triples; also channel/power allocation | Mission-conditioned importance of typed scene content is a useful precedent. Its encoder/resource-allocation performance does not establish our decoder-queue results. |
| [Transmit What You Need, 2024 preprint / 2025 journal](https://arxiv.org/abs/2412.13646) | Objects, layouts, relations and scene graphs compared for visual tasks; redundant-subgraph filtering | Match representation detail to the task; do not assume counts suffice or that full graphs are always necessary. Its task/reconstruction findings are not urgency labels. |
| [Deep Sets, 2017](https://arxiv.org/abs/1703.06114) | Shared element encoding, invariant pooling and conditioning | Supports variable object sets and ready/history sets. The pooling theorem does not guarantee that a particular compact embedding preserves all distinctions. |
| [Set Transformer, 2019](https://proceedings.mlr.press/v97/lee19d.html) | Attention over set elements; inducing-point attention to reduce scaling cost | Compare when pair interactions matter. It cannot infer absent semantic labels and its additional inference cost requires measurement. |
| [MMR, 1998](https://aclanthology.org/X98-1025/) | Query relevance combined with similarity to previously selected documents | Strong interpretable relevance/redundancy baseline. Content similarity is a proxy, not physical-event correspondence; do not call it actual receiver novelty. |
| [Approximating Min-sum Set Cover, 2003](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2003-21.pdf) | Minimize summed first-cover times of supported elements | Grounds distinct-evidence service delay. Static unit-cost approximation results do not automatically transfer to live arrivals and partial observations. |
| [Novelty and Diversity in IR Evaluation, 2008](https://cormack.uwaterloo.ca/novelty.pdf) | Information nuggets and redundancy-sensitive evaluation | Use supported fact/subtopic annotations for diversity metrics. Ordinary priority nDCG alone cannot prove additional-information benefit. |

A numerical arXiv lookup initially resolved to an unrelated physics paper. It
was discarded, and no design claim or citation relies on that lookup.

## Two information profiles, not a silent upstream change

H0 is the current frozen parameter export and mandatory baseline. Test its actual
incremental scene value with scene included, removed, shuffled and availability-
matched controls. Remove a field from derived context features as well as the
candidate vector during an ablation.

H1 is a proposed small typed-object extension: semantic category IDs from a fixed
vocabulary, normalized boxes, detector scores and explicit availability/status.
It remains within the scene-evidence family but changes the information contract.
It requires parameter-team agreement and versioned serialization before becoming
a runtime experiment. It is not implemented or approved for deployment here.

Compare H0 versus H1 with matched producer outputs, groups, teacher examples and
policy families. Information present only in audit artifacts is a privileged
representation diagnostic until a versioned H1 contract exists. Measure actual
bytes, extraction cost, truncation and detector errors. The old 35-byte claim
does not describe an extended record.

Do not start with a full scene graph/GNN. Start with typed objects and geometry.
Add relation tokens only when the producer can support them and a controlled
comparison shows their incremental value. Box overlap/proximity alone does not
establish injury, entrapment, causality or shared physical identity.

## How the model can use the evidence

Encode the permitted objects of image i with shared phi_object and invariant
pooling into z_i. Fuse z_i with SPI/relevance/transmitter novelty/masks to form
message embedding e_i. Pool current ready embeddings and processed-header
embeddings separately, preserving set size and missingness. A shared scorer
maps the candidate, ready context and processed context to a score. RankNet is
the preference loss, not the name of this architecture.

Only fixed, explicitly provided missions/registries are assumed. The teacher
sees mission paragraphs; an opaque mission ID is not arbitrary-language reasoning.
For 100 UEs, profile actual object/candidate/history counts, not just UE count.

Processed history means messages attempted, not verified reconstructed facts.
History is compatible with one-shot operation and causes no retry. Predicted
overlap may inform scoring, but it is not a hard duplicate gate. Sender novelty
and receiver overlap remain distinct. A similar-looking new condition may be
important. A novel irrelevant observation may be unimportant.

## Targets and evaluation

Keep assessed urgency/priority and supported incremental evidence as separate
targets. For independently annotated fact sets E_i and prior selected evidence H,
delta_i = F(H union E_i) - F(H) is a defensible marginal-evidence target on the
controlled cohort. Fact identities and conditions remain evaluator-only.
Unknown correspondence is not a zero-overlap target. Published damage labels do
not become urgency labels.

Train preference ranking on supported comparisons and qualified Qwen weak labels,
with label source recorded. Grade-based methods require actual graded targets.
Evaluate reference-class queue wait, first-service delay of distinct supported
evidence, misses and redundant selections, plus nDCG or pairwise agreement as
appropriate. Source-evidence selected is not independently verified decoded
meaning; reconstruction quality remains separate under the one-attempt assumption.

The task is related to min-sum coverage when messages overlap in useful facts.
That permits sequential interactions, unlike fixed per-message weighted waiting.
It does not prove RL must beat contextual greedy. First compare same-information
greedy, a short lookahead and exact tiny references; then justify RL from measured
headroom and observation sufficiency. Do not inherit a max-coverage 1-1/e bound
for our online latency objective.

## Next discriminating study

1. Freeze the actual H0 schema and source group manifest. Keep unavailable SPI
   and scene values explicit.
2. Build annotation-supported same-priority comparisons whose usefulness differs
   through mission, content or supported overlap. Include changes and repeated
   captures, not only independent random photographs.
3. Audit whether H0 separates them. Use H1 audit data only as a labeled privileged
   diagnostic until the new input contract is agreed.
4. Compare MLP, Deep Sets and a small Set Transformer on matched information and
   supervision. Run scene/no-scene and no-history/history ablations.
5. Report evidence-selection timing, priority timing, error cases and bytes/ms/
   memory/FLOPs. Escalate representation or model only for a diagnosed limitation.

## Pilot interpretation

Read pre-spi-check-20261030-094517/PILOT_AUDIT.md. Its 15 regression passes,
eight fixture learners, blocked teacher arms, finite J_time and 11 arithmetic
checks are valid bounded legacy evidence. They do not validate typed scene use,
Deep Sets, the new waiting/evidence objectives, natural cross-view understanding
or semantic superiority. No new production tests or model runs were performed
for this literature review.
