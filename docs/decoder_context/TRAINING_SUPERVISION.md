# Where scheduler training labels come from

25 October 2026. Focused literature review and proposed supervision design.
No VLM annotation, new model training, paid inference, or parameter redesign was
executed for this note. This supplements, rather than replaces, the separate
sequence-evaluation protocol. The present question is the source of training y.

## Decision

Given no new human annotators, use offline, mission-conditioned VLM pairwise
preferences to supervise a small contextual scorer receiving permitted header
parameters only. Use annotation-derived targets where the same mission has a
computable answer. Call model-generated targets pseudo-labels. They are usable
supervision without being independently verified operational truth.

This is the best-fit first proposal under the current constraints, not a measured
winner. A full gold permutation for every queue is unnecessary. A classifier
would also require externally supplied labels; changing architecture does not
create them. The existing portable contextual regression is implemented, while
the proposed preference-label generator and neural pairwise training are not.

## The missing distinction

1. Facts: what an image contains, supplied by annotations or estimated by a VLM.
2. Mission policy: what evidence the application wants first. The operator must
   define this; image class alone does not determine it.
3. Training target: a preference or utility derived from those facts and policy.

An image labeled building-damage is not automatically labeled decode-first.
Its contribution depends on the mission, already delivered evidence and any
declared urgency constraints. A priority target must identify that context.

## Original-paper evidence

### RankNet: Learning to Rank using Gradient Descent (ICML 2005)

Inspected formulation and pairwise probabilistic objective, Sections 2–4. Query-
document relevance labels induce ordered pairs; the model learns a scalar scorer
using logistic cross-entropy on score differences. Pair supervision need not
specify a complete ranking. RankNet supplies the learning mechanism, not labels
for our image application. We use larger score = preferred, with the standard
negative-log-sigmoid sign for a preferred/dispreferred pair.

[Original paper](https://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf)

### Unifying Distillation and Privileged Information (ICLR 2016)

Inspected Sections 1–3. Teacher training can use a richer representation that is
unavailable at student deployment; teacher outputs supervise the student. This
supports the information boundary image-rich teacher versus header-only student.
The paper's standard formulation also uses labels to train the teacher; it does
not itself establish that an off-the-shelf VLM is a correct scheduling annotator.
Teacher information cannot make absent student evidence recoverable in general.

[Original paper](https://leon.bottou.org/publications/pdf/iclr-2016.pdf)

### RankGPT: Is ChatGPT Good at Search? (EMNLP 2023)

Inspected Sections 3–4 and reference training loader. ChatGPT orders 20 BM25
passages per query; 10,000 MS MARCO training queries supply inputs. Teacher
permutations generate strict pairs for a DeBERTa cross-encoder trained with
RankNet. The student still reads query/passage text. Its loader repairs omitted
IDs using original order, which we should not copy as semantic supervision.
This establishes teacher-order distillation, not our header sufficiency.

[Original paper](https://aclanthology.org/2023.emnlp-main.923.pdf),
[reference loader](https://github.com/sunnweiwei/RankGPT/blob/main/specialization.py)

### Instruction Distillation Makes LLMs Efficient Zero-shot Rankers (2023 preprint)

Inspected Sections 3.2–3.4. Teacher comparisons receive query plus two documents;
reversing their positions reduces order sensitivity. Aggregated pairwise choices
yield ranks, used for pointwise-student RankNet training. This principally changes
instructions within the same model size. Its invalid/other-output value of 0.5
must not be adopted as proof of a genuine tie in our data.

[Original paper](https://arxiv.org/pdf/2311.01555)

### VisionPrefer / VP-Score (NeurIPS 2024)

Paper title: Multimodal Large Language Models Make Text-to-Image Generative
Models Align Better. Inspected Section 3, Section 4.1 and Appendix F.2. GPT-4V
rates four images against a prompt on four quality aspects; strict preferences
derived from aggregated scores train a BLIP-based scalar reward model using
pairwise logistic loss. Ties occur in annotation examples, but an explicit
training rule for equal aggregate scores was not identified. This is direct
image-preference supervision; its criterion is generation quality, and its student
also sees images. The linked implementation repository returned 404 during review.

[Original paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/9421261e06f1a63a352b068f1ac90609-Paper-Conference.pdf)

### RLAIF vs. RLHF (2023 preprint; ICML 2024)

Inspected Sections 2.1–2.2. A task-instructed LLM compares response pairs; answer-
token probabilities yield soft preferences. Reversed presentation and averaging
mitigate position bias. A reward model learns these distributions with cross-
entropy. This supports AI label generation and distillation, separately from RL.
Token probabilities are not calibrated operational correctness probabilities.

[Original paper](https://arxiv.org/pdf/2309.00267)

### Meaningful Moments (October 2026 preprint)

Inspected full text, particularly Sections 3.1–3.4, future work and limitations.
Qwen3-VL-32B produces action-conditioned segment-importance scores and binary
pseudo-labels. The question concerns evidence needed to recognize a known action.
This is a recent visual-importance label-generation example. It tests frozen
recognizers with different sampling; learned selector training is future work.
No human temporal-importance ground truth is provided.

[Original paper](https://arxiv.org/abs/2610.16009)

### Semantic-aware Transmission Scheduling: a Monotonicity-driven DRL Approach (2023)

Inspected system model, MDP reward and critic training. Known remote-estimation
dynamics give an AoI-dependent estimation-error cost. Reward is negative summed
cost; transition samples supply bootstrapped critic targets. There is no human
master ranking. This works because application cost is specified and computable.
Their reward is not a ready-made visual-priority label for our receiver.

[Original paper](https://arxiv.org/abs/2305.13706)

## Proposed training data construction

### 1. Freeze the annotation question

Start within one mission, for example: expose distinct supported evidence about
damaged buildings. State what counts as new evidence and when urgency overrides
repetition. Do not define the label by the same weighted header formula we want
the learner to improve. Cross-mission scheduling additionally needs an explicit
service policy; UE names or independently written mission paragraphs cannot
silently establish global priority.

### 2. Construct a causal decision sample

For each sample save two currently eligible images A/B, the mission version,
decision time, relevant permitted metadata, and successful-delivery history H.
The teacher may see source images corresponding to previously completed jobs.
That is privileged, source-ideal context; it is not verified reconstructed
knowledge at the receiver. Do not substitute all sender KB entries for H.

Separately store the actual candidate parameter values/masks and the header-only
history summary available to the student. Do not pass future arrivals, future KB,
fact IDs or raw images into student features. Start with fixed causal histories;
sample more histories later within training groups if the initial labels work.

### 3. Generate the label

The teacher receives images, mission, history and evidence-bearing metadata,
without scheduler predictions or the parameter values being validated. Ask:

> Given this mission and the evidence already delivered, which candidate should
> be decoded next? Base the answer on visible task evidence. Return A, B, tie or
> uncertain; identify the supporting evidence and important unknowns.

Store concise evidence statements, not a request for hidden reasoning. Example
output, illustrative only:

```json
{
  "mission_version": "building-evidence-v1",
  "history_id": "h017",
  "candidate_a": "a",
  "candidate_b": "b",
  "preference": "B",
  "evidence": {
    "a": "Repeats the supported damage fact already in history.",
    "b": "Provides a different supported damage fact."
  },
  "unknowns": [],
  "label_origin": "vlm_teacher",
  "teacher_version": "PIN_EXACT_CHECKPOINT_AND_PROMPT"
}
```

For verified annotated facts under the same coverage mission, a program can
instead compute g(A|H) = |F_A minus F_H| and compare it to g(B|H). This produces
an exact target for that declared objective. Missing annotations remain unknown;
cross-view correspondence must actually be supplied, not inferred from event ID.
Keep target origins separate and do not mix incompatible utility definitions.

### 4. Quality control before accepting targets

Reverse candidate presentation and map the result back to image identity. For
the first strict-pair training run retain consistent A/B preferences; preserve
and count ties, uncertain answers, malformed output and disagreements separately.
An uncertain answer is not a tie. A later explicit tie objective can be added.
Look for repeated contradictions and preference cycles within the same history.
Agreement is a consistency check, not correctness proof; filtering can bias data.
Use existing relevant annotations to check factual claims where available.

Record checkpoint, prompt, input hashes, generation settings, raw response,
parsed target and exclusion reason. No expensive all-pairs/all-history campaign
is needed before this works. A proposed 200-pair pilot, including reversed calls,
is a workload cap for annotation preflight, not a statistical sample-size claim.
No teacher model size, hardware fit, inference time or cost is verified here.

### 5. Train the header-only scorer

Let p_i be candidate parameters/masks, h the permitted runtime header-history
summary and m the mission information genuinely available at runtime. For a
single-mission pilot m can be constant. A multi-task scorer cannot assume access
to arbitrary task text unless an explicit supported representation is added.

    s_A = f_theta(p_A, h, m)
    s_B = f_theta(p_B, h, m)
    q = sigmoid(s_A - s_B)
    y = 1 for A preferred; y = 0 for B preferred
    L = -y log(q) - (1-y) log(1-q)

Use the same scorer for both candidates. With A preferred, the loss is
softplus(s_B - s_A). Inference scores each ready job and selects the largest
score; pairwise training does not require an all-pairs inference tournament.
After a successful completion, update history and rescore. These are immediate
mission-preference targets, not proof of globally optimal sequence planning.

Illustrative arithmetic: s_A=0.8, s_B=0.3 implies q=0.622459. If the label prefers
B, loss=-log(1-q)=0.974077. Gradient descent pushes A down relative to B. The
target is the preference; there is no need to assert that B has importance 9/10.

## Classifier and regression alternatives

A priority classifier requires categories with a mission-specific rubric, such
as actionable evidence, useful non-urgent evidence and irrelevant content.
Annotators or a VLM must assign those categories, followed by cross-entropy
training. Object classes do not directly supply them. Contextual repetition can
change the category, and many candidates will tie within a category.

Regression requires an absolute target, supplied by computable task gain or a
well-defined teacher scale. The current annotation-derived marginal-gain MSE
route remains valid within its declared coverage task. Pairwise supervision is
recommended for broader VLM judgments because it avoids inventing absolute scores.

## Information limitation and next decision

If teacher preferences depend on details missing from all permitted inputs,
the student cannot reproduce every preference. Identical inputs with opposed
labels expose this directly; near-identical inputs warrant investigation rather
than automatically proving impossibility. Model size does not fix exact input
collisions. Check mission, causal history, label noise and parameter sufficiency
before escalating architecture. Parameter changes belong to the parallel team.

First deliverable for a later authorized build: a provenance-bearing pair-label
JSONL and parameter-only training table, with strict-pair/tie/uncertain counts,
order-consistency checks, annotation support and collision diagnostics. Fit a
linear pairwise scorer and small shared MLP to the same accepted pairs. Split by
parent/event groups before pair construction; millions of pairs from a few images
are not millions of independent examples.

This note recommends training supervision. It does not make the labeler the sole
final evaluator, claim independent operational truth, or change the wire header.
