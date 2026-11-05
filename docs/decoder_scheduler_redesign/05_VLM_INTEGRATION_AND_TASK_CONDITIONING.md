# Qwen/VLM integration: task, parameter training and evaluation

**Date:** 22 October 2026  
**Status:** Proposed integration and pilot design. No Qwen checkpoint was downloaded or executed for this document.  
**Relationship to protocol 04:** The eight fields, receiver boundary and coverage loss remain unchanged. VLM outputs can provide producer predictions and assisted annotations; locked evaluation still needs independent reviewed facts.

## 1. Recommendation

Use a Qwen vision-language model as a **task-conditioned evidence/annotation teacher at the transmitter and in offline research**. Keep the receiver scheduler small.

There are four useful roles:

1. Turn an application-declared task into a candidate structured task profile.
2. Propose grounded relevance/evidence labels for parameter training.
3. Compare a current image against retrieved transmitter-KB observations.
4. Assist annotation and reconstruction-retention review.

These are different roles with separate artifacts. One Qwen score must not become the input feature, training label and sole benchmark truth for the same assertion.

The official Qwen3-VL report describes image/text and multi-image processing, with dense model sizes including 2B, 4B, 8B and 32B. Those capabilities motivate a pilot; they do not establish disaster-domain accuracy. [Report](https://arxiv.org/abs/2511.21631)

## 2. Placement in the system

~~~mermaid
flowchart TD
    T["Application/operator declares task"] --> TP["Validated versioned task profile"]
    T --> TC["Qwen proposes structured task interpretation"]
    TC --> TP
    X["Source image + available metadata"] --> V["Qwen evidence/relevance producer OR distilled student"]
    TP --> V
    X --> E["Shared embedding + detector + deterministic measurements"]
    KB["Prior transmitter KB"] --> RET["Top-k retrieval"]
    E --> RET
    RET --> NC["Optional Qwen comparison: repeated or added task evidence"]
    X --> NC
    TP --> NC
    V --> P["Eight scalar fields + validity/provenance"]
    E --> P
    NC --> P
    P --> H["Header + unchanged image latent"]
    H --> S["Receiver scalar-header ranker"]
    S --> D["One fixed decoder"]
    D --> O["Reconstruction"]
    O --> J["Optional blinded VLM evidence-check proposals"]
    HR["Independent human-reviewed source facts"] --> EV["Benchmark metrics"]
    J --> REVIEW["Validate/review output fact support"]
    REVIEW --> EV
~~~

Qwen reads original images at the transmitter. The receiver ranker receives only the finalized header fields and validity bits. Passing a pending latent to a general VLM is not valid image inference without a separately trained latent adapter.

Do not run Qwen on every pending receiver payload before deciding order: that would move expensive image understanding ahead of the decision and change the agreed information boundary.

## 3. Runtime task declaration

### The application declares the objective

Examples:

- Review visible building damage.
- Find evidence relevant to road obstruction.
- Locate people visible near affected infrastructure.

The application supplies the task before feature extraction and evaluation. Qwen can interpret its text into a profile; it does not choose a convenient task independently for each image.

If an image shows a beach and the model declares its task to be beach recognition, relevance becomes circular. Scene description and application objective are separate.

### Candidate profile

~~~json
{
  "task_id": "road_obstruction_review",
  "task_version": 1,
  "question": "Which images contain supported evidence relevant to road obstruction?",
  "target_concepts": ["road", "debris", "vehicle", "floodwater"],
  "evidence_rules": [
    "Describe supported observations",
    "Distinguish visible obstruction from unverified passability"
  ],
  "importance_rubric_version": "reviewed_application_rubric_v1",
  "freshness_rule_version": "registered_time_rule_v1"
}
~~~

This is a schema example, not a claim that the listed concepts prove obstruction. The application approves the task/rubric once. For runtime use, a validator maps new wording to an existing supported profile or returns unsupported. The VLM does not invent severity costs, deadlines or decay constants per image.

### One task per primary batch

Use one active, registered task per queue/batch in protocol 04. Task-dependent parameter production receives that profile. The receiver uses the matching ranker/profile.

If the task changes after headers were generated, old values may be incompatible. A building-damage relevance score cannot simply be renamed road relevance.

For the first version, change task at the next batch/session boundary. Route pending messages under their original compatible profile. Do not pretend to recompute source evidence from unavailable receiver pixels or silently introduce new header feedback/retransmission.

Later multi-task studies can train on several registered tasks using comparable score semantics. Arbitrary unseen-task generalization and simultaneous cross-task utility comparison require separate tests; eight scalars alone do not prove either.

## 4. Where Qwen helps each finalized parameter

| Parameter | VLM role | Required non-VLM support |
|---|---|---|
| Context relevance | Strong candidate: judge whether supported visible evidence answers the task | Independent relevance labels and calibration/validation |
| Scarcity | Suggest supported concepts to count | Causal KB frequency counts; rarity is not guessed by the VLM |
| Reliability | Explain assessability and unsupported claims | Correctness labels and empirical calibration; self-reported confidence is insufficient |
| Task importance | Propose evidence grade under the fixed rubric | Application-defined utility and independently reviewed grades/facts |
| Freshness | Read an explicitly supplied timestamp as data if needed | Trusted capture metadata and deterministic age; no visual age guessing |
| Spatial prominence | Propose grounded targets/boxes for audit | Coordinate validation and area calculation; no physical-distance claim from apparent size |
| Scene complexity | Propose entities/relations or a reviewed comparison | Deduplicated counts/diversity and a fixed ontology |
| Novelty | Compare retrieved prior/current evidence for useful additions | Transmitter KB retrieval, provenance and independent identity/evidence review |

VLM grounding is a candidate measurement producer, not automatically a replacement for a detector. Compare it on the same reviewed subset before adopting it.

## 5. Relevance labeling that we can actually implement

Give Qwen:

- The image.
- The declared task and concise evidence rubric.
- A fixed output schema.
- An explicit option for insufficient/unassessable evidence.

Ask for visible evidence and a relevance grade, not a free-form importance number.

### Initial relevance rubric

| Grade | Meaning |
|---:|---|
| 0 | No supported connection to the task |
| 1 | Indirect/contextual connection |
| 2 | Partially answers the task; key evidence unresolved |
| 3 | Directly answers the task with supported visible evidence |

The scope must be consistent: relevance to a question is different from whether a particular condition is true. A clear unblocked road can directly answer a passability question even if it has no obstruction. Write the task question accordingly.

### Structured prediction

~~~json
{
  "task_id": "road_obstruction_review",
  "assessable": true,
  "relevance_grade": 2,
  "visible_evidence": ["Water is visible across part of the road surface"],
  "unsupported_or_unknown": ["Whether traffic can safely pass"],
  "candidate_regions": []
}
~~~

This is illustrative output. It is not verified annotation or an inference result.

Use grade/3 as a bounded ordinal-derived relevance score only under a new registered producer profile. It is not a probability. Alternatively fit a supervised relevance probability against a clearly defined binary target, with separate calibration.

Schema validation is deterministic. Invalid JSON, unsupported grades or unassessable cases produce unavailable status or a recorded retry policy; they are not silently converted to irrelevance.

A short evidence list is enough for audit. Do not require or transmit a long reasoning transcript.

## 6. Parameter training: teacher and student

### Direct VLM producer

For the first pilot, Qwen can produce R and candidate task evidence directly at the transmitter. Cache the output once per image/task/profile. This lets us test usefulness before training another extractor.

Measure latency, token/image budget, malformed-output rate and human agreement. It is a baseline, not a presumption that a large model belongs on every deployed transmitter.

### Distilled producer

Use VLM proposals plus human-reviewed labels to train a smaller task-conditioned parameter head on shared image/task embeddings.

For ordinal relevance, a simple supervised loss is:

\[
L_R=-\sum_{c=0}^3 y_c\log p_c.
\]

The normalized expected grade can produce:

\[
R=\sum_c c\,p_c/3.
\]

This remains expected ordinal relevance, not a generic correctness probability. Reliability Q needs its own validated target.

The relevance-head loss is separate from the scheduler's coverage-weighted ranking loss. First train/freeze and validate parameter production; then construct actual headers and train the ranker.

### Label tiers

- **Silver:** unreviewed VLM-proposed labels.
- **Reviewed:** human-checked proposals, with review provenance.
- **Independent audit:** humans label without seeing the proposal.
- **Locked evaluation:** independently adjudicated test facts/grades.

Use silver labels for development or separately reported pretraining. The core reviewed protocol still uses its independent labels. If silver-only data train a ranker with the same coverage formula, label the experiment teacher-supervised and evaluate on the independent held-out cohort.

Do not merge silver labels into final ground truth without marking their source. Human review can also inherit suggestion bias; the independent blind audit measures that risk.

## 7. Transmitter novelty: retrieval followed by comparison

Preserve the existing sender-KB location:

1. Retrieve top-k prior images/evidence using the shared embedding.
2. Supply current image, retrieved observations, declared task and legitimately available time/site metadata to the VLM.
3. Ask for structured evidence correspondence.
4. Return same evidence, corroboration, additional supported fact, changed condition or unassessable.
5. Map to N only through a preregistered or fitted validated rule.
6. Update transmitter observation memory after scoring.

Do not ask an unconstrained model to declare a new disaster solely because an image looks different. A different view can reveal a new fact within a known incident. Similar-looking imagery can come from distinct incidents.

Keep cosine-only novelty as a required comparator. If retrieval misses the relevant previous image, the VLM cannot compare against it. Measure retrieval recall separately from comparison accuracy.

Limit k initially to 3 and save the candidates; test k=1 and k=5 as cost/coverage ablations. This is a proposed budget. Querying only three neighbors cannot prove absence of a fact throughout the entire KB.

Return unassessable when asset/incident correspondence is unsupported. A fluent explanation does not establish identity.

## 8. Benchmark assistance and judge separation

### Good use: reduce annotation effort

Qwen can propose visible facts, likely correspondence groups and cases worth human review. It can help organize a cohort and find disagreements.

### Required final truth

The primary benchmark still uses independently reviewed facts and weights. The primary coverage metric is computed deterministically from those labels, not returned as a free-form VLM score.

Training and testing must remain image/event/duplicate-group separated before pseudo-label generation or teacher adaptation. Freeze prompts and revisions before the locked test.

### Auxiliary automated judge

A second model family can provide an additional agreement check. It is not automatically independent truth: related training data or shared errors can remain.

Multimodal judge research has documented self/model-family preference effects for caption evaluation. That study does not measure our scheduling task, but motivates testing teacher/judge dependence here. [MLLM-as-a-Judge Exhibits Model Preference Bias, 2026](https://arxiv.org/abs/2604.11589)

If Qwen-only pseudo-facts produce a coverage score, call it **Qwen-assessed coverage**. Do not present it as independently verified semantic coverage.

## 9. Reconstruction evaluation

After the selected payload is actually decoded, a VLM can propose whether each independently reviewed source fact is supported by the reconstruction.

Give it the reconstruction, task and a specific candidate fact. Ask supported, unsupported or unassessable. Avoid revealing which scheduler/model produced the output. Randomize pair position in pairwise judge checks.

Check the candidate reconstruction independently for source-consistent evidence. Merely asking whether two images look alike can miss a small but critical lost fact.

Human-check a stratified output subset including failures, critical facts, tiny targets and VLM disagreements. Report false support, missed support and abstention rates. Validate before automating a larger retention estimate.

Do not add hallucinated facts to the source-offered utility. Do not count a judge's recognition of the expected label as proof it is visually supported.

Where the judge has not been validated, report automated agreement as an auxiliary result and retain unknown verified utility.

## 10. Measurements for the VLM integration

| Component | What to measure |
|---|---|
| Relevance | Grade confusion, macro-F1, weighted agreement, ordering agreement and abstention |
| Direct versus student | Same held-out labels, extraction latency and memory |
| Novelty retrieval | Recall of reviewed relevant prior evidence in top-k |
| Novelty comparison | Repeat/new-fact/changed-condition errors and unassessable rate |
| Claim reliability | Brier/NLL only for calibrated probabilities with independent correctness truth |
| Reconstruction judge | Fact-support precision/recall, false support and audit agreement |
| Scheduler | Protocol 04 A_10, coverage@10, high-priority fact recall and verified time curves |
| Runtime | Transmitter VLM/extractor cost separately from receiver parsing/ranking/decoding |

The first-stage scheduler comparison fixes all headers across policies. A second-stage comparison of CLIP versus VLM versus student is a producer-plus-system experiment: its effect cannot be attributed to the ranker alone.

Keep the loss fixed while comparing producers, and keep the producer fixed while comparing ranking losses.

## 11. Small pilot before adoption

Use 120 development images, group-separated from the final test. Label each under three reviewed tasks, giving up to 360 image-task cases. The same image under several tasks is not several independent images.

Suggested tasks are building-damage review, road-condition review and people-location review only where the selected images and supported labels permit them.

Two reviewers label relevance independently without seeing Qwen first. Compare:

1. Existing embedding/task similarity.
2. Direct Qwen task relevance.
3. A small student when enough independent training labels exist.

Use the same source image and task wording across producers. Measure whether a task change causes the expected relevance change, while preserving examples that are genuinely relevant to several tasks.

Add reviewed current/history cases for novelty and a reconstruction subset for judging. This is feasibility evidence, not the final multi-task journal benchmark.

Start with a pinned Qwen3-VL-8B-Instruct as an offline teacher candidate if available compute permits; compare a 2B/4B candidate for cost. These variants are documented by the Qwen team. Hardware feasibility, memory and latency on this machine are unmeasured. [Official repository](https://github.com/QwenLM/Qwen3-VL)

Do not select by size alone or assume the largest model wins our task. Freeze model revision, processor, image resizing, prompt, decoding settings and maximum output length. Greedy generation reduces sampling variation but is not a guarantee of hardware-level determinism.

## 12. Integration decision

**Adopt first:** task-conditioned relevance/evidence proposals, annotation assistance and a reviewed pilot.

**Compare next:** direct VLM versus small student; cosine-only versus retrieved-evidence VLM comparison.

**Keep unchanged:** eight header fields, complete-latent receiver scheduling, transmitter novelty location and protocol 04 coverage loss.

**Require before final claims:** independent held-out facts, explicit missingness, validated reconstruction support and measured extraction overhead.

Qwen supplies semantic hypotheses and scalable assistance. The application supplies the task, deterministic code supplies measurements, independent labels supply the benchmark, and the small ranker supplies the receiver order.
