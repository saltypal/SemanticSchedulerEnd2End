# Semantic parameter extraction presentation

## Slide 1 — From image to scheduler decision

**Message:** The extractor converts an image into a compact semantic header. The scheduler then decides which latent should be served first.

```text
image → visual evidence → parameter extraction → semantic header → latent scheduler
```

## Slide 2 — The complete parameter contract

### Evidence layer

1. Semantic evidence
2. Visible extent of effect (optional in the first RL milestone)

### Scheduler-feature layer

3. Context relevance
4. Delivery-history novelty
5. Task impact
6. Urgency
7. Evidence sufficiency
8. Interpretation confidence

Semantic evidence is the structured description from which the decision features are derived. It is not a second copy of the SwinJSCC latent.

## Slide 3 — Semantic evidence

### What it means

Semantic evidence is a compact, structured summary of what the image depicts and how the depicted entities relate.

### What it contains

```text
scene category
detected entities
event hypothesis
spatial relations
model confidence and provenance
```

### Models

- CLIP or SigLIP: global image embedding and candidate concept matching.
- YOLO-World or Grounding DINO: open-vocabulary objects, boxes, and detector confidence.
- SAM2: masks for selected objects.
- Geometry/scene-graph module: relations between boxes and masks.
- Event head or transparent evidence rules: event hypothesis.

### Example

```text
scene = road / vehicle scene
entities = car, flame, smoke
relations = flame overlaps car; car is on road
event = vehicle_fire
```

### Scheduler representation

The full evidence is kept for audit. The scheduler receives a fixed-size summary such as an event belief, entity count, relation score, and bounded evidence vector.

## Slide 4 — Context relevance

### Meaning

How well does the image serve the current application task?

### Extraction

1. Convert the task description into a text embedding.
2. Convert the image and/or structured evidence into an embedding.
3. Calculate cosine similarity.

```text
context_relevance = cosine(task_embedding, evidence_embedding)
```

The value should be normalized and calibrated before training the scheduler.

### Example

Task: `identify visible vehicle fire or road hazard evidence`.

A burning-car image receives high relevance. An Eiffel Tower image receives low relevance for that task but could receive high relevance for a tourism task.

### Models

CLIP/SigLIP plus the application task profile.

## Slide 5 — Delivery-history novelty

### Meaning

Does this image add new information compared with evidence already delivered for the same task?

### Extraction

1. Store embeddings and event summaries for delivery-confirmed images.
2. Compare the new image with the knowledge base.
3. Use distance from the most similar delivered evidence.

```text
novelty = normalized_distance(current_evidence, nearest_delivered_evidence)
```

### Important rule

Queued or generated images must not be treated as delivered history. Novelty is updated only after delivery status is known.

### Models and systems

CLIP/SigLIP embeddings plus a vector database or indexed knowledge base.

### Scheduler effect

High novelty usually increases priority because the image may contain information not yet available to the application.

## Slide 6 — Task impact

### Meaning

If the image is delivered, how much can it change the application’s decision or consequence?

### Extraction

Use a task profile containing:

```text
event → possible decision → consequence of missing it
```

For a prototype:

```text
vehicle fire / collision = high impact
road obstruction / flooding = medium impact
normal scene = low impact
```

For a stronger system, estimate expected decision value using event belief and a consequence/loss table.

### Models

Task-impact policy, decision table, or trained impact head.

### `impact_source`

This is provenance metadata such as `demo_rule`, `task_policy`, or `trained_model`. It tells us how the impact value was produced; it is not another priority feature.

## Slide 7 — Urgency

### Meaning

How quickly does the value of this information decay if delivery is delayed?

### Extraction

Urgency combines:

```text
event type + time sensitivity + deadline + persistence + delay cost
```

Example:

```text
active fire → high urgency
historical damage documentation → low urgency
temporary road blockage → high urgency
```

### Models

Task-specific urgency policy first; trained/calibrated urgency head later.

### `urgency_source`

Examples are `demo_task_policy`, `deadline_model`, or `trained_model`. It records provenance and does not itself alter the priority.

## Slide 8 — Evidence sufficiency

### Meaning

Is there enough usable visual evidence for the application to rely on the interpretation?

It is different from importance. A highly important image can still be blurry or badly exposed.

### Extraction signals

- sharpness and blur
- exposure and saturation
- image resolution
- detector confidence
- segmentation coverage
- agreement between models
- event answerability
- occlusion and ambiguity

The notebook uses a transparent image-quality proxy based on Laplacian sharpness and exposure. It is not a calibrated probability.

### Models

Quality metrics initially; calibrated quality/answerability model later.

### `sufficiency_source`

Examples are `uncalibrated_image_quality_proxy`, `calibrated_quality_model`, or `unavailable`.

## Slide 9 — Interpretation confidence

### Meaning

How much should the scheduler trust the extractor’s conclusion?

### Extraction

Combine calibrated evidence from:

```text
detector confidence
+ event confidence
+ relation consistency
+ cross-model agreement
+ calibration quality
```

Confidence should decrease when models disagree or when the image is ambiguous.

### Models

Calibration layer such as temperature scaling, isotonic regression, or a learned confidence head.

### Scheduler effect

Low confidence can trigger defer/retry behaviour instead of treating an uncertain event as fully reliable.

## Slide 10 — Visible extent of effect

### Meaning

How much of the image, scene, or set of entities is affected?

### Extraction

Use SAM2 masks and detector geometry:

```text
affected_area_fraction = affected_mask_area / image_area
```

Other values include number of affected objects, spatial spread, and road length blocked.

### Status

This parameter is useful but excluded from the first RL milestone until validated masks are available.

## Slide 11 — How the parameters interact

The parameters are not independent scores added with hand-written weights.

```text
scene + entities + relations
            ↓
      event hypothesis
            ↓
context relevance ← task profile
novelty           ← delivered-history knowledge base
task impact       ← consequence model
urgency           ← time-sensitivity model
sufficiency       ← quality/calibration model
confidence        ← model agreement/calibration
            ↓
      scheduler state
```

The RL scheduler receives these values and learns how to prioritize latents from observed reward, delay, novelty gain, and missed-decision cost.

## Slide 12 — Final semantic header

```json
{
  "context_relevance": 0.0,
  "novelty": 0.0,
  "task_impact": 0.0,
  "urgency": 0.0,
  "evidence_sufficiency": 0.0,
  "interpretation_confidence": 0.0,
  "event_belief": 0.0
}
```

The exact values are produced by the extractor. They are not arbitrary constants such as `0.03`; they must come from embeddings, detections, calibrated models, task profiles, or delivery history.

## Slide 13 — First RL milestone

Use these scheduler inputs first:

```text
context_relevance
novelty
task_impact
urgency
evidence_sufficiency
```

Keep semantic evidence, confidence, and provenance in the audit record. Add extent and richer evidence vectors after the first controlled experiment.

## Slide 14 — Claims we can make honestly

- The prototype demonstrates the parameter contract and extraction flow.
- CLIP prompt scores are similarity evidence, not calibrated probabilities.
- Rule-based impact and urgency are transparent baselines, not learned intelligence.
- The quality-based sufficiency value is a proxy until calibrated.
- Real spatial relations require detector boxes, masks, and geometry.
- Novelty requires delivery-confirmed knowledge-base history.
