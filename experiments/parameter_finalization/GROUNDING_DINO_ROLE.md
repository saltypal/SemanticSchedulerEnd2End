# Grounding DINO: proposed bounded role in parameter validation

## Decision

Use Grounding DINO initially as an OFFLINE object-localization and crop-proposal
arm after the plain VLM comparison. Do not change version8, promote a priority
label, replace YOLO by assumption, or add header fields. The local checkpoint
models/priority_classifiers/grounding_dino identifies
IDEA-Research/grounding-dino-tiny; its config specifies
GroundingDinoForObjectDetection with a Swin backbone. Existing weights are
available, but their current inference accuracy has not been tested here.

Grounding DINO accepts an image and text phrases and returns boxes with matching
scores. It is not a general urgency classifier, condition verifier, monocular
depth estimator or universal relationship reasoner. Model name “grounding” does
not guarantee that an associated VLM claim is true. A matching box for bridge
cannot establish whether the bridge is submerged or structurally failing.

## Proposed flow

Image + frozen object vocabulary -> GroundingDINO -> boxes/scores/phrases
 -> retained detections -> existing rectangular measurements
 -> bounded context crops alongside ORIGINAL image -> chosen VLM
 -> same direct four-class label -> compare to original-image-only arm.

Do not feed only crops; scene context may carry the hazard. Do not supply guessed
urgency labels as detector prompts. Initial fixed vocabulary should match the
existing YOLO vocabulary for the producer comparison, not be generated from the
VLM answer or evaluator annotations. Unsupported screenshot/document content
retains the global image path; no detections is not proof of a Normal scene.

Use native HF AutoProcessor/AutoModelForZeroShotObjectDetection and local-only
weights to avoid a separate custom-CUDA repository installation. Initial box
threshold0.35 and text threshold0.25 follow the official example, but are
candidate settings, not calibrated probabilities. Record model hash, processor,
vocabulary, input dimensions, resolved thresholds, device and latency. Transform
postprocessed boxes to original EXIF-oriented image coordinates before computing
count/categories/largest-area/union using existing scene_measurements utilities.
These measurements remain detector-dependent; do not concatenate YOLO andDINO
counts (duplicates would inflate density). Compare them as alternative producers.

For the crop arm: retain at most3 score-ranked boxes after declared NMS; add10%
context on each side and clip to the image; keep original asimage1 and deterministic
crop order. No-box cases fall back to original with an explicit no-proposal flag.
Fixed boxes/crops cached once and identical across VLM candidates. Multi-image
support must be preflighted in each model adapter; it is not yet implemented by
the strict one-image direct SPI assessor.

## Small discriminating experiment, conditional on VLMselection

Use the existing24-image development panel (enriched diagnosis, not randomaccuracy):
A original image only;
B original +DINO proposals/crops;
C original +matched-count fixed spatial crops (equal image count/resolutionbudget).
Keep chosenVLM, priorityprompt, generationbudget and sourceimages constant; report
pixel/token/input differences because adding crops also increases computation.
C helps separate targeted localization from simply spending more visual tokens.
A higher label agreement or fewer unknowns is not correctness. Review visible
factual mistakes, priority stability on harmless transformations, failure cases,
per-call time/memory and native detection labels where actually available.
Current300 source labels generally lack boxes; cannot compute legitimate mAP
from image-level presence labels. Detector agreement is not independent validation.

Retain only on observed helpful error correction within cost budget. No benefit
or new hallucinations -> omit DINO from runtime; keep offline overlays if useful.
Scene YOLO baseline and35-byteheader remain unchanged until evidence warrants
an explicit producer-version change. No KG/SAM/depth/tracking added by this plan.

## Sources inspected

- Original paper: https://arxiv.org/abs/2303.05499
- Official code and example thresholds/phrase formatting:
  https://github.com/IDEA-Research/GroundingDINO
- Native HF model documentation:
  https://huggingface.co/docs/transformers/model_doc/grounding-dino

Status: research/design only; no new detector inference, crop-assisted VLM run,
GPU job or model-comparison modification performed for this note.
