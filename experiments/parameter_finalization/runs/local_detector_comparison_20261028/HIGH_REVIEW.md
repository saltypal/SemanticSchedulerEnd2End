# Sol High validation: local detector experiment

Assessment-only review of the saved outputs, selected overlays and experiment
source. No new detector inference, model download or production edit.

## Verdict

The experiment is useful as a descriptive failure/behavior study. It establishes
that both local CPU paths completed seven cases and produced geometrically valid
outputs for those calls. It does not establish detector accuracy, a reliable count
of distinct real-world entities, semantic priority, or a universally better model.

Retain YOLO-World as a **provisional low-cost operational default** if an existing
compatible producer is needed. Do not claim it was selected for accuracy. Keep
Grounding DINO as an optional offline/open-vocabulary producer; its observed
cost and false-positive behavior do not justify adding it universally or running
it alongside YOLO. Do not silently merge either model's counts.

## Evidence reviewed

Six real cached images (3 LADI,3 MEDIC) plus one bridge horizontal flip; explicit
local vocabulary person/car/building/bridge, not the production16 nouns. Reviewed
combined results.json/REPORT.md, detector worker and summary source, manifests,
DINO dam/bridge/bridgeflip/crowd/map overlays and YOLO crowd overlay. Parent reports
19 Luna Medium checks passed; this review did not rerun that suite.

Read-only numerical checks across all14records confirmed count equals retained
box count, category count equals retained label cardinality, original-coordinate
box bounds and largest-box fraction. Union correctness is supported by the
existing geometry checks, not a newly annotated segmentation reference.

## Material findings

1. **DINO has a clear domain false positive.** The earthquake map/infographic
   (image_id c40233fc...) is assigned one large person box covering the right-hand
   geographic map, largest/union fraction0.4322. This AI visual diagnosis is a
   specific visible inconsistency, not a human-annotated benchmark statistic.
   Largest-area evidence would be misleading if interpreted as person prominence
   or emergency importance. No detections on the same map from YOLO does not
   imply the image is unimportant; a map may carry important information.

2. **Retained detections are not distinct people.** The crowd image8be286a3...
   yields DINO19person+1building. Five same-label box pairs have intersection over
   smaller area>0.9 with IoU only0.018-0.152: head/partial-body boxes can coexist
   with larger boxes after class-aware IoU0.5NMS. YOLO yields3person boxes and
   visibly misses multiple foreground people. Neither count is verified unique
   population size. Describe the field as retained detection count/entity-load
   proxy, never asserted number of people. More aggressive containment merging
   is not automatically correct because genuinely occluded people overlap.

3. **DINO adds useful localized evidence but not the condition claim.** On the
   bridge scene it detects a bridge whereas YOLO yields zero. The overlay can
   guide inspection of the visibly above-water deck. The bridge box cannot verify
   submersion, structural safety or urgency. DINO's two tiny car boxes in the dam
   scene do not validate or refute the broader dam-failure narrative.

4. **Flip sensitivity is limited evidence.** DINO bridge count changes3to4;
   largest fraction0.16012to0.16255 and union0.16012to0.16329. Extents are relatively
   stable while count changes. YOLOzero/zero consistency is not correctness.

5. **Timing must retain its scope.** DINO first-call extraction spans27.7-48.9s;
   YOLO0.44-0.56s in these saved calls. Each used a fresh process, and load time is
   separate. worker measures from after provider.load to after image opening,
   conversion and provider.measure: this includes image I/O and postprocessing,
   not only neural inference. Native processor budgets differ (DINO800/1333 vs
   YOLO640); do not label the ratio a matched-compute or warmed speedup.

6. **Small reporting/provenance corrections.** REPORT.md column 'Image hash
   prefix' is actually the image_id prefix. Rename it. Add source-file hashes,
   actual package versions and CPU/thread identity if not already captured in a
   separate receipt; source_commit alone does not identify dirty code. Record a
   post-run receipt as retrospective rather than pretending it existed at start.
   Imported old priority/evidence fields in the image manifest are historical
   sampling metadata, not detector truth; keep them explicitly evaluator-only.

## Useful contribution and next boundary

Cached mission views are valid mechanical summaries of selected detected nouns
with no extra neural forward. They illustrate relevance-dependent spatial extent,
not urgency. Union coverage avoids geometric double-counting, but cannot remove
false detections. Counts, category diversity and box fractions are optional,
producer-dependent descriptors; they are not complete scene understanding.

No additional model fleet, scene classifier, graph or threshold search is justified
by this pilot. Preserve explicit model/availability/applicability status. For maps,
screenshots and documents, ordinary-object metrics should not be represented as a
validated description of their informational content. Do not infer normality or
lowpriority from zero boxes. Grounding DINO may be useful for targeted visual
inspection or alternatives to fixed vocabulary; this pilot does not demonstrate
crop-assisted VLM gains. Keep the SPI teacher path independent.
