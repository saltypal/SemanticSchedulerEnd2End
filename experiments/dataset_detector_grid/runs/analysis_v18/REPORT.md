# Detector settings: measured behavior and provisional recommendation

Source: immutable version18 compact outputs, 300 identical MEDIC images, fixed16-noun vocabulary, assigned T4 per detector. YOLO20/20 settings complete. DINO34/56 settings complete; setting34 has only204images and is excluded from complete-setting comparisons. Grid ordering means box threshold.45 and DINO cap variants are untested. No annotated reference boxes; no accuracy/recall optimum or universal detector ranking can be established.

## Operating decision

Keep YOLO-World runtime default **confidence.25, NMS IoU.5, size640, cap300**. This is a provisional engineering baseline, not an empirically established accuracy optimum. Keep DINO optional offline. For a fixed-category DINO experiment, shortlist **box.35, text.35, NMS IoU.3, shortest-side640/max-side1333, cap300**, with an explicit exact-vocabulary label filter and a count of excluded ambiguous phrases. This configuration is selected for lower phrase ambiguity and moderate suppression, not proven better object localization.

| Setting | Median extraction ms | Mean exact-vocabulary boxes/image | Images with zero exact-vocabulary detections | Noncanonical phrase boxes |
|---|---:|---:|---:|---:|
| YOLO .15/.5/640 |19.22|2.90|33.3%|0/870|
| YOLO .25/.5/640 |19.00|1.99|40.0%|0/596|
| YOLO .25/.5/960 |24.22|2.17|38.7%|0/651|
| YOLO .40/.5/640 |19.13|1.32|48.3%|0/395|
| DINO box.25/text.25/IoU.5/640 |371.43|7.89|2.7%|68/2436|
| DINO box.35/text.25/IoU.5/640 |368.69|3.47|13.7%|40/1081|
| DINO box.35/text.35/IoU.3/640 |369.64|3.52|12.7%|9/1065|

Latency scope: individual detector call including tensor processing/postprocessing and synchronization, excluding file loading/model loading. Medians include the observed run's cold first calls; not an isolated warmed pipeline or CPU benchmark. GPU assignments were separate, but CPU/host resources shared. No FLOPs/memory result follows from these timings.

## What supports the decision

- YOLO960 increased median latency27.5% versus640 and mean box count9.2%; this is not proof those additional boxes are correct. It recovered detections in four additional images net; individual gains/losses need reference annotations.
- Lowering confidence.25→.15 added274boxes, but did not make the network cheaper or establish better recall. Raising it to.40 removed201boxes and increased empty outputs. There is no accuracy evidence to move from.25.
- At the YOLO baseline, maximum observed boxes/image was26. Caps100/300/600 produced exactly identical retained detections on all300images. Cap300 is retained for compatibility/headroom; no measured accuracy or speed benefit justifies changing it. Saturation must remain reported on future crowded images.
- DINO800 took about541ms versus369ms at640 for the paper-like.35/.25/.5 baseline; exact-vocabulary mean boxes3.48 versus3.47. No evidence here justifies the extra cost.
- DINO phrase thresholds change the meaning of category names, not just the number of retained boxes. Low text thresholds can produce phrases joining multiple vocabulary nouns. These must not become invented extra scene categories. The strict shortlist reduced noncanonical phrases to9/1065 (0.85%); the.35/.25/.5 baseline had40/1081 (3.70%).
- Same-category containment pairs are sensitivity diagnostics, not verified duplicate-object errors. The strict DINO IoU.3 setting had28 such pairs, versus35 atIoU.5, with little aggregate coverage change. We cannot know whether suppression removed duplicate or distinct physical objects.
- Empty detections are not proof of an empty scene. Full-frame boxes are not automatically wrong. More categories, higher coverage and higher counts are not accuracy metrics.

## Limitations and implementation follow-up

Fixed category filtering is an analysis correction, not an already measured rerun of the production Grounding DINO adapter. Its current literal-phrase count is not equivalent to YOLO's fixed-class count. Do not silently publish the filtered tables as raw runtime outputs. Preserve raw rows and the excluded-phrase count.

The grid has no flip trials, human box review or box annotations. The earlier six-image local study found both missing objects and an unsupported DINO person region on an infographic, but those six images are not automatically the same300 and cannot supply this grid's error rate.

Keep the selected YOLO baseline during SPI work. Any final accuracy claim needs a separate annotated-box comparison; the DINO shortlist remains experimental. Complete-setting CSVs and summary.json contain all measured cases and one-factor sensitivity pairs; no incomplete setting was used to nominate a winner.
