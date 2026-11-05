# Current decision for the scheduler-policy owner — 27 October 2026

The core three remain SPI, context relevance and UE novelty, with trusted capture
metadata. Retain the names **Spatial Prominence** (largest relevant-box fraction)
and **Scene Complexity — detected entity load** (relevant instance count).
Combined relevant-box union coverage is a third supporting spatial value. None
measures physical proximity or directly determines urgency. Transmit raw count;
fit any numerical scaling inside the training fold. No KG/REBEL/depth/tracking
stage is part of this version.

Read [the finalized spatial contract](docs/SPATIAL_PARAMETER_DECISION.md) first.
The intended spatial input is `[relevant_instance_count,
largest_relevant_box_fraction, relevant_box_union_fraction]`, with three masks,
status and saturation. Combine with SPI/relevance/novelty/capture; no fixed
positive area/count urgency rule. Unsupported detector applicability is missing,
not Low priority. These are spatial-extent aggregates, not positions/depth or
visual relations. Current source still emits old all-object four-field geometry;
task filtering and the new wire extension have not been implemented. Preserve
old header parsing. Compare core-only/core+count/core+extent/core+all under
independent labels, identical queues and matched budgets. No test/model run was
performed for this decision. Detailed rationale is appended in JOURNAL.md.

## Earlier implemented contract and historical outcomes

# Parameter contract for the scheduler-policy teammate

**Owner:** BetterScheduler parameter extraction. This document describes the
current interface and evidence boundary; it does not prescribe a ranking policy.

## Runtime values

| Field | Meaning and units | Availability |
|---|---|---|
| SPI | Low/Medium/High/Critical image-visible response-attention label | The current verified fallback supports Low/High/Critical only; Medium is unsupported |
| Relevance | CLIP image/UE-mission cosine, [-1,1] | Null if text/image embedding unavailable |
| Novelty | `(1-max cosine)/2` versus this sender's comparable history, [0,1] | Null if no comparable history; first observation=1 only with cold-start flag |
| Capture timestamp | Trusted UTC milliseconds | Null unless supplied and trusted |
| Allowed delay | UE/operator configuration, milliseconds | Null if unspecified |
| Instance count | Retained YOLO-World detections for its fixed vocabulary | Null if detector failed/unavailable; zero is a valid detector output |
| Category count | Distinct predicted categories in retained detections | Same availability as detector |
| Largest-box fraction | Largest clipped rectangular box/image area | Null when no box exists |
| Union-box coverage | Union rectangular box/image area, overlap once | Zero if detector succeeded with no retained boxes |

`novelty_cold_start` is a separate boolean flag. SPI's per-class model scores,
boxes, category names and producer details are audit outputs; they are not in
the current compact 35-byte header. Scores are not calibrated probabilities.

## Information boundaries

- Parameters are created from the current image, the UE's mission and its
  causal KB. Image-derived content is available at the UE; the downstream policy
  receives parameters/header, not the image or future sender history.
- Do not feed native annotations, evaluation labels, source dataset, train/test
  split, image identifiers or routing IDs into semantic features. IDs in data
  artifacts exist for routing/provenance and must not become learned inputs.
- The offline VLM can see images and mission context for labelling/review, but
  its output is model-assisted. Labels used to evaluate importance require an
  independently supported reference at the correct image/task scope.
- If identical transmitted fields correspond to scenes whose importance differs,
  no policy can recover the missing distinction from those fields alone. The
  retained audit category names can help diagnose that gap, but they are not
  automatically part of the transmitted feature set.
- Box extent is not physical distance or urgency. Appearance novelty is not
  confirmed new real-world information. Valid numeric zero is not missing.

## Access

`FinalParameterExtractor` in `minimal_parameters.py` provides single, batch and
serial transmitter-memory operations. `parameter_cli.py` provides the executable
path. `ExtendedParameterHeader.from_bytes` parses the35-byte v2 parameter header;
`MinimalHeader` continues to parse the28-byte v1 header. The existing decoder
queue is not yet adapted to v2, and profile/model identifiers need a common
registry. The current selected-model/parameter metadata is in
`config/parameter_release_v2.json`; it does not automatically configure a
consumer. See `docs/PARAMETER_FINAL_REPORT.md` for byte layout and a command.

Current development outputs: `results/ops/spi_final_20261027b/cached_parameters_latest/`.
These300 rows use the existing three-class head, with all scene fields unavailable;
the separate retrieved YOLO results have not been merged into that cache.
Metadata fields used for provenance in
that JSONL are not semantic ranker features. The records are development-exposed.
The VLM/YOLO experiment ended at the text-control gate:9/12 correct, Medium0/3;
no new image labels or YOLO output in that run. A corrected retry also returned
Medium0/3 and stopped SPI labelling. Its independent YOLO worker processed300
images, with202 no-detection records. The existing fallback remains weak
(Critical recall2/10 on Qwen-labelled validation). Historical YOLO predictions
were replayed separately; they do not validate the new cohort.
