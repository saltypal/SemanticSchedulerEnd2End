# Phase 0: annotation audit status

## Scope and storage

User prohibits downloading the full dataset locally. No image archive was downloaded. The raw metadata/access snapshot occupies 19,343,449 bytes (18.45 MiB). The client JAR was downloaded for access inspection only, never executed. No training or cloud job was launched for this audit.

Run: `BetterScheduler/results/phase0_trecis/20261026T134634Z`.
Reproduce: `python BetterScheduler/audit_trecis.py --run BetterScheduler/results/phase0_trecis/20261026T134634Z`.
`FETCH_MANIFEST.json` records sources and checksums; `AUDIT.json` records all downloaded annotation rows.

## Measured corpus findings

| Snapshot | Top-level rows | Unique post IDs | Unique event/post pairs |
|---|---:|---:|---:|
| 2018–2020B | 97,577 | 72,935 | 72,983 |
| 2021A–2021B | 66,586 | 63,266 | 63,280 |

The combined union contains **135,052 post IDs**. Summing the per-file event/post counts gives 136,263, so this published-sized total must not be treated as a verified globally unique post count. There are 1,209 posts assigned to multiple event IDs. Event aliases, repeated pools and genuine cross-event reuse still need separating before splitting.

At top level, 670 posts have conflicting priorities. Flattening nested judgements while excluding their merged parents and deduplicating exact leaf records yields 163,236 leaf records; 8,606 posts have more than one leaf priority. These are record-level observations, not a completed assessor-reliability analysis. Annotator/event identity consistency needs checking before treating leaves as independent votes.

Unique posts with an unambiguous top-level priority: Low 95,362; Medium 21,383; High 13,599; Critical 4,035. Three posts have only Unknown. Do not drop disagreements silently or count repeated judgements as independent images.

There are 25 observed category names in the older snapshot and 45 in the newer snapshot. The latter adds image/source/usefulness categories. These must not all be described as the original 25 information types.

The public Drive page is reachable, but its embedded inventory contains only the first 50 files. Pagination was not exhausted. This does not establish complete archive availability, image integrity or image/annotation matching. Pixels audited: zero.

## Consequences for the final design

1. SPI classes present in these files are Low, Medium, High, Critical; Unknown is unavailable supervision, not a fifth urgency level.
2. Priority is attached to a message. An image-only predictor trained from it has weak visual supervision: important information can exist in text and be invisible in the photo. Matching media and measuring this limitation are mandatory before an image-SPI performance claim.
3. Keep event/site/duplicate groups together. Random tweet splits are insufficient.
4. Retain judgement distributions and conflicts in the audit. A maximum-priority merge is a benchmark convention, not consensus.
5. Do not claim general screenshot, personal-photo or universal emergency urgency from crisis-message labels.
6. Compare categorical and ordinal prediction on identical grouped data. Use per-class errors, calibration and high-priority misses, not overall accuracy alone. No architecture has been empirically selected yet.

## Research consultation

The requested Sol High researcher inspected the official overview and user profiles. It reported that image labels attach to tweets rather than individual images, and that priority assumes an emergency-officer context. Its session hit a usage limit before delivering the final review. Its proposed ordinal visual approach remains a hypothesis; it is not a trained or validated result.

Primary sources: [official data page](https://www.dcs.gla.ac.uk/~richardm/TREC_IS/2020/data.html), [2021 overview](https://www.dcs.gla.ac.uk/~richardm/TREC_IS/2021/TRECIS_ISCRAM_2022.pdf), [2021 guidelines](https://www.dcs.gla.ac.uk/~richardm/TREC_IS/2021/2021A/TREC%202021-A%20Incident%20Streams%20Track.pdf).

## Additional completed checks

All 122 topic records are now parsed as TREC SGML (75 + 47), rather than incorrectly treating concatenated blocks as XML. The original ontology contains 25 types. There are 1,149 post IDs shared between annotation files. Of 162,176 unique event/post/assessor keys, 969 have multiple record versions and 359 have conflicting priorities. These require explicit handling before constructing soft targets. Older top-level rows have 21,118 blank texts; newer top-level rows have none. Blank text does not establish a missing image.

A conservative same-post event graph contains 80 candidate components after excluding the artificial calibration panel. This is not a verified independent-event count. The panel contains 64 unique posts, all represented elsewhere. The image-category-union proxy identifies 8,217 posts, which does not reproduce the paper's 7,297 image-annotated tweets; that cohort difference remains unresolved.

See [SPI_DESIGN_DECISION.md](SPI_DESIGN_DECISION.md) for the final architecture recommendation, label rules, loss, metrics and claim boundary.

## Outstanding audit checks

Remote media inventory and matched-image coverage; event aliases; leaf annotator identity; annotation schema/version mapping; independent image-priority supervision. These are unresolved. Full dataset audit completion is not claimed.
