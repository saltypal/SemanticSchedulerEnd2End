# Technical specification: Disaster Decoder V3

## System boundary and status

The receiver accepts a complete digital job reference plus a serialized semantic header and ranks eligible jobs for task-conditioned service. Header extraction, policy loading, ingress/dispatch, training, evaluation and artifact verification are implemented. Real images supplied the evidence; constructed queues supplied the workload. Actual latent bytes/codec inference were not benchmarked. Legacy queue behavior and radio/transport implementation are unchanged. See the research report for measured results and the runtime guide for exact API calls.

## Wire contract (implemented)

Big-endian packed prefix `>BBHIIQ8B` plus big-endian CRC32, total 32 bytes. Fields: version byte (3), availability mask byte, uint16 profile ID, uint32 message token, uint32 KB version, uint64 capture time in ms, eight uint8 scores in `R,S,Q,I,F,P,C,N` order, uint32 CRC32 over the first 28 bytes. Available scores are rounded to 0..255; missing scores must encode as zero and must not be interpreted as measured zero. Parser validates length/version/CRC/profile/mask invariants. This is a digital header contract, not latent payload accounting.

## Parameter meanings and current extractor rules

- `R` task alignment: mapped cosine similarity `(cos+1)/2`, using supplied task relevance or compatible normalized task/image embeddings.
- `S` semantic rarity: normalized self-information of eligible concepts against available transmitter-local observation history; unavailable without supported concepts or with unavailable memory. Empty available memory uses the declared smoothed-frequency convention.
- `Q` claim confidence: `max(q,1-q)` for the profile's critical claim only when a reliable held-out temperature calibration is supplied; otherwise unavailable. This is confidence, not correctness.
- `I` task importance: profile-weighted probabilities from ordered native LADI logits, sigmoid plus descending isotonic projection. Profiles: 1001 building damage weights `(0,1/3,2/3)` over `buildings_any`, `buildings_affected_or_greater`, `buildings_minor_or_greater`; 1002 road damage `(0,1)` over `roads_any`, `roads_damage`; 1003 structural flooding `(1/3,2/3)` over `flooding_any`, `flooding_structures`.
- `F` freshness: `exp(-max(0, now-capture)/tau)`, only with trusted valid timestamps; default tau is 30,000 ms.
- `P` target evidence: largest retained target-class box area divided by image area; detector status, valid dimensions, score threshold, clipping and class-wise NMS are required.
- `C` scene complexity proxy: half normalized log count plus half normalized class entropy over retained detections in building/vehicle/person/tree/bridge classes. Not a calibrated cognitive-load measure.
- `N` local novelty: `(1 - maximum cosine similarity to compatible recent transmitter embeddings)/2`; first observation yields 1; missing/incompatible history yields unavailable.

Profiles use IDs 1001/1002/1003 for building damage, road damage, and structural flooding. Their eligible concepts, critical claims, and detection targets are defined in the Python module and are the implementation authority.

## Executed extraction

Frozen CLIP ViT-B/32 supplied alignment/novelty embeddings, LADI-small-reference supplied native task logits, and Grounding DINO-tiny supplied object boxes. All three succeeded on 1,934 unique images. Qwen3-VL-4B ran through Ollama across two T4s and produced 32 schema-valid silver proposals, not primary labels. Producer revisions and preprocessing settings are pinned in the saved source/profile. Native labels and Qwen outputs remain separate. Road Q was unavailable because B had six positive critical claims; F was unavailable without trusted capture times.

Pinned dataset revision: `5f2dbfe8c466d32edafd1bab847ec5252309acdb`; obtain validation and test only. Validation assignment: A=50%, B=20%, C=30%, grouped by source to prevent image leakage. Lock test until protocol and model choices are frozen. Controlled observation coverage is an experimental design and must not be described as verified coverage. Same exact image may recur under shared identity; differing images do not establish different physical incidents.

## Executed ranking and evaluation

CAHR used a 16-32-16-1 MLP with exact coverage-swap-weighted pairwise logistic loss. Thirteen methods were compared across three tasks with A-only shared training queues, B-only calibration, C-only model selection and a held-out official-test evaluation. Extra seeds and nine retrained ablations were building-only. All 37,440 exported-model/episode rankings were reproduced; 480 primary-run artifacts and all compressed traces passed integrity checks. Outcomes were mixed, not proof of CAHR superiority. The eight values plus eight masks are the only learned policy inputs; evaluator identities and labels remain outside the policy.

`INFRA.decoder_queue.ReceivedMessage` represents transport-established complete jobs. `DecoderQueue.enqueue` checks the CRC, token/profile binding and duplicate admission; `dispatch` excludes future arrivals, calls the immutable policy and removes the chosen opaque job. Only an available F may be updated from a trusted UTC clock. CRC is not source authentication. Artifact loading validates schema, feature order, finite weights and checksums; producer-profile agreement requires coordinated external configuration.

Expected artifacts: `results/disaster_decoder_v3/<runid>/` containing resolved protocol, source/data/model provenance, split manifest, extraction cache, checkpoints, predictions, metrics, and failure/status records. Keep latent bytes, digital header bits, and channel uses distinct.
