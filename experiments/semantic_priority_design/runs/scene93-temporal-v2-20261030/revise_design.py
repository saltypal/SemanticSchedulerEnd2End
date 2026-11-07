"""Reproducible documentation revision; never edits producers, learners or results."""
from pathlib import Path
import hashlib
import json
import re

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[3]
DEST = ROOT / 'docs/decoder_context/SEMANTIC_RANKING_RESEARCH_DESIGN.md'
text = (RUN / 'MASTER.before.md').read_text(encoding='utf-8')


def replace_section(number, title, body):
    global text
    pattern = rf'(?ms)^## {number}\. .*?(?=^## \d+\. |\Z)'
    text, count = re.subn(pattern, lambda _: f'## {number}. {title}\n\n{body.strip()}\n\n', text)
    if count != 1:
        raise ValueError(f'expected exactly one section {number}; got {count}')


text = text.replace('**Protocol ID: SCHED-20261030-v1. Date: 30 October 2026.**',
                    '**Protocol ID: SCHED-20261030-v2. Date: 30 October 2026.**')
old_intro = text[text.index('This is the single executable'):text.index('## 1.')]
text = text.replace(old_intro, '''This revision modifies the existing frozen plan; it does not discard its models, objectives, reference boundaries or negative results. The user has now explicitly adopted the final scene-array v2 contract and asked for diverse streams plus a VLM-supervised temporal scheduler as the first experiment. Those decisions supersede the former H0-only primary input, optional ARRAY16 comparison and D-first execution order. D/E losses, the sixteen previous learned methods, baselines, RL/RLAIF arms and resource ceilings remain. The v1 document and the supplied final cache specification are preserved under experiments/semantic_priority_design/runs/scene93-temporal-v2-20261030/.

This is a design specification, not a claim that new policies, broad datasets, teacher reliability or real inference have been validated. Qualified labels/assets and compute still determine whether an execution is complete, partial or blocked. No production code, parameter producer, SPI model, codec or remote job is changed by this document revision.

''')

replace_section(1, 'Fixed system, revised main experiment and retained studies', r'''
Select one complete ready message for one nonpreemptive decoder. New messages arrive during service. Each message receives one attempt; poor reconstruction remains poor. No radio grants, retransmissions, retry action, preemption, encoder redesign or live VLM inference.

**First experiment P: VLM-only supervision of a parameter-only temporal ranker.** Qwen sees actual candidate images, each UE mission, a fixed receiver objective and bounded causal image histories offline. The student sees only received SPI, relevance, novelty, scene-array v2, timing and header history. Its main loss is accepted teacher pairwise preference loss; no delay target, published priority or manually weighted semantic formula is secretly added to that arm. VLM-only describes the source of scheduler supervision, not a 9B model running on every dispatch or new VLM fine-tuning.

**Retained D study:** minimize importance-sensitive waiting/lateness under the existing declared simulator budgets. Delay-trained regression, choice, classifier, neural ranking and RL comparisons remain as specified below. D is the operational cost criterion on cohorts with valid references; P is the first training experiment, not a substitute definition of delay.

**Retained E study:** minimize first-service delay of distinct independently supported facts. It is required for claims about useful novelty/redundancy. A shuffled broad pool cannot manufacture physical correspondence.

The main proposed model is a shared temporal encoder plus Deep Sets contextual scoring, trained with RankNet-style Qwen preferences. Static Deep Sets and the former summary MLP remain direct controls. All sixteen former methods are retained; the temporal model is the seventeenth architecture/registry entry. A model's D/P/E variants are label/reward experiments, not separate discoveries.

Every artifact records task_id, input_profile, architecture, target_source, reward_source, seed and source/model hashes. The broad receiver objective is fixed for the first revised experiment: multi-source visual situational awareness, giving earlier service to supported time-sensitive incidents and mission-useful additional evidence while monitoring neglect of routine messages. It replaces building-damage-only receiver purpose; the original building-damage cohort remains a separately reported restricted comparison. UE mission relevance is evidence, not permission for device identity to dictate priority. A neutral phone can show a critical incident; a disaster drone can send a routine repeat.

Ownership remains parameters/ for extraction/SPI/vocabulary/sender KB/serialization; Scheduler/ for inference/training; Evaluation/ for lifecycle/simulator/references; Codec/ paused. All experiments and new outputs stay under the canonical repository's experiments/ hierarchy.

~~~mermaid
flowchart TD
 A[Broad image manifests and real sequence sources] --> B[15 UE missions; causal offered streams]
 B --> C[Frozen parameter producers and sender KB]
 C --> D[Complete parameter record and opaque payload]
 D --> E[Receiver buffer; arrivals continue]
 E --> F[Current candidates; per-UE header windows; processed history]
 F --> G[Shared temporal encoder and set scorer]
 G --> H[Select one; one nonpreemptive attempt]
 H --> I[Processed-header log; no retry]
 I --> F
 J[Offline RGB and corresponding causal histories] --> Q[Qualified Qwen preferences]
 Q --> K[Train parameter-only student]
 K --> G
 R[Separate references and fixed D/E evaluators] --> S[Delay; ranking; evidence; compute report]
 H --> S
~~~
''')

replace_section(2, 'Final scene-array v2 input and transport boundary', r'''
### 2.1 Final primary input: SCENE93_V2

The authoritative producer contract is parameters/scene_arrays.py plus parameters/scene_vocabulary.py and the supplied final caching specification v2. This explicitly replaces ARRAY16 as the main input. H0 remains an ablation/control only where genuinely available; no invented global union coverage is reconstructed from largest rectangles.

The ordered vocabulary is complete COCO80 IDs0..79, countable extensions building/bridge/utility pole/traffic sign/animal IDs80..84, and regions road/water/debris/smoke/flame/snow/ground/grass IDs85..92. Use the exact exported registry, not a separately typed list. There are85 countable classes and8 region classes, with93 stored slots in each array. Region count slots are canonical zero with count masks clear.

For each class retain detection count where applicable and largest clipped XYXY rectangle divided by the EXIF-upright image area. Count/area masks, detector_ok and saturated preserve availability. No retained box does not prove absence or safety. Rectangular area does not establish distance, severity, condition, relationship or segmentation. Animal/specific-animal and traffic-sign/stop-sign channels can overlap; they must not be summed as unique assets. Region areas are detector hypotheses, not measured physical extent.

Frozen upstream detector: yolov8s-worldv2.pt SHA256 9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792; Ultralytics8.3.237; FP32; conf.25; class-aware NMS IoU.5; imgsz640; max_det300; no augmentation. Explicit set_classes() uses the93-class registry. CLIP setup source is a1d071733d7111c9c014f024669f959182114e33; actual text-feature/producer hashes must match. Class text encoding is one setup cost. The scheduler does not retune or replace this producer.

The fixed scene feature block is374 entries in this order:
log1p(counts)[93], largest_area[93], count_masks[93], area_masks[93], detector_ok, saturated.
Join by image/raw hash and producer identity outside inference. Features are derived by deserializing the received scene bytes, not by using higher-precision sidecars that the actor would never receive. Float arrays are an audit comparison only. A failed detector produces unavailable measurements, not an apparently empty scene.

Add four received SPI one-hot entries, one relevance and one novelty scalar, their six availability masks, and novelty-cold-start. Unknown SPI has four zero values with all four masks clear; known SPI has exactly one active value and all four masks set. Thus the base semantic student vector is374+6+6+1=387 entries. SPI confidence is not silently added when the finalized record contains only a class. Normal aliases Low only when available. New parameters require an explicit version, not dynamic field discovery.

Current wait, history time gaps and status flags are derived receiver context, not new detector outputs. Never put images, raw boxes, embeddings, payload contents, producer hashes, source labels, annotations, true delay budgets, fact identities, future arrivals/outcomes or remaining episode horizon into learned features. Full routing identity groups history and routes selection but is not a learned UE ID embedding.

### 2.2 Exact bytes and compatibility

The standalone scene block is369 bytes =2,952 bits:
2 version/status +32 vocabulary SHA256 +32 producer SHA256 +186 uint16 count slots +93 uint8 area slots +24 mask bytes.
Area rounding error is at most.5/255. Vocabulary/producer/order mismatches reject before inference. Mask padding, illegal region counts, invalid flags, length and incompatible models must reject. The current historical35-byte header is not declared integrated with this block. 35+369=404 bytes is merely arithmetic until a real composite wire envelope exists. Actual total parameters equal serialized routing/envelope plus SPI/relevance/novelty plus this block; report measured bytes and avoid adding the scene block twice when also exporting JSON. Codec payload bytes remain unknown while Codec is paused.

The374-feature block and387-entry augmented vector have different meanings from byte length; tensor float32 memory is not radio payload size. Temporal context is receiver memory, not automatically transmitted on every message. Profile/version/hash metadata belongs in artifacts and compatibility checks, never predictive features. All previous H0/ARRAY16 weights are dimensionally incompatible and require distinct artifacts/retraining.

### 2.3 Causal preprocessing and caching

The existing4,000-record CrisisMMD cache selection remains an upstream extraction workload, not evidence of broad coverage or completed93-class inference. Its supplied report states software tests/fixture notebook passed; actual expanded-vocabulary accuracy, all4,000 detections and CPU/GPU costs remain unmeasured. Reuse successful rows only after raw-image, scene-registry and producer hashes match. Do not restart extraction or mutate that notebook from this scheduler revision.

Image-only scene arrays can be cached once and reused under new compatible missions. Mission relevance and transmitter novelty cannot be reused across changed missions/source-prefix histories. Sender KB follows the producer's causal per-UE/mission contract (current experiment FIFO128): query -> serialize -> commit each offered observation; reset across incompatible sessions/splits. Receiver policy must not alter this sender stream. A broader image pool does not authorize replacing the upstream KB algorithm.

The scene producer has already applied log1p to its returned count features. Apply it exactly once. Fit missing-value imputation and standardization on training groups only. Constant standard deviation becomes one; masks/status flags remain explicit. Registered always-inapplicable region counts have zero storage/masks, not invented negatives. Freeze feature order and preprocessing per artifact; no dynamic vocabulary expansion from evaluation data.
''')

# Preserve the original independent-reference rules; extend source roles below.
text = text.replace('Dataset roles are fixed: LADI and MEDIC/CrisisMMD for compatibility/priority/task evidence; xBD controlled building overlap for E; RescueNet source transfer; FloodNet structural-flooding scheduler-task transfer, withheld from fitting and selection. An unavailable source is marked unavailable, not replaced with guessed labels. Start source access with 20 records and retain the 2GB new-source-image download cap. Prior 240-group 144/48/48 counts are a feasibility target, not confirmed assets or a power calculation.',
'''The original dataset roles remain: LADI and MEDIC/CrisisMMD for compatibility and supported task labels, xBD controlled building overlap for E, RescueNet source transfer, and FloodNet structural-flooding scheduler-task transfer withheld from fitting/selection. Broader everyday/sequence sources are added in Section11; dataset category, anomaly or caption labels do not become emergency priority. An unavailable source remains unavailable, not replaced with guessed labels. Audit20 records/source and retain the2GB new-source-image download cap. Prior240-group144/48/48 counts remain a restricted-cohort feasibility target, not a cap on a logical image manifest or confirmed new assets. Materializing or teacher-labeling a huge pool is not free.

The new main P arm uses only Qwen preferences for scheduler optimization. Native annotations serve independent controls/evaluation, and train the separately named D/E comparator arms where valid. Do not mix them into P and still call it VLM-only. Reusing images from SPI training is allowed only in development with exposure disclosed; those parent/duplicate groups cannot serve as fresh independent final scheduler evaluation. No source's missing annotation is converted to Low.

The teacher's images are privileged training information. There is no guarantee that their distinctions survive into387 parameter features. Audit exact collisions of received features AND permitted history/timing with inconsistent preferences. For repeated identical binary inputs, the empirical majority-label accuracy is an identifiable ceiling on that sample, not a proof about near-neighbor cases. Near-collision disagreement is an ambiguity diagnostic only. If rescue versus recreation or a screenshot fault is absent from all supplied signals, report an information limitation rather than claiming a smarter ranker recovers missing visual relations.''')

text = text.replace('Core task schedule: all feasible learned arms below run D; Deep Sets RankNet, PPO and DDQN also run E when assets are valid; Deep Sets RankNet-P and VisionPrefer compare weak semantic supervision. This is the fixed comparison set, not a giant task-by-model sweep.',
'''Core task schedule: first run static Deep Sets-P and temporal Deep Sets-P against the fixed untrained references on the same real teacher labels; train a summary MLP-P on those same pairs as the small distillation control. The temporal P arm is the new main candidate. The original sixteen feasible arms remain in D, and original Deep Sets/PPO/DDQN also run E where assets are valid. VisionPrefer remains a separately named weak-supervision comparison. Temporal-D is a matched mechanism comparison only on a valid D cohort; this avoids a full architecture-by-task factorial. No new urgency labels or rewards are inferred merely because the image pool is larger.''')
text = text.replace('Scalar importance cannot be read from H0 by a hand formula;',
                    'The finalized parameter record is not assigned importance by a hand formula;')

replace_section(6, 'Shared set scoring and causal temporal extension', r'''
### 6.1 Base models and shared information

The primary semantic vector x is387 entries from Section2. Append log1p(current receiver wait/d0) to current candidate tokens. Processed headers carry age-since-attempt and a history-role bit. Keep the most recent32 processed headers globally. Completion records an attempted job/header, not verified reconstructed meaning; no retry or semantic-success oracle is added.

Summary models retain the previous candidate, masked queue/history means, availability fractions, log1p sizes, empty-history flag and minimum comparable history distance with validity indicator. Only co-available standardized values enter distance; do not take distances over IDs or mix masks as image meaning. Exclude self. No comparable fields means missing comparison, not observed equality. A summary deliberately compresses legal history; its information loss is tested rather than hidden.

MLP remains64/32 ReLU, scalar output, no dropout/batchnorm. Linear remains one affine score. Static Deep Sets retains phi32/32 ReLU, separate ready and processed mean pools, candidate skip features, set sizes/empty flags, and rho64/32 ReLU. Set Transformer remains two blocks, dimension32, four heads, FF64, layernorm, dropout0 and legal masks. All models receive the same eligible messages and frozen producer; static/temporal context representations are an explicit factor.

### 6.2 Main temporal ranker

Use one shared GRU, not one trained model per UE. At decision t, construct H_i(t) from the three most recent complete receiver arrivals of candidate i's source/session before t, excluding i itself. Retain a four-record recent-arrival ring per UE so excluding a recent candidate still permits up to three records; an older candidate uses the latest three. Mark each history item as waiting/processed/in-flight using current lifecycle state. This prevents 'has arrived at receiver' from becoming 'already decoded'. Source capture order is used only when trusted and consistent; otherwise order by receiver arrival/sequence and record temporal_order_source. The same ordering rule feeds teacher and student. All history records have arrived by t; no future source frame appears.

Take the selected three records in that declared order. Project each x through shared phi into32 dimensions, append log1p(time gap/d0), lifecycle status and gap/availability indicators, then encode with one GRU hidden width32. Reset the GRU to zero for each bounded window: hidden state cannot secretly retain older images after a split/session reset. Empty history has a zero state and explicit flag. Full routing IDs are dictionary lookup keys only, never numeric/embedding inputs. Retain independent source/session rings so churn and session reuse cannot cross-contaminate history.

Let h_i=GRU(H_i), e_i=phi(x_i), q=mean_ready(e), g=mean_processed(phi(x)), and u=mean over active UE history encodings. Construct u from one canonical recent three-arrival window per active UE, not once per queued candidate; otherwise a chatty UE is over-counted. A shared rho64/32 scorer returns:
s_i=rho(e_i, x_i, h_i, q, g, u, wait_i, sizes, masks).
The active-UE pool uses received evidence from active UE sessions, not simulator-only future captures. A UE with no received evidence has an explicit empty token. Active-state joins/departures are lifecycle metadata; if a departure signal is unavailable, keep the last session until its declared timeout and log this rule.

The architecture has no fixed UE slots or fixed candidate output layer. Parameters are shared for N=4,50,60,100 and other positive counts. Set means/sizes are permutation-invariant across UEs/candidates, while history order is intentionally meaningful inside each UE. No claim of infinite-scale constant cost follows: storage is O(4N +32 +Q) headers and queue inference is linear in Q for fixed hidden width/history length, with an O(N) UE pool. Cached encodings must be invalidated when history membership, age/status inputs or model weights change. Simply reusing a state after those inputs change would be incorrect.

Use history_length=3 for the frozen first experiment; do not search16/32/128 windows. It permits a bounded actual-image teacher request. The existing32-item global processed history remains. A separately identified zero-history ablation removes temporal features and uses the static set model; a header-window mean control distinguishes temporal order from merely more records. A flat ordered-window MLP control receives the same three header records/gaps/status masks and padding, avoiding an architecture comparison against a deliberately deprived input.

### 6.3 Selection, export and causality

Deploy argmax over currently ready candidates, with ties by arrival/sequence/full identity. Busy means no dispatch. Training PPO samples only legal candidates. Empty ready queues wait for actual arrivals, never a learned WAIT. Invalid model artifacts fail before service; unexpected nonfinite scores invoke FIFO plus counter.

Export the full387-feature registry, transforms, shared phi, exact GRU equations/gate order/bias convention, pooling and rho. PyTorch and NumPy GRU formulations must agree at gate boundaries, empty windows and mixed window lengths; do not assume the existing dense exporter implements recurrence. Include activation/gate FLOPs and context rebuilding in measured cost. Pending/history state and source rings are part of resumable environment checkpoints; weights-only is not a full resume.
''')

text = text.replace('| deepsets_ranknet | Deep Sets scorer | L_pair; main D member, separate E/P label versions |',
                    '| deepsets_ranknet | Static Deep Sets scorer | L_pair; retained D model and matched P/E controls |')
text = text.replace('| visionprefer | Deep Sets scorer | L_pair from four-aspect average teacher preferences |',
'''| visionprefer | Deep Sets scorer | L_pair from four-aspect average teacher preferences |
| temporal_deepsets_ranknet | Shared GRU plus Deep Sets scorer | L_pair from qualified Qwen P labels; new main arm; matched D version only on valid references |''')
text = text.replace('LambdaMART quantization: g_i=',
'''For the first P comparison, linear-P and summary-MLP-P are the existing linear_choice/mlp_choice architectures with task_id=P and a two-candidate teacher target p_A=t, p_B=1-t. Their two-way choice cross-entropy is exactly the RankNet binary preference loss on s_A-s_B, including t=.5 ties. This preserves their choice-loss definition while matching the accepted pair labels; do not reuse D-best-tie targets in the P arm. Static/temporal RankNet-P directly use L_pair. Priority-classifier labels still require four-tier supervision; LambdaMART/ListMLE still require their supported grades/orders, not guessed targets from disconnected pairs.

LambdaMART quantization: g_i=''')
text = text.replace('All existing IDs remain recognizable;', 'All sixteen existing IDs remain recognizable;')
text = text.replace('D core comparison uses all16 feasible learned arms and the baseline set.',
                    'D core comparison retains all16 former feasible learned arms and the baseline set; the temporal-D mechanism control is additional.')

# Keep the existing reward mathematics but make the bounded teacher context and first run explicit.
text = text.replace('Teacher sees actual RGB images, declared receiver purpose, UE missions, trusted/simulated metadata clearly distinguished, and bounded causal context. Hide predicted SPI/H0 values, scheduler outputs and reference facts.',
'''Teacher sees actual RGB images, declared receiver purpose, UE mission paragraphs, trusted/simulated metadata clearly distinguished, current receiver wait and bounded causal context. Hide received SPI/relevance/novelty/scene scores, scheduler predictions and independent reference facts. Show up to two current candidate images, their H_i windows of up to three source images each, and up to two most recently processed images: at most ten unique RGB images/request. Reuse a displayed current image by reference if it also occurs in another window. Every image has neutral role/time/status labels; never concatenate them into an unmarked collage that destroys chronology. Teacher context is bounded, not a claim of complete receiver knowledge.

At deployed replay the student retains the32-header processed context and the full ready/active-UE pools, while Qwen never receives an unbounded queue. A processed image is shown as source evidence from an attempted job, not as a verified decoder reconstruction. For the first P optimizer and held-out teacher-agreement probes, construct an explicit PAIR_VIEW: queue pool q contains only the two displayed candidates, processed pool g only the two displayed processed records, and UE pool u only the displayed candidate-source histories. Build both teacher and student views from the same immutable snapshot; do not let hidden extra UEs/candidates enter those pooled inputs. This removes the false claim that merely capping g makes full contexts equal. Static/temporal/flat-history controls use the same projection; removing history is an intentional information ablation. At deployment the same shared scorer receives the full legal queue, and the expansion from paired training contexts to large queues is a generalization test, not already validated behavior. Report pair-view agreement and full-queue D/E outcomes separately. Teacher-local pairs do not certify a full-queue optimal choice.

Upstream novelty may summarize longer sender history; it remains an allowed producer feature. The offline teacher is not granted unseen future or unsent source frames, and must abstain when the displayed history cannot justify a claimed novelty distinction. The RGB-versus-parameter representation difference remains privileged distillation, even after visible context membership is matched.''')
text = text.replace('Candidate/prefix pairs use24 training and8 held-out validation groups before collection;',
'''Candidate pairs use16 training,8 validation and8 locked pilot-test context groups before collection. Prefix pairs retain24 training and8 validation groups; they are reward-model feasibility labels and supply no separate teacher test claim. Group allocation applies to source/video/incident components, not just pair-row IDs. If overlapping pair components prevent these roles, retain components and mark missing-role support rather than split the same source history across roles;''')
text = text.replace('Largest planned teacher request is preflighted before collection. At most2 current images plus bounded necessary causal context per tier call;',
'''Largest planned teacher request (up to ten unique images) is preflighted before collection at the actual registered image-token settings. The one/two/ten-image checks and reversals consume the declared control/recovery budgets, not uncapped extra inference. At most2 current images plus bounded necessary causal context per tier call;''')
text = text.replace('Round0: controls, compatible tier labels and initial candidate/prefix labels.',
'''First P execution uses the control, tier and candidate-pair portions only, with unused prefix/replay call slots reserved for the retained RL studies. It does not spend an additional216 calls per model. All static, temporal, linear/MLP P models and VisionPrefer reuse the same eligible pair/context records. There are only32 unique candidate pairs in this first budget: it is a feasibility experiment, not enough evidence for a broad journal ranking claim. A large image manifest does not multiply teacher labels. Qualified existing labels may be reused only with matching task/mission/history/schema provenance. Further collection requires a separately budgeted run.

The main P learner uses L_pair only, optimizer weight decay as specified, and lambda_D=lambda_E=0. Qwen pair output is A/B/tie/uncertain, plus concise supported evidence and named change/additional-information/mission factors. An explicit tie maps to target.5, uncertainty is skipped. Do not convert self-reported confidence or explanation length into a label weight. Every queried comparison belongs to an actual causal ready snapshot or a declared common probe; its pair label is not an assertion that one candidate is globally best among100 unseen alternatives. Train P on PAIR_VIEW contexts as defined above. Validate deployed full-queue next-choice behavior separately by D/E on supported cohorts and through qualified small-set choices where separately labeled. Do not claim pair accuracy proves globally consistent selection. D/E learners continue to train on full simulator contexts under their own independent targets.

Temporal labeling is identical except that the ordered H_i image windows are visible. There is no extra no-history teacher labeling pass in this216-call run: the32 swapped candidate pairs already consume64 calls. Student no-history, flat-history and mean-history controls reuse those same accepted labels without extra inference. Reversing A/B preserves each UE window and roles. History-order perturbation is a student diagnostic on real ordered streams; arbitrary time reversal is not automatically a physically valid labeled trajectory.

Round0: controls, compatible tier labels and initial candidate/prefix labels.''')
text = text.replace('D core comparison retains all16 former feasible learned arms and the baseline set; the temporal-D mechanism control is additional. P/teacher arms differ',
'''The first P comparison contains FIFO/RR/random, SPI-FIFO, relevance-only, novelty-only, SPI-then-novelty and predicted-budget EDF; linear-P, summary-MLP-P, static-Deep-Sets-P and temporal-Deep-Sets-P train on identical accepted pairs. The flat-history MLP and mean-history controls isolate representation. None of the rule baselines is trained on hidden reference priority. Delay-trained neural/ranking models are reported as target-source comparisons when ready, not misrepresented as P learners.

D core comparison retains all16 former feasible learned arms and the baseline set; the temporal-D mechanism control is additional. P/teacher arms differ''')

replace_section(11, 'Broad image pools, dynamic UE missions and stream construction', r'''
### 11.1 Source roles: broad manifest, selective acquisition

Use diverse established sources, not an indiscriminate web scrape. The selected core additions are COCO and Open Images for everyday mixed scenes, plus BDD100K for real temporal streams when an authorized image/clip subset is accessible. ImageNet and Places365 add object/scene breadth when existing assets or access permit. Their category labels support sampling and detector checks, not scheduler priority. Screenshots need an explicit separate UI source; a laptop detector class cannot prove an operational error.

| Source and primary reference | Added value | Use/limit |
|---|---|---|
| [COCO](https://cocodataset.org/) | People, objects, indoor/outdoor clutter, food/sports/furniture with annotations | Core everyday pool; no urgency labels or temporal order |
| [Open Images V7](https://storage.googleapis.com/openimages/web/download_v7.html) | Diverse scenes, labels/boxes and selected relationships | Core broad pool; published relationships may test a missing signal, never leak into actor |
| [ImageNet](https://www.image-net.org/download-images.php) | Wide object-category coverage | Optional sampled slice; official image access may require login; not natural UE sequences |
| [Places365](https://github.com/CSAILVision/places365) | Indoor/outdoor scene diversity | Optional scene slice; category breadth is not emergency evidence; archive access preflight required |
| [BDD100K](https://github.com/bdd100k/bdd100k) | Actual ordered driving videos with varied conditions | Core temporal source if accessible; single annotated keyframes alone are not a sequence |
| [DoTA](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly) | Traffic-anomaly temporal boundaries and clips | Targeted temporal stress; anomaly is not a calibrated four-tier priority; released frame archive is large |
| [UCF-Crime](https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/) | Normal/anomalous surveillance episodes | Optional CCTV temporal stress; coarse training video labels do not label every frame |
| [CrisisMMD](https://crisisnlp.qcri.org/crisismmd) / MEDIC | Existing disaster observations and compatibility assets | Preserve pinned4,000 selection and known exposure; not general93-class coverage |
| [xBD/xView2](https://github.com/DIUx-xView/xView2_baseline) | Annotated buildings/conditions and controlled overlap | Original independent E arm; not two independent datasets |
| [RescueNet](https://github.com/BinaLab/RescueNet-A-High-Resolution-UAV-Semantic-Segmentation-Dataset) | Aerial disaster scene/source transfer | Preserve supported label mapping and source-held-out report |
| [FloodNet](https://github.com/BinaLab/FloodNet-Supervised_v1.0) | Structural-flooding task transfer | Still withheld from scheduler fitting/selection for the original transfer claim |

These are evaluated source choices, not a claim of downloaded assets. Official ImageNet image downloads are access-controlled; several Places/BDD host requests did not resolve during this inspection, although their author repositories describe the sources. DoTA's author release describes55GB of extracted frames, exceeding this block's2GB acquisition cap. Do not download a giant archive to obtain a handful of frames or silently use an unverified mirror. Preflight schema, image bytes, licenses/terms, annotation validity and source identity for20 records before making a source active. The BDD paper/toolkit source supports the temporal choice; it does not establish that a particular local archive is available.

Keep one source manifest scalable to large pools: source/version, original identifier, content hash, parent/video/event group, official split, annotation provenance, acquisition status and license/attribution metadata. Selective caching, not the manifest length, determines compute/storage. Target an additional2,000 static parent images for feasibility (800COCO,600Open Images,400Places365,200ImageNet), plus at most120 real clips with up to8 extracted frames each if small authorized assets fit the same2GB new-source cap. Existing disaster rows are additional; intended totals are not counts of materialized independent groups. Access failure leaves the quota unmet and source unavailable; no silent redistribution or claim of completion. These caps are workload choices, not power calculations.

Source metadata and grouping remain outside learned features. Audit duplicates across all collections before splitting, including disaster copies reproduced in general-image sources. Split whole video/incident/parent components60/20/20 before selecting frames or episode construction. An official held-out source split is not casually repurposed as training. Final family/held-out-task tests cannot be used to adjust the source mixture.

### 11.2 Three named stream families

**MIXED_IID (40% of first training traces):** independently sample diverse images into queues. This directly tests the user's ordinary-photo/traffic/disaster/food/person-near-fire mix. It is a valid synthetic mixed-workload stress, but assigned ordering/location is not a real trajectory. Mark capture geography unknown or simulated; do not describe random unrelated pictures as repeated views of a place.

**CAUSAL_SEQUENCE (40%):** assign real source clips to compatible UE missions, preserve frame order, and interleave multiple UEs by receiver arrival time. Sample up to8 source frames per clip using declared source timestamps; arrival jitter never exposes a frame before its own capture/extraction/transport readiness. Repeated adjacent frames, changed appearance and changed condition are distinct cases. Only annotations/declared exact-copy construction establish shared facts or actual change; appearance difference alone does not.

**MISSION_CONTEXT (20%):** use eligible parent images under different mission paragraphs in separate grouped traces, recomputing relevance/novelty. Include high-SPI/low-relevance, low-SPI/clear incident, high-novelty irrelevant content and crowded-routine versus sparse-critical comparisons. These are challenge strata, not automatic preference labels. If the evidence does not support the contrast, Qwen may tie/abstain. Keep matched image derivatives in one split. The common receiver objective stays fixed; UE mission changes do not create fifteen unrelated definitions of global priority.

Sample sources by the frozen per-family manifest and report realized proportions. Do not label all COCO images routine or all CrisisMMD images urgent. Include source-matched positives/negatives and source-held-out results; a learned source shortcut can survive removal of source IDs. Never route all critical images to one UE type. Keep cold starts, local exact repeats, cross-UE repeats, long unserved histories and all-empty detector scenes. Exact-copy parent identity is evaluator-only. Random pool diversity and temporally meaningful novelty are separate mechanisms and separate reported strata.

### 11.3 Fifteen retained UE templates

Each template supplies device description, mission paragraph, capture-rate/trajectory metadata and availability. These are project templates informed by IMT-2030 use-case families, not fifteen standardized6G classes. Device type never grants priority.

| Device | Fixed mission intent |
|---|---|
| General smartphone | Share user-selected visual observations; no presumption of emergency |
| Citizen incident phone | Report visible incidents and changes useful to local response |
| Structural survey drone | Inspect buildings for visible damage and newly exposed structural evidence |
| Person-search drone | Locate visible people and evidence relevant to a search task |
| Flood/evacuation drone | Observe inundation and passable/blocked evacuation routes |
| Wildfire drone | Monitor visible smoke/flame and supported changes in fire conditions |
| Road/bridge drone | Inspect transport structures and visible obstruction/damage |
| Ordinary vehicle camera | Record immediate road scene and obstacles relevant to travel |
| Emergency vehicle camera | Report response access and visible incident developments |
| Traffic-intersection CCTV | Observe flow, collisions and obstructions at a stationary intersection |
| Public-space CCTV | Observe visible public-space incidents and crowd changes |
| River/underpass camera | Monitor water/route conditions from a fixed viewpoint |
| Industrial inspection robot | Report equipment/area conditions relevant to its inspection mission |
| Agricultural drone | Observe crop/field conditions and supported changes |
| Maintenance field camera | Document infrastructure inspection findings and relevant change |

The complete actual paragraphs are versioned extractor inputs, not labels generated afresh for each scheduler. All learning arms use the same paragraphs/parameters. Because the student receives relevance rather than raw mission language, this study tests the selected task-relative parameterization; it does not establish arbitrary language-instructed scheduling. If a changed mission yields identical permitted features but requires opposite behavior, record an information collision, not an invented hidden mission embedding.

### 11.4 Workload, timing and outcomes retained

Populations4/12/50/60/100 with join/pause/resume/leave; variable per-source message counts. Train on32..512 offered messages/trace and test scaling with50 images/UE (2,500/3,000/5,000 for50/60/100). One hundred active UEs need not mean100 queued messages. Population, load, source distribution and history length are independent declared axes, not one giant factorial.

Retain service d0=1 primary; lognormal(mean1,log-std.5) stress; offered loads.5/.9/1.2; initial backlog min(8,N); separate all-at-once control; uniform arrivals or two bursts containing70% of traffic in20% of the arrival window. Compute window from offered count/load and report realized backlog/rates. Hold total load fixed for population comparisons; report load-growth experiments separately. T=arrival-window end+32d0 is fixed before replay and hidden from actor.

Capture -> extraction-ready -> complete arrival -> dispatch -> finish are distinct timestamps. Use real source timestamps where known, otherwise clearly simulated times with uncertainty. Stationary cameras do not move; bounded synthetic speed caps2/2/15/20m/s for phone/robot/drone/car remain illustrative. Unrelated internet photos receive no fabricated claim of coherent physical GPS. Missing capture time/location stays unavailable; inference runtime is measured rather than silently subtracted from a deadline.

UE departure does not delete received jobs. Session resets clear the correct rings/KB and never merge identities. Every offered message is accounted as not-arrived, queued, in-flight, processed or explicitly excluded. Hidden identity-qualified outcomes/service realizations are generated once and shared across policies. All-success core;5% exception stress and1/10% sensitivities; unchanged durations/outcome-hidden policies must have unchanged selection-cost trajectories. No retry and no reconstructed-quality feedback. Only three unit-duration jobs can start by time2 at0,1,2: infeasible simultaneous budgets are recorded overload, not a fabricated training failure.
''')

text = text.replace('**D primary:** macro-average J_D across matched episodes/load strata;',
'''**P first-experiment report:** held-out qualified pair agreement, strict/tie/uncertain counts and per-family/mission/source breakdown. A teacher-labeled pilot test is silver agreement, not independently verified triage. Report D timing and E evidence where valid on the same replay; absent references mean those cells are unavailable, not a zero score. Compare temporal versus static/flat-history/mean-history under identical labels before attributing gains to temporal reasoning.

**D operational criterion retained:** macro-average J_D across matched episodes/load strata;''')
text = text.replace('**Transport:** actual parameter JSON bytes, candidate profile/vocabulary hashes and serialization latency.',
'''**Transport:** standalone369-byte scene block plus measured complete parameter/envelope bytes, candidate profile/vocabulary hashes and serialization latency. Report JSON and binary alternatives separately rather than double-counting both as transmitted. Required scene-value ablations are SPI/relevance/novelty only, those fields plus93-class arrays, and those fields plus temporal history. Class-family ablations remove corresponding values AND masks/history-derived features, not a manual channel-importance formula. Count/area/mask-only and detector-unavailable controls test shortcuts. The primary model retains all93 channels; a successful ablation does not silently redefine the final vocabulary.''')
text = text.replace('populations4/12/50/100 and three loads; message counts48/96/256/512 by population.',
                    'populations4/12/50/60/100 and three loads; message counts48/96/256/384/512 by population (15 matched traces).')
text = text.replace('Validation set:12 matched traces', 'Validation set:15 matched traces')
text = text.replace('P-only and E-specific checkpoints select their named validation objective (P accepted-pair accuracy; E cost);',
'''P-only and E-specific checkpoints select their named validation objective (P accepted-pair accuracy, ties by lower held-out pair BCE then fewer parameters; E cost);''')
text = text.replace('15. Artifact checksums, model/profile/target compatibility and rejection of legacy weights under new semantics.',
'''15. Artifact checksums, model/profile/target compatibility and rejection of legacy weights under new semantics.
16. Scene v2 exact369-byte roundtrip,93 order,374 scene features/387 semantic entries, region mask semantics, wire-derived area quantization and one-time log1p.
17. Temporal ring/window construction, lifecycle status, source/session isolation, history membership at decision time and no future images in teacher views.
18. UE/candidate permutation invariance with history order preserved; intentional history-order changes can change temporal output without introducing array-position IDs.
19. Static/flat-history/mean-history versus GRU same-context controls, NumPy GRU parity and reset/checkpoint continuation.
20. Broad-pool source/clip grouping, MIXED_IID versus CAUSAL_SEQUENCE provenance, mission-dependent cache recomputation and scene-cache reuse checks.
21. Teacher request unique-image cap, swap consistency with histories intact, pair split component isolation and final silver-test access log.
22. Exact feature/history collisions and source-predictability diagnostics; no claim that93 classes recover unencoded visual relationships.''')
text = text.replace('Execution order is dependency-driven, not a new architecture-selection stage:\n1. Freeze current source/input artifacts, preserve dirty working tree and old results.\n2. Implement D/E evaluators and typed targets beside the legacy completion objective.\n3. Implement processed-header adapter and exact observation profiles.\n4. Run acceptance/numerical checks and the fixed2–3-episode correctness run.\n5. Connect qualified reference/teacher records; blocked arms retain explicit statuses.\n6. Train registered models under fixed defaults; validate/export/resume.\n7. Run locked comparisons, scaling, ablations and report.',
'''Execution order now implements the user's VLM-first request without discarding the rest:
1. Freeze supplied scene-v2 source/artifacts, preserve dirty working tree and old results; consume upstream outputs without launching/replacing its cache job.
2. Implement the387-entry adapter, per-UE windows, same-context teacher records and broad source/trace manifests. Dataset access audits are selective and within the existing cap.
3. Implement retained D/E evaluators/typed labels beside the legacy completion objective so every trained model has a valid operational report where references exist.
4. Run meaningful scene/lifecycle/temporal/leakage/numerical/export tests and the fixed2–3-episode correctness run. A synthetic label fixture validates plumbing only.
5. Qualify Qwen on the exact ten-image request envelope, then collect bounded P preferences. Failure blocks real VLM training; do not silently substitute synthetic preferences as real labels.
6. Train the static/MLP/temporal P comparison first under identical labels; validation/export/resume and independent D/E outcomes where assessable. No SPI classifier or codec work is included.
7. Run the retained delay-trained/classifier/tree and RL/RLAIF/replay arms under their exact original losses and budget. Prefix/replay feedback stays within the same216-call allocation.
8. Run the locked scaling/ablation/transfer comparisons and report all partial, unavailable and negative results. No claim of learned superiority follows from a2–3-episode software pilot.''')
text = text.replace('Deep Sets/attention exporters require parity before deployment;',
                    'Deep Sets/attention/GRU exporters require parity before deployment;')
text = text.replace('Final specification status: model registry, targets, losses, defaults, teacher allocation and metrics are fixed.',
'''Revised specification status: SCENE93_V2 is the primary input; VLM-supervised temporal P is the first candidate; all sixteen previous models, exact D/E objectives, budgets and controls remain. The new temporal registry entry makes seventeen architecture IDs, with explicitly named target-source variants. Model registry, targets, losses, defaults, teacher allocation and metrics are fixed.''')
text = text.replace('- [YOLO-World documentation](https://docs.ultralytics.com/models/yolo-world/): custom vocabulary support, not verified detection of every hazard.',
'''- [YOLO-World documentation](https://docs.ultralytics.com/models/yolo-world/): custom vocabulary support, not verified detection of every hazard.
- [GRU / RNN Encoder-Decoder, 2014](https://aclanthology.org/D14-1179/): shared recurrent encoding of ordered observations. Its language experiment does not validate our temporal scheduler.
- [Unifying distillation and privileged information, ICLR2016](https://leon.bottou.org/publications/pdf/iclr-2016.pdf): teacher-only representations during learning versus student inputs at inference. It motivates the RGB-teacher/header-student boundary, not a guarantee that omitted visual distinctions are recoverable.
- [COCO paper](https://arxiv.org/abs/1405.0312), [Open Images description](https://storage.googleapis.com/openimages/web/factsfigures_v7.html), [Places365 author repository](https://github.com/CSAILVision/places365), [official ImageNet access](https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php), [BDD100K author repository](https://github.com/bdd100k/bdd100k), [DoTA author release](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly): source diversity and sequence/annotation roles, not independent scheduler-priority truth.
- [Qwen3.5-9B official model card](https://huggingface.co/Qwen/Qwen3.5-9B): multimodal backend and supported invocation; local task-specific reliability/cost still requires the declared controls.
- [ITU IMT-2030](https://www.itu.int/en/ITU-R/study-groups/rsg5/rwp5d/imt-2030/pages/default.aspx): broad use-case context, not an official list of these fifteen UE templates.''')

text += '''## 17. Change record and acceptance evidence for this revision

Retained from v1: one-shot complete-message nonpreemptive service; all sixteen method IDs and their losses; exact D waiting/lateness equations and E fact-delay objective; no invented real-world deadlines; qualified weak teacher labels;216 total calls; same-policy traces; seeds; train-only preprocessing; independent references; failures/misses; NumPy export/resume; paired-group evaluation; local-first execution and six aggregate GPU-hour ceiling; encoder/SPI ownership boundaries.

Changed by explicit user instruction: primary H0/optional16-class input becomes finalized93-class scene v2; P temporal distillation runs before the retained D/RL suite; broad IID pools are added as legitimate synthetic workload alongside real causal streams;50/60/100 UE scales and fifteen missions remain; bounded per-UE image histories teach the parameter-only temporal scorer. The source/vocabulary is not dismissed because some channels lack disaster relevance.

New or strengthened validity checks: exact387-entry adapter; wire-derived features; unknown detector evidence; per-UE GRU/session invariance; teacher/student history correspondence; broad-source and video grouping; source shortcuts; context-dependent cache keys; explicit silver-test pair role; exact-input ambiguity; measured context-building cost and complete parameter-byte accounting.

This revision is accepted only as a documented protocol update after source/contract inspection and focused assessment. No actual detector inference, teacher inference, learner training, codec work, dataset download or Kaggle launch is claimed. The preserved legacy pilot remains evidence only for its original completion-coverage objective. The next authorized implementation uses this revision; historical outputs retain their original protocol IDs.
'''

DEST.write_text(text, encoding='utf-8', newline='\n')
receipt = {
    'protocol': 'SCHED-20261030-v2',
    'sha256': hashlib.sha256(DEST.read_bytes()).hexdigest(),
    'words': len(text.split()),
    'lines': len(text.splitlines()),
    'primary_input': 'SCENE93_V2',
    'scene_features': 374,
    'semantic_features': 387,
    'scene_bytes': 369,
    'learned_registry_entries': 17,
    'first_experiment': 'temporal_deepsets_ranknet-P',
    'retained_studies': ['D', 'E', 'P'],
    'production_implementation_or_training_claim': False,
}
(RUN / 'freeze.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps(receipt, indent=2))
