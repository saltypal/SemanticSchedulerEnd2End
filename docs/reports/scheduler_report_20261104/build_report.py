from pathlib import Path
from html import escape
import re
import hashlib
import json
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from pypdf import PdfReader

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
TEXT = r'''# Semantic decoder scheduling
## Parameter development and policy design report

4 November 2026

This report consolidates the saved parameter-development and scheduler-policy records in SemanticSchedulerEnd2End. The project has defined a receiver-side scheduling problem, implemented parameter extraction and receiver simulation components, run bounded integration studies, and prepared a richer temporal ranking experiment. It has not yet demonstrated that the final proposed scheduler improves real mission outcomes.

The current design selects one complete received image message for a single nonpreemptive decoder. A small model uses numeric parameters and causal history at dispatch time. A vision-language model supplies offline training preferences from images and mission context. The decoder does not run that teacher during selection.

The strongest available evidence is specific. A synthetic stream pilot exercised eight trained learners and six references. A separate cache preflight verified 6,000 image records and their received feature representation. The later SPI comparison completed 36 development candidates before a storage failure. None of these establishes end-to-end semantic scheduling superiority.

### Reading the evidence

The report distinguishes a design decision, source implementation, recorded execution, and independent validation. A passed software test supports its tested contract. Agreement with VLM labels supports teacher imitation. Neither establishes real emergency urgency or the usefulness of a decoded image.

### Source boundary

The requested chats are Scheduler - Parameters and Scheduler - Policy Design 2. The required read_thread tool was unavailable in this session, so this report does not claim a direct review of either chat. It instead uses the canonical repository's saved specifications, journals, run reports and selected source contracts. Source identifiers appear throughout and resolve in the appendix. No new training, inference, tests or remote operations were performed for this report.

Repository HEAD at inspection was 032c23a3dafc7bfeb39a84ba55193b6b6f3a22a0. HEAD alone does not identify uncommitted documents or prior executed snapshots. Historical run identities retain their own evidence boundaries.

[[PAGE]]
# 1 Problem and system boundary

Multiple devices send images with different missions, content and capture histories. Complete messages can reach the receiver faster than one decoder can process them. Queue order then determines which evidence becomes available first and which observations wait. The project studies that receiver ordering decision. [S1, S2]

The scheduler acts only when the decoder is free. It chooses one eligible complete message and commits to one nonpreemptive attempt. Arrivals continue during service. A busy decoder does not dispatch another job. A message's poor reconstruction does not trigger a retry in the final design.

### Data and control flow

1. A sender extracts SPI, mission relevance, causal novelty and scene measurements from an image and its permitted context.
2. The sender serializes the parameter record and sends it with a complete payload.
3. The receiver queues complete arrivals and constructs candidate and history features.
4. The policy scores eligible messages and selects one identity.
5. The decoder attempts that message. The receiver updates lifecycle records and continues selection when service ends.

The scheduler receives numeric evidence and availability masks. It does not receive RGB images, latent contents, native annotations, future arrivals, actual future outcomes or evaluator fact identities. Routing identity groups messages into source sessions but is not a learned numeric priority input. [S1]

### Ownership

| Component | Responsibility |
| parameters | Extraction, SPI models, scene vocabulary, sender history and serialization |
| Scheduler | Selection algorithms, training and portable inference |
| Evaluation | Receiver lifecycle, stream simulation, reference labels and metrics |
| Codec | Image encoding and decoding, paused in the revised main experiment |

The current canonical repository is under WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End. Older records refer to SemanticSchedulerBuild or previous INFRA and SemParam paths. Those names identify historical work, not another active source of truth. [S2, S3]

The final experiment uses one synthetic service slot until a codec supplies measured service costs. Consequently, its planned delay values are simulator quantities. They are not measured radio or hardware latency. Radio allocation, retransmission, encoder redesign and preemption remain outside the scheduler's scope. [S1]

[[PAGE]]
# 2 How the parameter design developed

An early specification described eleven semantic parameter families. These included scene category, spatial relations, event hypotheses, effect extent, relevance, novelty, task impact, urgency, image quality, interpretation confidence and decision significance. It defined their meanings carefully, but explicitly described a proposed contract rather than measured extraction capability. [S4]

The later practical path concentrated on SPI, mission relevance, sender novelty and capture metadata, with scene evidence added through versioned contracts. The final policy specification adopts the full scene-array v2 cache as its information reference. These revisions should not be presented as one unchanged header or as eleven operationally validated outputs. [S1, S5]

| Parameter | Meaning and limitation |
| SPI | Four labels, Low, Medium, High and Critical, for depicted intervention need under a declared rubric. It does not verify a current live emergency. |
| Mission relevance | Image and supplied mission alignment. A cosine similarity is not a probability of usefulness. |
| Sender novelty | Appearance difference from permitted sender history. It does not prove that the receiver has already learned the same fact. |
| Capture and timing metadata | Trusted timestamps and supplied timing requirements. Pixels alone do not establish currentness. |
| Scene arrays | Per-class retained counts and largest rectangle area, with masks and detector status. These do not establish severity or relationships. |

SPI's intended image-only rubric separates no supported intervention, concrete nonurgent corrective action, substantial consequence requiring prompt attention, and direct evidence of immediate serious harm or critical failure. Ambiguous, unreadable and unsupported inputs have an assessment status. They must not silently become Low. Mission-conditioned SPI remains a separate target change rather than a hidden change to the image-only classifier. [S5]

### Availability and history

Missing values need explicit masks. An unavailable detector differs from an operational detector that retained no objects. Unknown SPI differs from Low. Cold-start novelty differs from a measured low novelty value. These distinctions let the policy learn from actual evidence without treating failed extraction as reassurance.

The final planned sender history follows query, serialize, then commit for each offered observation, with a causal per-source and per-mission FIFO of 128 entries. Receiver scheduling must not change that offered sender stream. The older eleven-family document instead discussed successfully delivered history for its novelty proposal. The report preserves this semantic change rather than conflating sender novelty with receiver knowledge. [S1, S4]

[[PAGE]]
# 3 Scene arrays and communication cost

The scene-array v2 contract contains 93 ordered classes. It combines the full COCO80 vocabulary, five additional countable categories and eight region categories. There are 85 countable classes. Region count slots remain zero with their count masks clear. The source registry defines the authoritative order. [S1, S6]

For each class, the contract stores a retained count where applicable and the largest clipped rectangle area divided by the oriented image area. Separate count and area masks preserve availability. The detector_ok and saturated fields describe producer state. A retained rectangle is geometric evidence, not segmentation, physical extent or verified damage.

### Feature dimensions

| Block | Entries |
| log1p transformed counts | 93 |
| Largest normalized rectangle areas | 93 |
| Count and area availability masks | 186 |
| Detector status and saturation | 2 |
| Total scene features | 374 |
| SPI one-hot, relevance and novelty | 6 |
| Their availability masks and novelty cold start | 7 |
| Total base semantic student vector | 387 |

Receiver wait, history age and lifecycle indicators enter as additional context. The student must use values reconstructed from received bytes. Higher-precision audit sidecars cannot supply information unavailable at runtime. Apply log1p to counts exactly once. [S1]

### Wire bytes differ from feature dimensions

The self-describing scene block is 369 bytes. It contains two version/status bytes, two 32-byte hashes, 186 count bytes, 93 area bytes and 24 mask bytes. Area quantization has a maximum rounding error of 0.5/255. [S1, S6]

The existing compact core is 28 bytes. Its historical extended header is 35 bytes with seven summary-scene bytes. The 93-class scene block is a different contract. Adding 35 and 369 gives 404 bytes arithmetically, but does not establish an implemented composite message. Total transmission must include the actual envelope, routing, scalar parameters and payload. Float32 tensor memory is not radio payload size. [S1, S7]

The verified mini-cache report calculates a possible sparse scene cost of 34.4513 bytes on average, 45 at the 95th percentile and 73 at maximum under a preconfigured profile. Those are cache-based estimates. A compact serializer, profile provisioning and total on-wire cost remain unverified. This supports a lossless sparse transport experiment before discarding fields. [S8]

[[PAGE]]
# 4 SPI training and observed results

SPI development exposed failures in both labeling and student generalization. The records preserve parser failures, weak class recall and later incomplete runs. The current evidence supports a development-stage classifier study, not a qualified operational urgency model. [S5, S9]

An early teacher run generated ten responses but failed strict parsing because the required availability field was absent. No usable labels or student model resulted. Later work repaired execution and compared students against VLM pseudo-labels. The distinction matters because a transport or parser repair does not validate the teacher's judgments. [S10]

### Earlier bounded comparisons

On the same 96 exposed development examples, frozen CLIP with a linear head recorded macro-F1 0.463708. EfficientNet-B1 recorded 0.446224. The CNN's accuracy was higher, but Medium and Critical recall fell, so it was rejected under the declared class-recall gate. A separate frozen SigLIP study recorded macro-F1 0.508697 and still failed readiness for Medium and High. These results used small, development-exposed pseudo-labeled cohorts. They do not establish a universal encoder ranking. [S9, S11]

### Later 4000 image comparison

The recorded version 11 campaign selected 4,000 CrisisMMD images with 3,933 usable mixed-protocol VLM labels. It preserved earlier group and split assignments. The run completed 12 original-setting candidates each for CLIP, SigLIP2 and DINO, for 36 completed development candidates. ConvNeXtV2 completed none of its 12 candidates. [S9, S12]

The provisional leader among completed candidates was CLIP B32 with single-query cross-attention, with development macro-F1 0.6336517065. It is a partial-study development leader. The study did not complete all encoders, balancing matrices, final selection, test evaluation or full-pipeline profiling.

The run ended with a no-space-left-on-device error during ConvNeXtV2 extraction after about 5,803 seconds. Later notebook serialization errors followed storage exhaustion. The failure does not erase completed fits, but saved metrics alone do not prove every corresponding deployable checkpoint was retrieved. [S9]

A separate provisional mini-cache classifier recorded test macro-F1 of about 0.600 against pseudo-labels. Its native test groups had already been evaluated in that study. They remain excluded from fitting and selection where specified, but cannot be advertised as a fresh independent final scheduler test. Scores across these different cohorts and regimes should not be interpreted as one controlled improvement curve. [S9, S12]

[[PAGE]]
# 5 Final scheduler policy design

The locked policy specification is SCHED-20261031-v2.2. Its first experiment trains a parameter-only temporal ranker from offline Qwen preferences. It retains delay-supervised and independently supported fact-delay studies as separate comparisons. This specification records intended behavior and acceptance rules, not completed training. [S1]

### Main temporal model

At each decision, a shared image-header projection maps candidate features into 32 dimensions. A shared GRU with hidden width 32 encodes the three most recent permitted complete receiver arrivals from that candidate's source session, excluding the candidate. A four-record ring allows that exclusion while retaining three prior records.

History records retain their waiting, processed or in-flight state. Arrival does not mean decoded. The ordering uses trusted capture order only when consistent, otherwise receiver arrival and sequence. The GRU resets for every bounded window, so it cannot retain undeclared older information.

The score combines the candidate, its temporal state, a ready-queue mean pool, a bounded processed-header mean pool, an active-device history pool, current wait, sizes and masks. A shared 64/32 scoring network returns one score per candidate. Set pooling removes dependence on arbitrary candidate ordering while the GRU preserves meaningful order within a source history. [S1]

The model has no fixed output slot per device. Shared parameters support different device populations. Compute still grows with the queue and active population. The planned bounded storage is proportional to four records per source, 32 processed headers and the queue. Cached encodings must be refreshed when their history membership, lifecycle inputs or weights change.

### Offline supervision and deployment

The initial teacher candidate is Qwen2.5-VL-7B-Instruct with NF4 loading. It sees actual images, missions, the fixed receiver objective and bounded causal image histories offline. Accepted pairwise preferences train the student. A qualified teacher and usable contextual label cache are prerequisites, not established results.

For candidate scores sA and sB, pair probability is sigmoid of sA minus sB. Pairwise cross-entropy trains strict preferences and supported ties. Uncertain comparisons are skipped under the protocol. An image-rich teacher can distinguish cases that the 387 numeric features cannot. Identical received observations with conflicting teacher preferences therefore require an information-loss audit. [S1]

Deployment chooses the highest-scoring eligible message with deterministic identity-based ties. Empty queues wait for arrivals. Nonfinite scores invoke FIFO and a counter. Invalid or incompatible artifacts must fail before service. Export validation must compare the full recurrent computation across implementations, including gates, masks and empty histories.

[[PAGE]]
# 6 Objectives and comparison methods

The final plan separates three studies so that preference imitation, waiting cost and distinct evidence do not become interchangeable claims. [S1]

### P study of teacher preferences

P uses accepted image-and-context teacher preferences as its scheduler supervision. It measures agreement with that qualified teacher and downstream behavior where independent references exist. It must not silently add manual delay targets or native priority labels to its training loss and still claim VLM-only supervision.

### D study of waiting and lateness

Let a be complete arrival, b dispatch and W the wait b minus a. The declared acceptable-deferral budgets are 2, 4, 8 and 16 synthetic service slots for Critical, High, Medium and Low reference tiers. These are experimental budgets, not verified real harm thresholds.

The cost of a wait is W/delta plus max of zero and W/delta minus one. The second term increases the marginal cost after the budget. Accrued cost freezes at dispatch and remains in the total. Unselected jobs continue to accrue cost until the evaluation horizon. Removing them from the denominator would reward starvation.

The normalized primary score is total accrued cost divided by the number of assessable offered jobs on matched complete-reference episodes. The supervised one-slot deferral target is the difference between cost at W plus one service slot and cost at W. It is a local deferral cost, not a globally optimal action value. RL uses negative increments of total cost with fixed scaling, including service arrivals and the terminal interval. [S1]

### E study of distinct evidence

E measures delay until first service of independently supported fact-condition units. Repeated views of one supported fact count once. Changed conditions require distinct reference units. Controlled annotated crops can establish overlap; visual similarity alone cannot establish cross-camera identity. Source-reference coverage does not certify that a decoder reconstructed the meaning. [S1]

### Controls and learned comparisons

FIFO, round robin, random and same-information rules provide operational controls. The locked learned registry has 17 entries, including linear and MLP choice models, delay regression, RankNet, LambdaMART, a priority classifier, static Deep Sets, a Set Transformer, ListMLE, masked PPO, Double DQN, prioritized replay, PPO warm start, three VLM-related arms and the temporal ranker.

The first useful comparison is bounded. Match linear, summary MLP, static Deep Sets and the temporal model on accepted P labels. Add a mean-window control and an ordered-window MLP to test whether gains come from temporal order, additional history or architecture. Retaining the full registry does not mean all 17 methods have been trained under the final protocol.

[[PAGE]]
# 7 Work executed and its limits

### Synthetic streaming pilot

The corrected run stream-pilot-20261029-220835 used CPU execution, three evaluation episodes with device pools of 4, 60 and 100, and message counts of 32, 48 and 64. It trained supervised models for three epochs and supplied 64 decisions per RL arm. This was an integration pilot, not convergence training. [S3]

Eight fixture-trained learners and six references produced 42 policy-episode traces and 742 decision-log rows. Portable models and locally generated checkpoints reloaded on CPU. The recorded focused checks totaled 15 passed cases. LambdaMART export checks covered 100 trees and 744 saved-example predictions, with maximum absolute discrepancy 2.665e-15. These are recorded historical checks, not reruns for this report.

The earlier stream-pilot-20261029-220639 run is invalidated because its fixture violated detector cross-field constraints and source changed during execution. It remains preserved and must not contribute comparative scores. [S3]

The successful pilot used synthetic parameters, constructed world references and legacy completion-based coverage. It did not run real cameras, final temporal learning, qualified Qwen preferences, radio transmission or image reconstruction. Its successful-history behavior also predates the final processed-header design. It cannot validate the later dispatch-wait objective merely because both concern scheduling.

### Verified mini cache ingress

The mini-wsl-ingress-20261030-v2 report records checks of 6,000 image files and all raw hashes, with 5,952 unique images. The cache contains 1,000 Open Images, 1,000 Places365 and 4,000 CrisisMMD records. All scene records passed the 93-class contract and received-wire parity checks. The exported feature matrix has 6,000 rows and 387 features. [S8]

No exact-hash or group split crossings were found by that preflight. Near duplicates remain unaudited. Moreover, 2,069 observations share raw hashes with SPI-head training records. The whole pool is development material.

The cache has no independently available urgency references. Mission relevance and causal novelty are unavailable, with both values and masks clear. Image-level SPI predictions are not contextual scheduling preferences. The preflight therefore verifies ingestible data and feature consistency, not readiness to train the final P or D scheduler.

The recorded WSL environment had NumPy but lacked torch and scikit-learn for this work. A qualified training runtime, stream-conditioned parameter completion and contextual preference records remained outstanding in that receipt. No claim about their current live state follows without another check.

[[PAGE]]
# 8 Assessment and next complete increment

The project has a precise scheduling boundary, versioned parameter contracts, reusable learning implementations and bounded execution evidence. The remaining problem is an evidence and integration gap between the final design and the tested system. Adding another model before closing that gap would not establish a better scheduler.

| Status | Defensible statement |
| Defined | Final temporal P study and retained D/E objectives are specified in protocol v2.2. |
| Implemented contracts | Compact headers and 93-class scene serialization exist in source. |
| Recorded execution | A legacy synthetic stream pilot and real cache ingress preflight completed. |
| Partial results | The later SPI study completed 36 development candidates before storage exhaustion. |
| Not established | Final temporal training, independent urgency accuracy, real useful-evidence gains and total end-to-end latency. |

### Recommended next increment

First freeze a small development stream from verified cache rows, with explicit mission, source session, arrival order and reset boundaries. Populate relevance and sender novelty using their actual causal producer. Preserve unavailable values where computation is unsupported.

Then validate a bounded image-rich teacher panel and persist accepted context-specific preference pairs with model, prompt and history hashes. Keep abstentions and disagreements. Audit received-feature collisions before fitting the student, because missing visual distinctions cannot be recovered by training alone.

Train the matched linear, MLP, static set and temporal controls on the same accepted labels and group boundaries. Validate recurrent export parity and representative lifecycle behavior before a larger sweep. Use a fresh independent cohort for confirmatory claims. Existing exposed split names do not create one.

Implement and verify the exact final D evaluator separately from the legacy completion-coverage evaluator. Use complete-reference episodes for total D cost, and report assessed coverage for partial-reference streams. Report P agreement, D waiting cost and E fact-delay results separately. [S1, S3, S8]

### Acceptance evidence

The next result should include immutable source and producer identity, serialized input parity, group-isolation checks, causal-history checks, held-out preference agreement, per-tier waiting and miss distributions where references exist, neglected-message accounting, measured selection cost and checkpoint recovery evidence. It should state which codec and transport costs are still absent.

The first claim worth testing is whether bounded temporal context improves preference agreement or independently referenced scheduling outcomes over matched static models. Until that experiment runs, temporal benefit is a hypothesis. The existing records justify proceeding to that experiment, not declaring it successful.

[[PAGE]]
# Appendix Sources and provenance

All paths below are relative to the canonical repository. Source files were read on 4 November 2026. Historical metrics come from their saved records. No external literature was newly reviewed for this report, and no novelty or state-of-the-art claim is made.

S1. docs/decoder_context/SEMANTIC_RANKING_RESEARCH_DESIGN.md. Locked protocol SCHED-20261031-v2.2. Main source for final objectives, 93-class inputs, temporal design, method registry and teacher boundary.

S2. README.md. Canonical four-component ownership and receiver flow. Some status text describes earlier stages and must be read alongside dated later records.

S3. docs/decoder_context/STREAM_BUILD_STATUS.md. Corrected synthetic pilot, invalidated run, focused checks, export parity and implementation limitations.

S4. docs/FINAL_PARAMETER_SPECIFICATION.md. Eleven-family proposal dated 28 September 2026. Used to explain design evolution, not as a claim of current extraction capability.

S5. parameters/PLAN.md. Four-class SPI target, assessment status, teacher and student separation, group boundaries and proposed development gates.

S6. parameters/scene_arrays.py. Inspected source for the versioned arrays, masks, compatibility checks and byte layout. parameters/scene_vocabulary.py defines the registry imported by this module.

S7. parameters/minimal_parameters.py. Inspected source for the 28-byte compact core, 35-byte historical extension and explicit optional values.

S8. experiments/stream_scheduler/runs/mini-wsl-ingress-20261030-v2/REPORT.md. Cache hashes, wire parity, 6000 by 387 feature export, exposure, missing references and sparse-cost estimates.

S9. parameters/JOURNAL.md. Dated development record. Its 1 November entry records the version 11 partial comparison, storage failure and provisional leader. Its earlier entries record the frozen SigLIP study and scene contract checks.

S10. parameters/KAGGLE_RUNS.md. Historical teacher/parser failure and remote run provenance. Run entries establish only their stated notebook/version scope.

S11. experiments/spi_classifier_repair/RESULTS_REVIEW.md. EfficientNet-B1 v15 comparison against the earlier frozen CLIP baseline, failed readiness gates and retrieval update.

S12. experiments/spi_comparison_4000/PLAN.md. Recovery design, frozen-split repair and explicit development exposure of existing native test groups.

### Requested chat identifiers

Scheduler - Parameters. 01a16d36-ce1a-7802-82bf-70f4c2c1aec1.

Scheduler - Policy Design 2. 01a18b11-ea73-73d2-b0f9-73391ae34e38.

These identifiers record the requested provenance. Their contents were not accessed through read_thread. A later direct chat review may resolve decisions that the local records omit. No statement here attributes wording or approval to an unread chat.

### Reproduction boundary

The accompanying Markdown contains the report text. build_report.py creates this PDF and a source-hash manifest from that text. It does not execute project models or alter scientific run outputs. Validation of the report checks document structure and rendering, not the historical experiments themselves.
'''

TEXT = TEXT.replace('[[PAGE]]', '\n\n[[PAGE]]\n\n').replace('# Semantic decoder scheduling\n## ', '# Semantic decoder scheduling\n\n## ')
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='ReportTitle', fontName='Helvetica-Bold', fontSize=26, leading=30, spaceAfter=17, textColor=colors.black))
styles.add(ParagraphStyle(name='SectionTitle', fontName='Helvetica-Bold', fontSize=18, leading=22, spaceAfter=15, textColor=colors.black))
styles.add(ParagraphStyle(name='Subhead', fontName='Helvetica-Bold', fontSize=11.6, leading=15, spaceBefore=11, spaceAfter=7, textColor=colors.black))
styles.add(ParagraphStyle(name='ReportBody', fontName='Helvetica', fontSize=10.2, leading=13.5, spaceAfter=8))
styles.add(ParagraphStyle(name='CellBody', fontName='Helvetica', fontSize=9.3, leading=12.5, spaceAfter=0))
styles.add(ParagraphStyle(name='CellHead', fontName='Helvetica-Bold', fontSize=9.3, leading=12.5, textColor=colors.white))
styles.add(ParagraphStyle(name='ListItemReport', parent=styles['ReportBody'], leftIndent=14, firstLineIndent=-14))
styles.add(ParagraphStyle(name='SourceBody', parent=styles['ReportBody'], fontSize=9.5, leading=12.3, spaceAfter=7))

def para(s, style='ReportBody'):
    return Paragraph(escape(s), styles[style])

story=[]
first=True
appendix=False
for block in TEXT.strip().split('\n\n'):
    block=block.strip()
    if block=='[[PAGE]]':
        story.append(PageBreak())
    elif block.startswith('|'):
        rows = [[c.strip() for c in ln.strip().strip('|').split('|')] for ln in block.splitlines()]
        rows=[row for row in rows if not all(re.fullmatch(r'[-: ]+',c) for c in row)]
        data=[[para(c,'CellHead' if i==0 else 'CellBody') for c in row] for i,row in enumerate(rows)]
        widths=[165,339] if len(rows[0])==2 else None
        t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#334155')),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#CBD5E1')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F3F5F7')]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
        story.extend([t,Spacer(1,10)])
    elif block.startswith('# '):
        appendix=block.startswith('# Appendix')
        story.append(para(block[2:],'ReportTitle' if first else 'SectionTitle'))
        first=False
    elif block.startswith('## '):
        story.append(para(block[3:],'SectionTitle'))
    elif block.startswith('### '):
        story.append(para(block[4:],'Subhead'))
    elif re.match(r'1\. ',block):
        for line in block.splitlines(): story.append(para(line,'ListItemReport'))
    else:
        story.append(para(block,'SourceBody' if appendix else 'ReportBody'))

def footer(canvas,doc):
    canvas.saveState()
    canvas.setFont('Helvetica',8)
    canvas.setFillColor(colors.HexColor('#64748B'))
    canvas.drawString(54,28,'Semantic decoder scheduling | 4 November 2026')
    canvas.drawRightString(558,28,str(doc.page))
    canvas.restoreState()

pdf=OUT/'Semantic_decoder_scheduling_report.pdf'
doc=SimpleDocTemplate(str(pdf),pagesize=(612,792),leftMargin=54,rightMargin=54,topMargin=42,bottomMargin=43,title='Semantic decoder scheduling parameter development and policy design',author='Project technical report')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
(OUT/'Semantic_decoder_scheduling_report.md').write_text(TEXT.replace('[[PAGE]]\n',''),encoding='utf-8')
sources=['docs/decoder_context/SEMANTIC_RANKING_RESEARCH_DESIGN.md','README.md','docs/decoder_context/STREAM_BUILD_STATUS.md','docs/FINAL_PARAMETER_SPECIFICATION.md','parameters/PLAN.md','parameters/scene_arrays.py','parameters/minimal_parameters.py','experiments/stream_scheduler/runs/mini-wsl-ingress-20261030-v2/REPORT.md','parameters/JOURNAL.md','parameters/KAGGLE_RUNS.md','experiments/spi_classifier_repair/RESULTS_REVIEW.md','experiments/spi_comparison_4000/PLAN.md']
manifest={'inspection_date':'2026-11-04','head_at_inspection':'032c23a3dafc7bfeb39a84ba55193b6b6f3a22a0','direct_chat_review':False,'sources':[]}
for i,s in enumerate(sources,1):
    p=ROOT/s
    manifest['sources'].append({'id':f'S{i}','path':s,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
reader=PdfReader(pdf)
manifest['report_pages']=len(reader.pages)
manifest['pdf_sha256']=hashlib.sha256(pdf.read_bytes()).hexdigest()
(OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'pdf':str(pdf),'pages':len(reader.pages),'words':len(TEXT.split()),'page_text_lengths':[len(p.extract_text()) for p in reader.pages]},indent=2))
