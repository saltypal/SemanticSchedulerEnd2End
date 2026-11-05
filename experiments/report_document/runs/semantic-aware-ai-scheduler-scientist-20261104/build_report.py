from pathlib import Path

import ast,json,csv,re,hashlib

from docx import Document

from docx.shared import Inches,Pt,RGBColor

from docx.oxml import OxmlElement

from docx.oxml.ns import qn

from docx.enum.text import WD_ALIGN_PARAGRAPH



R=Path(__file__).resolve().parent;ROOT=R.parents[3];F=R/'figures'

prior=ROOT/'experiments/report_document/runs/semantic-aware-ai-scheduler-20261104/build_detailed_report.py'

for node in ast.parse(prior.read_text(encoding='utf-8')).body:

 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='refs' for t in node.targets):refs=ast.literal_eval(node.value)

refs['SQGAN']=('F. Pezone, S. Barbarossa and G. Caire. SQ-GAN: Semantic Image Communications Using Masked Vector Quantization. IEEE TCCN, 2025.','https://arxiv.org/abs/2502.09520')

refnums={k:i+1 for i,k in enumerate(refs)}

D=Document();s=D.sections[0];s.page_width=Inches(8.5);s.page_height=Inches(11)

s.left_margin=s.right_margin=Inches(.75);s.top_margin=Inches(.6);s.bottom_margin=Inches(.6)

for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Heading 3','Caption']:

 st=D.styles[name];st.font.name='Times New Roman';st.font.color.rgb=RGBColor(0,0,0)

for st in D.styles:

 pr=st.element.find(qn('w:pPr'))

 if pr is not None:

  for el in list(pr.findall(qn('w:pBdr'))):pr.remove(el)

D.styles['Normal'].font.size=Pt(11);D.styles['Normal'].paragraph_format.line_spacing=1.03;D.styles['Normal'].paragraph_format.space_after=Pt(6)

D.styles['Title'].font.size=Pt(25);D.styles['Heading 1'].font.size=Pt(16);D.styles['Heading 2'].font.size=Pt(12.5)

D.styles['Caption'].font.size=Pt(10);D.styles['Caption'].paragraph_format.keep_with_next=False

for name in ['Heading 1','Heading 2']:D.styles[name].paragraph_format.keep_with_next=True

foot=s.footer.paragraphs[0];foot.alignment=2

field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');foot._p.append(field)

figure=0;tab=0;record=[]

def cite(t):

 for k,n in refnums.items():t=t.replace('['+k+']','['+str(n)+']')

 return t

def p(t):

 t=cite(t);D.add_paragraph(t);record.append(t+'\n')

def h(t):D.add_heading(t,2);record.append('## '+t+'\n')

section_number=0

def section(t):

 global section_number

 if re.match(r'^\d+\.',t):

  section_number+=1;t=re.sub(r'^\d+\.',str(section_number)+'.',t)

 heading=D.add_heading(t,1)

 if record:heading.paragraph_format.page_break_before=True

 record.append('# '+t+'\n')

def table(title,headers,rows,widths=None):

 global tab

 tab+=1;c=D.add_paragraph('TABLE '+str(tab)+'. '+title);c.alignment=1;c.paragraph_format.keep_with_next=True

 t=D.add_table(rows=1,cols=len(headers));t.style='Table Grid';t.autofit=False

 if widths:

  for col,w in zip(t.columns,widths):col.width=Inches(w)

 for cell,v in zip(t.rows[0].cells,headers):cell.text=v

 for row in rows:

  for cell,v in zip(t.add_row().cells,row):cell.text=cite(str(v))

 for i,row in enumerate(t.rows):

  row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))

  for ci,cell in enumerate(row.cells):

   if widths:cell.width=Inches(widths[ci])

   sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'E6ECF1' if i==0 else ('F6F7F8' if i%2==0 else 'FFFFFF'));cell._tc.get_or_add_tcPr().append(sh)

   for pp in cell.paragraphs:

    pp.paragraph_format.space_before=Pt(4);pp.paragraph_format.space_after=Pt(4);pp.paragraph_format.line_spacing=1

    for rr in pp.runs:rr.font.size=Pt(10);rr.bold=i==0

 t.rows[0]._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'));D.add_paragraph()

 record.append('### '+title+'\n'+'\n'.join(' | '.join(map(str,row)) for row in [headers]+rows)+'\n')

def fig(name,caption,width=6.7):

 global figure

 figure+=1;path=F/(name+'.png')

 if name in ['ordering','time_definitions']:width=5.3

 if name in ['image_sources','label_pipeline']:width=5.5

 if name=='spi_recall':width=4.5
 if name=='header_packet':width=6.2

 from PIL import Image

 with Image.open(path) as im:width=min(width,6.8*im.width/im.height)

 D.add_picture(str(path),width=Inches(width));D.paragraphs[-1].alignment=1;D.paragraphs[-1].paragraph_format.keep_with_next=True

 c=D.add_paragraph('Fig. '+str(figure)+'. '+caption,style='Caption');c.alignment=1

 record.append(f'![{caption}]({path.as_posix()})\n')



t=D.add_paragraph('Semantic Aware AI Scheduler',style='Title');t.alignment=1

for line in ['Satya Srinivas Paladugu [BL.AI.U4AID24063]','Nandini Padavala [BL.SC.U4CSE24133]','Jahnvi KV [BL.SC.U4ECE24034]']:

 a=D.add_paragraph(line);a.alignment=1;a.paragraph_format.space_after=Pt(3)

h('Abstract')

p('Multiple devices deliver image messages to a receiver with a single nonpreemptive decoder. A receiver-side scheduler selects the next complete message for reconstruction. This project studies whether information about image importance, mission relevance, repetition and visible scene content can make that decision better. Each complete message contains semantic scheduling metadata and an encoded image payload. Selection uses the metadata and causal receiver context; reconstruction uses the selected payload.')

p('The work includes parameter extraction, four-class image-priority classification, a 6000-image dataset, simulated device streams and evaluation of eight rule-based policies and fourteen learned schedulers. On the matched development workload, FIFO recorded an average waiting penalty of 5.8220, SPI-first recorded 3.3741 and LambdaMART recorded 3.2183. Lower penalty means less costly waiting under the declared test rules. LambdaMART was the best of the methods compared, with 44.72 percent lower penalty than FIFO and 4.62 percent lower penalty than SPI-first. These are simulation results using model-generated priority references. All queue timings are expressed in ms, with a constant-service control of 1 ms per message; this is a simulator calibration, not measured SQ-GAN decoding latency.')

h('Project goal')

p('The goal is to show important and useful images to the receiver earlier. For example, a routine photograph should not automatically be processed before an image of a serious incident merely because it arrived first. The scheduler must also account for images left waiting. It cannot claim success by ignoring messages it never selects.')

section('2. Header and latent payload architecture')

fig('architecture','Sender and receiver architecture with SQ-GAN as the digital codec. Parameter extraction and latent encoding operate at the sender; reconstruction follows receiver-side selection.')

h('What the header contains')

p('The header is the information the scheduler can inspect cheaply. It contains the predicted priority, mission relevance, appearance novelty, scene measurements and the status of those measurements. Identity and timing fields tell the receiver which job it is handling and when it arrived. A status flag distinguishes a missing measurement from a measured zero.')

h('What the latent payload contains')

p('We selected SQ-GAN as the digital neural source codec [SQGAN]. It maps image features to discrete codebook entries through vector quantization, allowing the encoded representation to be serialized as a digital payload. Semantic masking allocates representation capacity to task-relevant content. This is a better architectural fit for the project’s byte-buffered receiver interface than treating analog channel symbols as an ordinary packet.')

p('The SQ-GAN paper reports perceptual and semantic-segmentation advantages over comparison codecs at very low bit rates [SQGAN]. Those published results motivate the choice; they are not a local, rate-matched codec benchmark on our scheduler dataset. Quantization makes the representation digital but does not make the reconstructed image pixel-exact.')

p('The repository contains an SQ-GAN digital adapter with native 256 by 512 RGB input and output. The transmitter obtains the paired semantic map, encodes the representation and produces SQ01 payload bytes. The receiver decodes those bytes without the original RGB image or transmitter map. SQ01 is the project’s experimental packet framing, not the paper’s entropy coder. Checkpoints and shared codebooks are setup assets rather than per-image header fields.')

p('Unlike direct analog JSCC mappings [JSCC], SQ-GAN provides discrete source representations suitable for byte-oriented message storage. The payload adapter still defines codec identity, tensor shape and framing.')

h('Why use parameters instead of decoding everything first')

p('Decoding every waiting image before choosing would spend the very decoder time that the scheduler is trying to manage. Numeric parameters provide a common input across images and codecs. They are still an incomplete description. Two images can have the same object counts but very different meaning, so the scheduler cannot recover facts that the header never captured.')





section('3. Receiver buffer and one-message dispatch')

fig('receiver_dispatch','Message flow at the receiver. Complete header-plus-payload messages enter the buffer. One configured scheduling algorithm scores eligible headers and dispatches exactly one selected payload to SQ-GAN decoding.')

p('The transmitter produces two outputs for each observation: scheduling parameters and an encoded latent payload. They are assembled into one complete message. The receiver admits the message to its buffer only after the message is complete and compatible with the declared codec profile.')

p('When the decoder is idle, the receiver invokes one selected scheduling policy, such as FIFO, SPI-first, RankNet or LambdaMART. It does not combine all policies into one decision. The policy evaluates currently buffered candidates and returns one message identity. The receiver removes that message from the ready set and sends its payload to the decoder. Other messages remain buffered, and new arrivals may join during processing.')

p('Dispatch is nonpreemptive and one-shot. A new Critical message cannot interrupt the payload already being decoded, and the scheduler does not create an application-level retransmission. The receiver updates its lifecycle records after completion and then takes a fresh queue snapshot. The algorithm comparison changes the configured selector while retaining the same workload and receiver lifecycle.')



section('3. Parameters and why each one is needed')

table('Parameter purpose and literature basis',['Parameter','Plain meaning','Reason and paper connection'],[['SPI','How serious does the visible situation appear?','Adds an image-level assessment. CrisisMMD and MEDIC motivate incident-image analysis [CRISIS], [MEDIC]. Our four urgency levels are project-defined.'],['Relevance','Does this image match the device’s mission?','CLIP compares image and task-text representations [CLIP]. The same image can matter differently for different missions.'],['Novelty','Does this look different from this sender’s recent images?','Separates appearance change from repetition. Content and age are also distinct in semantic freshness work [AOII]. This is an appearance proxy, not verified new knowledge.'],['Scene evidence','What configured objects or regions were detected?','YOLO-World and Grounding DINO motivate text-conditioned detection [YOLO], [DINO-DETECT]. Counts alone do not establish urgency.'],['Timing and status','When was it captured or received, and which measurements exist?','Prevents an old image, a failed detector and an empty scene being treated as the same case.']],[1.0,2.05,3.95])

h('Mission relevance with an example')

p('A drone inspecting road blockage and a camera recording household inventory have different missions. An image of a damaged road may match the first mission strongly and the second weakly. We encode the image and the supported mission description using CLIP, normalize both vectors, and calculate their cosine similarity [CLIP]. A similarity of 0.6 is a model score. It is not a 60 percent probability that the image is useful.')

h('Appearance novelty with an example')

p('The sender compares the current image with its stored images for the same device and mission. If the closest stored image has cosine similarity 0.8, novelty is calculated as one minus 0.8, divided by two, giving 0.1. An identical representation gives zero. The sender queries history before adding the current image, so an image cannot compare against itself.')

p('The executable default keeps 64 recent sender entries unless another capacity is supplied. Empty history has an explicit cold-start flag. Sender history records offered observations, while receiver history records processed outcomes. They serve different purposes. A new viewpoint can look novel without adding a new fact; a small important change can remain visually similar.')



section('4. Scene extraction and representation')

h('What the detector measures')

p('We use YOLO-World with a fixed category list [YOLO]. For each supported category, the extractor records the number of retained detections and the largest bounding rectangle as a fraction of the image. In a 400 by 200 image, a 100 by 80 rectangle occupies 8000 of 80000 pixels, giving an area fraction of 0.1. This is rectangle size in the image, not the object’s physical size or distance.')

p('The vocabulary has 93 entries: 80 COCO categories, five additional object categories and eight regions. The added objects are building, bridge, utility pole, traffic sign and animal. The regions are road, water, debris, smoke, flame, snow, ground and grass. Counts apply to 85 object categories. Region-count fields remain unavailable because several water detections do not mean several independent floods.')

h('Why YOLO-World was retained')

table('Detector-call measurements on the same 300 MEDIC images',['Operating point','Median call time','Mean boxes','Zero-detection images'],[['YOLO, confidence 0.25, IoU 0.5, size 640','19.00 ms','1.99','40.0%'],['Grounding DINO, box 0.35, text 0.25, IoU 0.5','368.69 ms','3.47','13.7%'],['Grounding DINO, box 0.35, text 0.35, IoU 0.3','369.64 ms','3.52','12.7%']],[3.4,1.1,.9,1.6])

p('The detector comparison used 16 nouns. It had no reference boxes, so more detections cannot be called better recall. YOLO had lower measured call time and was retained as the lower-cost baseline. These timings do not establish the speed or accuracy of the later 93-class setup. Grounding DINO provides a different text-grounded detection mechanism [DINO-DETECT].')

h('Features and bytes are different')

p('The evaluated scheduler uses a 387-entry base vector containing scene measurements, SPI, relevance, novelty and availability information. This model-input dimension is not a transmitted byte count. A compact packet must separately specify its field widths, quantization and compatibility profile.')

section('5. Compact header packet design')

fig('header_packet','Proposed 64-byte fixed scheduling header followed by the variable-length SQ01 latent payload. Byte offsets specify the header layout. The six scene descriptors are a reduced profile, not the full scene representation used for the reported comparisons.')

p('The proposed packet uses a fixed 64-byte scheduling header. It carries protocol and profile information, message identity, capture timing, payload description, quantized semantic scores and a bounded scene description. The variable-length SQ-GAN payload follows the header. The 64-byte figure excludes transport framing and the latent payload.')

table('Proposed 64-byte header allocation',['Byte offsets','Field group','Bytes'],[['0–3','Version, header length and profile identifier',4],['4–15','Source, session and message identifiers',12],['16–27','Capture timestamp and allowed delay, in ms',12],['28–35','Payload length and codec/shape profile',8],['36–39','SPI, relevance, novelty and status',4],['40–63','Up to six compact scene descriptors',24],['Total','Fixed scheduling header',64]],[1.2,4.8,1.0])

p('Each scene descriptor contains a category identifier, quantized count, quantized area and availability/status information. Both endpoints must share the profile’s vocabulary, field interpretation and quantization. Unknown or incompatible profiles are rejected. The reduced scene block omits categories beyond the selected descriptor slots; count clipping and area quantization must remain explicit.')

p('The 64-byte layout is a design specification. Reported scheduler scores retain the existing full scene features; they do not establish equivalent performance with this reduced header.')



section('5. Images and SPI classification')

fig('image_sources','Image sources in the 6000-image dataset. CrisisMMD supplies the crisis-image labeling cohort; Open Images and Places365 broaden the image pool.')

h('Which images were used and why')

p('The image pool contains 4000 CrisisMMD images, 1000 Open Images images and 1000 Places365 images. The 6000 files contain 5952 distinct raw images. Earlier parameter development used 300 images, including 220 LADI-v2 and 80 MEDIC examples. These stages are separate datasets, not extra independent test images.')

p('CrisisMMD contains disaster-related social-media images and associated humanitarian annotations [CRISIS]. MEDIC combines disaster-image sources, including overlapping CrisisMMD material [MEDIC]. Such data was suitable for an initial incident-focused SPI task. Its native annotations describe damage, objects or humanitarian categories. They do not directly provide our four intervention-need labels.')

p('Restricting SPI training to crisis imagery limits the claim. A classifier trained there may make errors on ordinary indoor scenes, screenshots or unfamiliar events. Adding everyday images to the receiver dataset broadens inputs but does not prove the classifier generalized correctly to them.')

h('Why four classes')

table('The four SPI levels',['Class','Meaning in this project','Example boundary'],[['Low','No visible problem requiring intervention, or routine content.','An ordinary street scene does not become urgent merely because people are present.'],['Medium','A visible issue needing attention without clear immediate severe danger.','A maintenance defect can need correction without an immediate emergency.'],['High','Serious visible damage or disruption supporting prompt attention.','Severe damage does not by itself prove a person is in immediate danger.'],['Critical','Visible evidence supporting immediate severe intervention need.','The label must not invent hidden victims or an unseen imminent collapse.']],[.8,3.05,3.15])

p('Four levels distinguish routine content, nonurgent issues, serious incidents and immediate severe need. Binary incident detection would merge several of these cases. Four levels also keep the classifier output small. The levels are a chosen annotation rubric, not universal emergency categories established by a paper. An unreadable or unsupported image has an unavailable assessment rather than automatically becoming Low.')



section('6. Labeling and training the SPI model')

fig('label_pipeline','SPI training and runtime are separate. A VLM supplies offline labels. A smaller image model predicts SPI during parameter extraction.')

h('How labels were obtained')

p('A vision-language model sees an image and a prompt defining visible evidence, the four classes and the required output structure. The system saves the image identity, raw response, parsed label and teacher configuration. Qwen2.5-VL provides the multimodal model foundation [QWEN]. A well-formed response is usable data, but is not proof that the judgment is correct.')

p('The 4000-image labeling file combines 2540 earlier checkpoint assessments and 1460 continuation assessments. The continuation used a pinned Qwen2.5-VL-7B-Instruct checkpoint with thinking disabled. There were 3933 available labels and 67 unavailable assessments. Because the protocols were mixed, this is not described as a single-teacher, single-prompt dataset.')

p('Earlier teacher attempts had malformed responses and unsupported visual claims, including dam or bridge details. Strict parsing detects format failures; it cannot detect every hallucination. The student learns from these AI-generated labels, so its reported accuracy measures agreement with them rather than independently verified real-world urgency. This is supervised classification from pseudo-labels, not automatically reinforcement learning from AI feedback.')

table('SPI class support after unavailable labels were excluded',['Class','Training','Validation','Test','Total'],[['Low',782,419,273,1474],['Medium',339,149,200,688],['High',899,445,325,1669],['Critical',48,24,30,102],['Total',2068,1037,828,3933]],[1.6,1.35,1.35,1.35,1.35])

h('From an image to a class')

p('The retained SPI pipeline first preprocesses the image, then uses CLIP ViT-B/32 to obtain a 512-value image representation [CLIP]. A logistic-regression head multiplies those values by learned weights and produces four class scores. The head has 2052 scalar coefficients and intercepts. The image encoder accounts for most of the pipeline cost. The CLIP representation used for these predictions is different from the reconstruction codec’s latent payload.')



section('7. SPI model comparisons and results')

h('What was compared')

p('The broader study compared four frozen feature encoders: CLIP, SigLIP2, DINO and ConvNeXtV2 [CLIP], [SIGLIP], [DINOV2], [CONVNEXT]. Frozen means their pretrained weights were reused rather than retraining the entire encoder. The saved encoder identity determines which DINO model or fallback was actually used.')

p('The twelve heads were logistic regression, ridge, linear SVM, RBF-SVM, k-nearest neighbors, random forest, extra trees, histogram boosting, MLP, mean-MLP, gated attention MIL and single-query cross-attention. Linear heads test whether the existing features already separate the labels. Nonlinear heads allow more complex boundaries. Token heads examine patch-level features instead of relying only on one pooled vector. Attention-based multiple-instance learning motivates weighted aggregation [MIL].')

h('The classifier used for the dataset')

p('CLIP with class-balanced logistic regression was fitted on 2068 training examples. Three regularization values, 0.1, 1 and 10, were compared on 1037 validation examples. The selected value was 10. The resulting model was evaluated on the saved 828-image test split and then used to populate SPI predictions in the image dataset.')

table('SPI classifier results against AI-generated labels',['Metric','Result'],[['Validation macro-F1','0.60245'],['Test macro-F1','0.59997'],['Test accuracy','64.86%'],['Low recall','79.49%'],['Medium recall','36.00%'],['High recall','69.54%'],['Critical recall','73.33%']],[4.8,2.2])

p('Accuracy counts all correct predictions together. Recall asks how many images of one class were recognized correctly. Macro-F1 gives each class equal influence, so the common classes cannot hide every minority-class error. Medium recall of 36 percent means 72 of its 200 test examples were recognized as Medium.')

fig('spi_recall','Class recall for the SPI classifier. Medium was the weakest class. These values measure agreement with pseudo-labels, not human-verified urgency.')



section('8. The waiting budget explained')

h('Waiting-time budgets')

p('In this experiment, a waiting budget means how long a message is allowed to wait before processing should start. It does not force the message to wait that long. A Critical image can start immediately. It also does not mean that its decoder takes 2 ms to run. Every job takes 1 ms in the constant-service control, regardless of class.')

table('Chosen waiting targets for the scheduling experiment',['Reference class','Target waiting time','Meaning'],[['Critical','2 ms','Waiting becomes late after 2 ms.'],['High','4 ms','Waiting becomes late after 4 ms.'],['Medium','8 ms','Waiting becomes late after 8 ms.'],['Low','16 ms','Waiting becomes late after 16 ms.']],[1.25,1.9,3.85])

p('The numbers 2, 4, 8 and 16 were chosen as experimental settings to make postponing important images more costly. They are not emergency-service standards and were not discovered by the VLM. Using them lets every policy face the same scoring rule. Missing urgency references cannot be filled by simply assigning Low.')

h('Waiting-cost function')

p('A short wait is usually less serious than a long wait. The same wait also matters more when the image has a shorter target. First divide the actual wait by the target. After the target is exceeded, add the excess fraction again. The result is called a waiting penalty. Smaller is better.')

p('For a wait w and a target b, penalty = w/b + max of zero and w/b minus one. Before the target, the penalty rises steadily. After the target, it rises twice as fast. At exactly the target, the penalty is 1.')

table('Worked examples for the same scoring rule',['Image','Wait','Calculation','Penalty'],[['Critical, target 2','1','1/2 + 0','0.50'],['Critical, target 2','2','2/2 + 0','1.00'],['Critical, target 2','4','4/2 + (4/2 − 1)','3.00'],['Low, target 16','4','4/16 + 0','0.25']],[2.0,.8,2.9,1.3])

p('Thus a 4-ms wait costs 3 for a Critical image and 0.25 for a Low image. This is how the test expresses that delaying the Critical image is more costly. The penalty is dimensionless. A score of 3 does not mean a 3 ms delay.')



section('9. How a complete schedule receives its score')

fig('budget_curves','The waiting-penalty curves for the chosen targets. Each dot marks the target wait, where penalty equals 1. Critical waiting accumulates penalty fastest.')

h('Worked scheduling example')

p('Suppose Routine, High and Critical images all become ready at time 0, and each takes 1 ms to process. Assume the Routine image arrived first among that tied group. FIFO processes Routine, then High, then Critical. Their waits are 0, 1 and 2. Their penalties are 0, 0.25 and 1, giving a mean of 0.4167.')

p('SPI-first processes Critical, then High, then Routine. Their waits are 0, 1 and 2, but the penalties are now 0, 0.25 and 0.125. The mean is 0.125. Both policies processed three images in 3 ms. SPI-first received a better score because it placed the costly wait on the less urgent image. This example assumes correct priority labels and is illustrative, not a measured run.')

h('Images left in the queue still count')

p('The experiment adds the penalty of every assessable offered image and divides by the number of those images. An image that starts processing stops accumulating waiting penalty. Its existing penalty remains in the total. An image still waiting when the episode ends contributes its elapsed wait up to that endpoint. Otherwise a policy could appear good by serving only a few easy images and ignoring the rest.')

h('How the score is used for learning')

p('Supervised schedulers learn which candidate would cause the most extra penalty if it waited one additional 1 ms service interval. Reinforcement-learning agents receive a negative reward when total waiting penalty increases. The stored reward divides that increase by 100 to control its numeric scale. This scaling is a training setting, not another waiting target.')

p('Reference priority is kept separate from runtime inputs. A policy acts on predicted SPI and permitted context; the evaluator scores it using the saved reference. The current references are silver labels, meaning model-generated judgments rather than independently verified operational urgency.')



section('10. Traditional and fixed-rule scheduling algorithms')

p('These methods do not train a queue-selection model. Each follows a fixed rule. Some ignore image meaning, while others use the extracted semantic parameters. All choose from the same ready queue and obey the same one-job-at-a-time rule.')

table('The eight fixed-rule policies evaluated',['Policy','How it chooses','What it tests'],[['FIFO','Choose the earliest arrival.','Whether arrival order alone is sufficient.'],['Round robin','Rotate service among senders.','Fairness between devices.'],['Random','Choose a random ready message using a saved seed.','A chance-based control.'],['SPI-first','Choose the highest predicted priority; break ties by arrival.','A simple importance-aware rule.'],['Relevance-first','Choose the strongest mission match.','Whether task alignment alone gives a useful order.'],['Novelty-first','Choose the largest appearance change.','Whether visual difference alone gives a useful order.'],['SPI then novelty','Choose priority first and novelty within the class.','Whether repetition helps resolve priority ties.'],['Predicted EDF','Choose the earliest arrival-plus-predicted-budget deadline.','A time-aware rule based on predicted classes.']],[1.3,3.1,2.6])

h('Why these are useful comparisons')

p('FIFO and round robin answer familiar questions: should the first arrival go first, or should devices take turns? SPI-first answers the stronger question: can a simple priority rule already solve much of this test? A learned method should be compared with that rule as well as FIFO. Otherwise its advantage may mostly come from using priority information at all.')

p('Novelty-first can select a visually different but unimportant scene. Relevance-first can favor a mission match without recognizing immediate need. Predicted EDF uses an experimental deadline derived from a predicted class, not a real application deadline. These differences explain what each rule measures without assuming in advance which one must win.')



section('11. Learned scheduling algorithms')

p('A learned scheduler receives numeric inputs and produces a score for each waiting candidate. The receiver selects the candidate with the highest score. Different models learn that score in different ways. The architecture describes how the score is computed; the loss or reward describes how training changes it.')

table('Supervised methods and their plain-language roles',['Method','How it learns or scores'],[['Linear choice','Adds learned weighted contributions from the inputs. It is the simplest trained scorer.'],['MLP choice','Uses a small neural network to learn nonlinear combinations and select a candidate.'],['Delay regression','Predicts the extra waiting cost caused by postponement. Huber loss reduces the influence of large individual errors.'],['Priority classifier','Predicts four reference classes and uses them for waiting-sensitive selection.'],['RankNet','Learns which member of a candidate pair should be ahead [RANKNET].'],['LambdaMART','Combines many small decision trees to learn useful ordering [LAMBDAMART].'],['Deep Sets RankNet','Scores a candidate using its own fields plus pooled queue and history information [SETS].'],['Tie-aware listwise','Learns an ordering across several candidates while respecting equal-cost cases. The implementation adapts listwise learning [LISTMLE].'],['Attention set ranker','Lets candidate representations exchange information before ranking. It is an adaptation of set attention [ATTENTION].'],['Temporal Deep Sets','Adds a GRU that reads up to three recent observations in order before scoring.']],[1.65,5.35])

h('What a pairwise ranker learns')

p('If waiting 1 ms longer is more costly for image A than image B, a pairwise ranker should give A the higher score. RankNet turns the score difference into a pair preference and adjusts its weights when the preference is wrong [RANKNET]. It learns ordering rather than a reconstruction of the image.')

h('What history adds')

p('A set model summarizes the current queue without treating array position as meaning. A temporal model also examines the order of recent observations. These are possible ways to use context. Their extra complexity does not guarantee better results, so they are compared on the same recorded workload.')



section('12. Reinforcement learning and training limits')

table('The four reinforcement-learning methods',['Method','Plain-language explanation'],[['PPO','Learns a probability for each legal choice and adjusts the policy using accumulated reward [PPO].'],['Double DQN','Learns the value of choosing a candidate. Separate networks choose and evaluate the next action during training [DDQN].'],['Double DQN with prioritized replay','Revisits stored training transitions more often when their prediction errors are larger [PER]. This replay priority is not operational image priority.'],['Ranker-initialized PPO','Starts PPO from an already trained ranker instead of a fresh actor, then updates it using queue rewards.']],[2.1,4.9])

p('Legal-action masks prevent the agent from selecting a message that is not ready. A decision advances through service and any required idle interval to the next decision point. The environment charges waiting accumulated during those intervals. New arrivals are therefore relevant even when the server is busy.')

h('Training budget is different from waiting budget')

p('A waiting budget tells the evaluator when an image is late. A training budget tells us when to stop running a learning experiment. For example, 20000 action decisions means an RL learner was allowed that many choices during training. It does not mean that an image may wait for 20000 ms.')

table('Recorded scheduler training settings',['Item','Setting'],[['Supervised training','64 development episodes and 512 sampled decision snapshots per method.'],['Supervised epoch limit','Up to 50 passes through training data, with early stopping after eight validation checks without improvement.'],['RL interaction limit','20000 action decisions per method.'],['Model seed','1729.'],['Validation workload','15 traces spanning device counts and offered loads.'],['Paired notebook runtime','Python 3.13.15, Torch 2.11.0, two T4 GPUs per paired notebook.'],['Recorded paired run durations','About 2 hours 56 minutes and 2 hours 46 minutes, including their assigned execution work.']],[2.1,4.9])

p('The project has struggled to complete its intended model comparison and repeated evaluation within available compute, GPU memory, storage and runtime limits. Fourteen scheduler methods nevertheless completed development evaluation. These limits do not establish that they converged. Some RL checkpoints became worse later in training. The full SPI comparison also stopped after disk exhaustion, so its unfinished parts cannot be described as completed.')



section('13. How the scheduling comparison was run')

h('A shared workload for a fair comparison')

p('The simulator saves message arrivals, processing times and possible outcomes before comparing policies. Each policy receives the same realization. Thus a policy is not allowed a different, easier set of arrivals just because it is a different algorithm. Only complete messages become eligible, and later arrivals remain unavailable until their arrival time.')

p('Development validation includes device populations of 4, 12, 50, 60 and 100. Fifteen mission templates represent phones, inspection and search drones, vehicle cameras, CCTV, industrial robots and other sources. A device mission supplies context, not an automatic priority. A normal phone can capture a serious incident; an inspection drone can send a routine repeat.')

h('Matched workload conditions')

p('One test keeps total offered load fixed while changing the number of devices. Another keeps each device’s rate fixed, so adding devices also adds traffic. Bursts concentrate arrivals in time. Joining and pausing devices alter achieved traffic. These controls separate the effect of more senders from the effect of more messages.')

h('What the reported cohort covers')

p('The broad pool contains 6000 images. The saved receiver report lists 5376 training, 417 validation and 207 test records for that pool, with a separate 3915-record cohort having assessable delay references. Those counts differ from the SPI classifier’s 2068/1037/828 split because the experiments use different eligible records and roles. They must not be presented as one split.')

p('The following comparison uses saved development results, not the protected final scheduler test. Model-generated priority references define the assessable urgency scope. Simulated times and mixed image streams are not live recordings of independent emergency incidents. The reported numeric differences describe these runs; they are not confidence intervals or evidence of deployment safety.')

h('Reading the next table')

p('All methods use the same waiting-penalty definition. Smaller is better. The percentage improvement over FIFO is the FIFO penalty minus the method penalty, divided by the FIFO penalty, then multiplied by 100. A negative value means the method was worse than FIFO. The colors in the plot distinguish ordinary scheduling, fixed semantic rules and learned policies.')



section('14. Direct comparison of all evaluated algorithms')

data=json.loads((R/'comparison.json').read_text(encoding='utf-8'));rows=data['rows'];fifo=next(x['penalty'] for x in rows if x['id']=='fifo');spi=next(x['penalty'] for x in rows if x['id']=='spi_fifo')

table('All eight fixed rules and fourteen learned schedulers on the development comparison',['Algorithm','Type','Penalty ↓','Change vs FIFO'],[[x['name'],x['family'],f"{x['penalty']:.4f}",f"{100*(fifo-x['penalty'])/fifo:+.2f}%"] for x in rows],[2.8,1.4,1.2,1.6])

p('LambdaMART recorded the lowest penalty, 3.2183. FIFO recorded 5.8220 and round robin recorded 5.8059. SPI-first recorded 3.3741. Thus the best learned model improved on FIFO by 44.72 percent, while the fixed SPI-first rule improved on FIFO by 42.05 percent. The additional reduction from SPI-first to LambdaMART was 4.62 percent. These percentages describe waiting penalty, not time saved in milliseconds.')



section('15. Comparison plots')

fig('all_methods','All 22 evaluated methods, sorted from lowest to highest penalty. Ordinary policies are gray, fixed semantic rules are green and learned methods are blue. The same scoring rule is used throughout.',width=6.25)

p('The plot makes the main comparison visible. Most learned models beat FIFO in this run, but a learned model is not automatically better than SPI-first. Delay regression and RankNet joined LambdaMART in numerically beating SPI-first. The remaining eleven learned methods did not. Novelty-first was worse than FIFO, showing that visual difference alone did not match this waiting objective well.')



section('16. What changes when priority is used')

fig('key_comparison','Four representative methods on the same development comparison. Most of the observed FIFO-to-LambdaMART reduction was already achieved by the simple SPI-first rule.')

h('Traditional policies')

p('FIFO and round robin produced similar aggregate penalties. Round robin shares opportunities among senders, but it does not automatically recognize an important image. Random was slightly worse than FIFO. These results show the limitation of ignoring image-priority information under a priority-sensitive scoring rule.')

h('Fixed semantic rules')

p('SPI-first was the strongest fixed semantic rule in this development table. SPI then novelty was close but had a higher penalty, 3.4349. Relevance-first and predicted EDF improved on FIFO but did not match SPI-first. Novelty-first had penalty 6.1917, which was worse than FIFO. A surprising image is not necessarily a costly image to postpone.')

h('Learned models')

p('LambdaMART, delay regression and RankNet had the three lowest learned penalties. The temporal and attention models did not improve on the simpler SPI-first control in this run. The four RL methods also had higher penalties than SPI-first; ranker-initialized PPO was the best among those four. This is the outcome of the recorded training settings, not a universal ranking of algorithm families.')

p('The target explains part of this result. It charges waiting according to reference priority and elapsed wait. It does not independently reward every visual novelty or mission distinction within a priority class. The comparison therefore measures success under that declared waiting rule, rather than complete semantic understanding.')



section('17. Which classes waited less')

fig('class_tradeoffs','Left: the percentage selected within the class target. Right: messages still unselected when the episode ended. Both views matter because improving urgent service can increase waiting elsewhere.')

table('FIFO and SPI-first on the same 15 validation episodes',['Policy','Class','Offered','Unselected','Selected mean wait','Within target'],[['FIFO','Critical',75,9,'21.441','17.3%'],['SPI-first','Critical',75,0,'2.929','89.3%'],['FIFO','High',1087,92,'22.099','33.3%'],['SPI-first','High',1087,12,'8.314','63.6%'],['FIFO','Medium',435,39,'20.884','42.5%'],['SPI-first','Medium',435,16,'15.005','63.4%'],['FIFO','Low',846,69,'19.282','52.8%'],['SPI-first','Low',846,181,'34.582','39.4%']],[1.1,.9,.8,1.0,1.6,1.6])

p('For Critical images, SPI-first left none unselected and had a selected mean wait of 2.929 ms, compared with FIFO’s 21.441. However, it left 181 Low images unselected, compared with FIFO’s 69. Prioritization therefore changed who received service sooner; it did not make every class better at the same time.')

p('Selected mean wait includes only messages that started processing. It must be read with unselected counts. The within-target percentage uses the evaluator’s eligible follow-up observations. Those denominators differ, so a rounded percentage should not be reconstructed simply by dividing selected messages by all offered messages. Class-level learned-model outcomes are not supplied by the retrieved aggregate table, so the report does not assign LambdaMART an invented Critical-class benefit.')



section('19. Computational cost and experiment scope')

h('Where resources were spent')

p('The expensive stages were not limited to scheduler scoring. They included VLM labeling, image-encoder feature extraction, storage of patch-level features, classifier searches and RL interactions. A 49 by 768 CLIP token array uses 150528 bytes per image in float32. Storing it for 3933 images takes about 565 MiB before compression and metadata. Multiple feature banks and checkpoints multiply that cost.')

p('The broader SPI comparison completed twelve original-setting heads each for CLIP, SigLIP2 and DINO, for 36 completed candidate results. ConvNeXtV2 feature extraction stopped when storage filled after about 97 minutes. Other balancing regimes and the complete final selection did not finish. The best development candidate among the 36 was CLIP with single-query cross-attention, macro-F1 0.63365. This is a different candidate and evaluation role from the dataset classifier’s test macro-F1 0.59997.')

h('Measured SPI inference cost')

table('Raw-image-to-SPI profile of CLIP plus logistic regression',['Measurement','CPU','GPU'],[['Median call','104.74 ms','28.56 ms'],['95th percentile','112.46 ms','38.84 ms'],['Timed calls','30','100']],[3.1,1.95,1.95])

p('The profile used the same ten development images, ten warmups, three CPU repeats per image and ten synchronized GPU repeats. It covers the SPI path, not the full detector, mission, novelty and serialization pipeline. The older detector timings used another scope, so their percentile values cannot simply be added to these values to create a measured full-header percentile.')

h('What the evidence supports')

p('The experiments support a completed extraction dataset, measured pseudo-label agreement for SPI, and development comparisons of fixed and learned schedulers. They do not establish general emergency understanding, exact image recovery or measured end-to-end radio latency. The SPI test cohort had already been exposed during development. The protected scheduler final test remained unopened in the supplied records. The delay references are model-generated, and queue variations from one seed are not independent disaster events.')

h('Conclusion')

p('Semantic priority changed processing order enough to reduce the recorded waiting penalty substantially compared with FIFO. LambdaMART gave the lowest score among the 22 compared methods, but SPI-first captured most of the observed improvement with a fixed rule. The results therefore distinguish two findings: using priority information helped strongly in this experiment, while the extra gain from learning was smaller. The report states both results together, with the waiting rule, workload and resource limits that define them.')



section('Appendix A. Complete SPI encoder comparison')

p('The values below are development macro-F1, not final test scores. Macro-F1 averages the four class F1 scores equally. All 36 rows completed in the original-setting comparison; ConvNeXtV2 did not complete its feature extraction. The same numbers are used in the source completed-comparison CSV.')

comp=list(csv.DictReader((ROOT/'experiments/parameter_extraction_report/completed_comparison.csv').open()))

heads=list(dict.fromkeys(x['head'] for x in comp))

table('All completed encoder and classifier pairs',['Classifier head','CLIP','SigLIP2','DINO'],[[name]+[f"{float(next(x['development_macro_f1'] for x in comp if x['head']==name and x['encoder']==enc)):.4f}" for enc in ['clip_b32','siglip2_b16','dino']] for name in heads],[3.4,1.2,1.2,1.2])

p('Logistic and ridge heads use linear decision boundaries. SVM adds either a linear or nonlinear boundary. k-nearest neighbors compares a new vector with stored training vectors. Tree ensembles combine many split-based rules. MLP learns nonlinear combinations. Mean-MLP averages patch features, gated attention MIL learns their weights, and single-query cross-attention learns a query that collects useful patch information. Each head must use the same encoder and preprocessing on which it was fitted.')



section('References')

for k,(title,url) in refs.items():

 pp=D.add_paragraph('['+str(refnums[k])+'] '+title+' ')

 hl=OxmlElement('w:hyperlink');rel=pp.part.relate_to(url,'http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink',is_external=True);hl.set(qn('r:id'),rel)

 rr=OxmlElement('w:r');tt=OxmlElement('w:t');tt.text=url;rr.append(tt);hl.append(rr);pp._p.append(hl)

 for run in pp.runs:run.font.size=Pt(10)

 record.append('['+str(refnums[k])+'] '+title+' '+url+'\n')

D.save(R/'Semantic_Aware_AI_Scheduler.docx')

(R/'Semantic_Aware_AI_Scheduler.md').write_text('\n'.join(record),encoding='utf-8')

sources=[prior,ROOT/'experiments/parameter_extraction_report/REPORT.md',ROOT/'experiments/parameter_extraction_report/completed_comparison.csv',ROOT/'experiments/results_analysis/runs/analysis-20261104/ANALYSIS.md',Path(data['baseline_path'])]

(R/'source_manifest.json').write_text(json.dumps({'sources':[{'path':str(x),'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} for x in sources],'compared_policies':len(rows),'figures':figure,'tables':tab,'new_training':False,'new_final_test_access':False,'rendering':'Microsoft Word PDF export because packaged LibreOffice renderer is unavailable'},indent=2))

print('Report built:',figure,'figures;',tab,'tables;',len(' '.join(record).split()),'words')

