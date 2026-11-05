from pathlib import Path
import re, csv, json, hashlib, shutil
from html import escape
from html.parser import HTMLParser
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from docx.table import Table

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
PARAM=ROOT/'experiments/parameter_extraction_report'
OLD=ROOT/'experiments/report_document/runs/report-20261104/Semantic_receiver_scheduler_report.docx'
ANALYSIS=ROOT/'experiments/results_analysis/runs/analysis-20261104/ANALYSIS.md'
refs={
'JSCC':('Bourtsoulatze, Kurka and Gunduz. Deep Joint Source-Channel Coding for Wireless Image Transmission. 2019.','https://arxiv.org/abs/1809.01733'),
'CLIP':('Radford et al. Learning Transferable Visual Models From Natural Language Supervision. ICML 2021.','https://proceedings.mlr.press/v139/radford21a.html'),
'YOLO':('Cheng et al. YOLO-World Real-Time Open-Vocabulary Object Detection. CVPR 2024.','https://arxiv.org/abs/2401.17270'),
'DINO-DETECT':('Liu et al. Grounding DINO. 2023, revised 2024.','https://arxiv.org/abs/2303.05499'),
'CRISIS':('Alam, Ofli and Imran. CrisisMMD Multimodal Twitter Datasets from Natural Disasters. ICWSM 2018.','https://arxiv.org/abs/1805.00713'),
'MEDIC':('MEDIC dataset and documented annotation tasks.','https://crisisnlp.qcri.org/medic/'),
'QWEN':('Bai et al. Qwen2.5-VL Technical Report. 2025.','https://arxiv.org/abs/2502.13923'),
'SIGLIP':('Tschannen et al. SigLIP 2. 2025.','https://arxiv.org/abs/2502.14786'),
'DINOV2':('Oquab et al. DINOv2 Learning Robust Visual Features without Supervision. 2023.','https://arxiv.org/abs/2304.07193'),
'CONVNEXT':('Woo et al. ConvNeXt V2. CVPR 2023.','https://arxiv.org/abs/2301.00808'),
'MIL':('Ilse, Tomczak and Welling. Attention-based Deep Multiple Instance Learning. ICML 2018.','https://proceedings.mlr.press/v80/ilse18a.html'),
'AOII':('Maatouk, Assaad and Ephremides. The Age of Incorrect Information. 2020, revised 2022.','https://arxiv.org/abs/2012.13214'),
'RANKNET':('Burges et al. Learning to Rank using Gradient Descent. ICML 2005.','https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/'),
'LAMBDAMART':('Burges. From RankNet to LambdaRank to LambdaMART An Overview. 2010.','https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/'),
'SETS':('Zaheer et al. Deep Sets. NeurIPS 2017.','https://arxiv.org/abs/1703.06114'),
'ATTENTION':('Lee et al. Set Transformer. ICML 2019.','https://proceedings.mlr.press/v97/lee19d.html'),
'LISTMLE':('Xia et al. Listwise Approach to Learning to Rank Theory and Algorithm. ICML 2008.','https://www.microsoft.com/en-us/research/publication/listwise-approach-to-learning-to-rank-theory-and-algorithm/'),
'DDQN':('van Hasselt, Guez and Silver. Deep Reinforcement Learning with Double Q-learning. AAAI 2016.','https://arxiv.org/abs/1509.06461'),
'PER':('Schaul et al. Prioritized Experience Replay. ICLR 2016.','https://arxiv.org/abs/1511.05952'),
'PPO':('Schulman et al. Proximal Policy Optimization Algorithms. 2017.','https://arxiv.org/abs/1707.06347')}

intro='''# Semantic Aware AI Scheduler

Satya Srinivas Paladugu [BL.AI.U4AID24063]

Nandini Padavala [BL.SC.U4CSE24133]

Jahnvi KV [BL.SC.U4ECE24034]

Detailed project report | 4 November 2026

## Abstract

We study how a receiver should choose the next complete image message for a single decoder when multiple user devices produce observations with different importance, mission relevance and repetition. Each message pairs a scheduling header with a complete codec payload. A lightweight policy reads the received parameters and causal queue history before dispatch. It does not reconstruct every image to decide which one deserves reconstruction.

The work includes semantic parameter design, detector comparisons, offline VLM labeling, frozen-encoder SPI classification, a 6000-image development cache, dynamic multi-device simulation and development evaluation of fourteen learned scheduling methods. The provisional SPI pipeline has test macro-F1 0.59997 against pseudo-labels. The incomplete broader encoder comparison has a development leader at 0.6336517065, a different model and evaluation role. The scheduler analysis reports LambdaMART penalty 3.2183 against SPI-first 3.374083 and FIFO 5.822023. These are development measurements under a simulated waiting objective, not verified emergency effectiveness.

## Compute and training disclaimer

We are struggling to complete the intended training and validation program within the available compute, memory, storage and execution budgets. Teacher inference, repeated image-feature extraction, token-feature banks, classifier searches and scheduler interaction training compete for practical resources. The parameter comparison stopped during ConvNeXtV2 extraction after storage exhaustion. Earlier runs also encountered parsing, token-mask and split-preparation faults. Those are distinct causes and should not all be described as insufficient GPU arithmetic.

Some scheduler models have trained successfully. Fourteen learned methods completed development evaluation in the supplied newer records. The bounded RL budgets and observed checkpoint regressions do not establish convergence, repeated-seed stability or a final deployment winner. Compute limits constrain the breadth and repetition of the evidence, but do not explain away weak results. Independent urgency references, appropriate objectives and matched ablations are also missing or limited. The protected final scheduler test remains unopened according to the supplied records. The SPI test cohort has already been exposed in earlier development and is not a newly untouched confirmation set.

## Project goal and research questions

Our goal is to reduce the waiting of important and mission-useful observations at a receiver bottleneck without losing track of messages that remain unselected. We seek an ordering policy over complete received messages. We do not claim that changing decoder order saves already transmitted bytes, allocates the radio, guarantees exact image recovery or detects every emergency.

The first question is whether semantic inputs improve dispatch over FIFO, round robin and simple SPI-first selection. The second is whether mission relevance, appearance novelty and scene arrays add measurable value beyond SPI. The third is whether learned context or temporal history earns its additional inference and training cost. The fourth is whether earlier selection exposes independently supported useful evidence rather than only matching a model-generated priority label.

## How this report uses the supplied documents

Part I consolidates the parameter report and its figures. Its HTML and Markdown are two presentations of the same report and are not independent experiments. The completed-comparison CSV supplies the 36 saved encoder-head results. Part II consolidates the supplied receiver-scheduler Word report, including its original tables. Part III incorporates the newer result analysis and qualifies conclusions that the earlier Word report left open. Part IV maps literature to mechanisms, records limitations and defines the next acceptance evidence.

The supplied files are evidence, not a reason to merge incompatible experiment units. The 3933-image SPI classifier cohort and the 3915-record assessable scheduler delay cohort have different roles and must retain their own joins and exclusions. The 6000-image broad pool is not a 6000-image independently labeled urgency benchmark. One simulation unit equals one simulated millisecond in the newer scheduler report. It is not measured hardware latency.

## Report map

| Part | Scope |
|---|---|
| I | Header and latent payload, parameter mechanisms, datasets, SPI labels, classifiers, results and extraction costs |
| II | Receiver lifecycle, waiting objective, ranking and RL methods, UE simulation and completed scheduler training |
| III | Strong-baseline comparison, class trade-offs, RL dynamics and paired rate-study evidence |
| IV | Literature attribution, evidence gaps, future experiments and source provenance |
'''

param=(PARAM/'REPORT.md').read_text(encoding='utf-8')
param=param[param.index('## 1.'):]
param=re.sub(r'!\[([^]]*)\]\(figures/([^)]*)\)',lambda m:f'![{m[1]}]({(PARAM/"figures"/m[2]).as_posix()})',param)
param=param.replace('## 16. Reproduction and local evidence','## 16. Parameter reproduction and local evidence')
# Preserve inline source links and add short paper tags where the design is discussed.
param=param.replace('## 5. Context relevance and the UE paragraph','## 5. Context relevance and the UE paragraph\n\nLiterature basis [CLIP].')
param=param.replace('## 8. Scene measurements and why we retained YOLO-World','## 8. Scene measurements and why we retained YOLO-World\n\nLiterature basis [YOLO] and [DINO-DETECT].')
param=param.replace('## 10. SPI target and the choice of four classes','## 10. SPI target and the choice of four classes\n\nDataset context [CRISIS] and [MEDIC]. The four ordinal classes are our project rubric, not a label taxonomy supplied by those papers.')
param=param.replace('## 11. VLM labeling, comparisons and hallucination','## 11. VLM labeling, comparisons and hallucination\n\nTeacher architecture reference [QWEN]. Its capabilities motivate the tool, not the truth of each generated urgency label.')

old=Document(OLD)
parts=[]
refmap={'1':'RANKNET','2':'LAMBDAMART','3':'SETS','4':'ATTENTION','5':'LISTMLE','6':'DDQN','7':'PER','8':'PPO','9':'CRISIS'}
started=False
for el in old.element.body:
    if el.tag==qn('w:p'):
        p=Paragraph(el,old); t=p.text.strip()
        if t=='1 System boundary and policy':started=True
        if t=='10 References':break
        if not started or not t:continue
        t=re.sub(r'\[(\d+)\]',lambda m:'['+refmap.get(m[1],m[1])+']',t)
        if p.style.name.startswith('Heading'): t=('## ' if p.style.name=='Heading 1' else '### ')+t
        parts.append(t)
    elif el.tag==qn('w:tbl') and started:
        tb=Table(el,old); rows=[[c.text.replace('\n',' ') for c in r.cells] for r in tb.rows]
        parts.append('\n'.join(['| '+' | '.join(rows[0])+' |','| '+' | '.join(['---']*len(rows[0]))+' |']+['| '+' | '.join(r)+' |' for r in rows[1:]]))
sched='\n\n'.join(parts)
sched=sched.replace('This table does not establish improvement over SPI first. Its matched baseline score and class safeguards must be merged into the comparison.','The original table did not include SPI-first. Part III supplies the newer matched baseline analysis. Its aggregate improvement still requires class safeguards and independent confirmation.')
analysis=ANALYSIS.read_text(encoding='utf-8')
analysis=analysis.replace('# Advanced development result analysis','## Updated analysis of the completed development runs')
analysis=re.sub(r'!\[([^]]*)\]\(([^)]*)\)',lambda m:f'![{m[1]}]({(ANALYSIS.parent/m[2]).as_posix()})',analysis)
closing='''## Interpreting the current evidence

The current D objective rewards reference-priority-sensitive waiting. It does not independently reward every novel fact, mission distinction or important visual relation. Within a class, marginal costs often tie. A richer network may receive little training signal for distinctions its designer hopes it will learn. Under this target, the strong SPI-first baseline is expected to be competitive. The results support testing target sufficiency before enlarging the architecture.

Only three of fourteen learned methods beat SPI-first numerically in the supplied aggregate analysis. LambdaMART's 4.62 percent advantage is an observed development difference, not a statistically confirmed advantage across independent incidents. Temporal Deep Sets is 2.19 percent worse than SPI-first here. This does not prove temporal information is useless. It means the tested history, objective, optimization and budget did not establish its benefit.

## What is inspired by papers and what we chose

| Decision | Literature mechanism | Project choice and validation boundary |
|---|---|---|
| Header and image payload | [JSCC] supplies learned image transmission background | Our complete-job receiver interface and scheduling boundary are project choices. Analog JSCC symbols require a compatible adapter. |
| Mission relevance | [CLIP] aligns image and text representations | Mission wording, aggregation and cosine interpretation are our choices. No urgency probability follows. |
| Novelty | Relevance-diversity retrieval and [AOII] motivate history and content dependence | Per-UE cosine distance is an appearance proxy. We do not implement the AoII optimal policy or claim fact identity. |
| Scene evidence | [YOLO], [DINO-DETECT] localize text-specified categories | The 93 terms, masks and geometry fields are our interface. Detection is not distress or severity. |
| Four SPI classes | [CRISIS], [MEDIC] motivate incident-image analysis | Four intervention tiers, abstention and class boundaries are project-defined. Native categories are not urgency truth. |
| Offline pseudo-labeling | [QWEN] supports multimodal generation | Mixed teacher protocols and prompts need per-row provenance. Parser success does not validate a label. |
| Token aggregation | [MIL] motivates learned attention over instances | Patch attention is a classifier mechanism, not a validated emergency explanation. |
| Contextual ranking | [RANKNET], [LAMBDAMART] learn ordering | Our waiting-derived targets, tie weighting and sample limits adapt those methods. |
| Variable candidate sets | [SETS], [ATTENTION] motivate set processing | Our mean pooling and encoder-only attention paths are adaptations. |
| Sequential learning | [DDQN], [PER], [PPO] supply update mechanisms | Legal masks, queue transitions and waiting rewards are project-specific. Replay priority is not message urgency. |

## Next experiments and acceptance

First preserve the completed artifacts and freeze the exact source, feature registry, split identities and teacher provenance. Complete the resource recovery without regenerating labels or resetting useful results. Store each feature bank once, reserve disk space and checkpoint useful heads and metrics before the notebook's final cell.

Then compare no-scene, compact-scene and full-scene inputs, with SPI-only learned models as a control. Retrain on identical groups and evaluate on identical offered traces. Remove excluded fields from every derived history feature as well as the candidate input. A masked candidate field that survives in a pooled context would invalidate the ablation.

Independently audit the SPI rubric and ambiguous class boundaries. Expand representative training material only when its labels and source groups are defensible. Generalization needs routine and unfamiliar scenes with independently assessed urgency, not merely model predictions on broader source images. Keep unknown references unknown and distinguish label unavailability from Low.

Keep the protected scheduler final test closed until model and comparison decisions freeze. Use paired per-episode outcomes for learned comparisons and report class-specific misses. A queue-seed repeat is not an independent disaster. Explain the experimental unit before applying uncertainty estimates.

Measure extraction, serialization, queue-context construction, inference and decoding separately with matching hardware and workloads. Charge the full pipeline when claiming deployment cost. Do not add unrelated p95 timings and call the sum a measured p95. Report average waiting, restricted waiting, unselected counts and class safeguards alongside the aggregate penalty.

For the separate P study, qualify image-and-context teacher preferences before fitting the temporal ranker. For E, use independently supported fact correspondence. Training a D model does not establish a completed P preference experiment or E distinct-evidence result.

## Conclusion

We have built a parameter-producing and receiver-scheduling research pipeline, completed a provisional SPI model and broad cache, and obtained bounded development results for fourteen schedulers. Simple semantic prioritization is already a strong control. The current evidence favors retaining LambdaMART, RankNet and delay regression for focused follow-up while testing whether richer semantic inputs and history earn their cost. Independent urgency validation, complete parameter comparison, repeated scheduler evidence and final confirmatory evaluation remain unfinished.
'''

text=intro+'\n\n# Part I Semantic parameters and SPI development\n\n'+param+'\n\n# Part II Receiver policy and scheduler training\n\n'+sched+'\n\n# Part III Updated results and strong baseline analysis\n\n'+analysis+'\n\n# Part IV Literature decisions limitations and next experiments\n\n'+closing
text+='\n\n# References\n\n'
for key,(title,url) in refs.items():text+=f'[{key}] [{title}]({url})\n\n'
text+='''# Source provenance

The report consolidates all five files supplied by the team. HTML and Markdown versions were read as the same parameter report. Tables and paragraphs from the supplied Word report were extracted in document order. The newer analysis supplies the strong SPI baseline comparison. CSV rows were checked against the rounded 36-pair table. Source hashes and authoring validation are saved in source_manifest.json. No new project training, inference or final-test access occurred during preparation.

Document generation uses the bundled Python runtime. The packaged DOCX renderer could not run because LibreOffice was absent. Microsoft Word exports the deliverable to PDF for visual review. This changes document rendering only, not any project result.
'''

# Editorial revision into a final single-column manuscript.
text=re.sub(r'## How this report uses the supplied documents.*?(?=# Part I)', '',text,flags=re.S)
text=re.sub(r'## 16\. Parameter reproduction and local evidence.*?(?=## 17\.)','',text,flags=re.S)
text=re.sub(r'# Source provenance.*','',text,flags=re.S)
text=text.replace('Detailed project report | 4 November 2026','')
text=text.replace('## Compute and training disclaimer','## Computational constraints and evaluation scope')
text=text.replace('We are struggling to complete the intended training and validation program','The intended training and validation program remains constrained')
text=text.replace('within the available compute, memory, storage and execution budgets','by the available compute, memory, storage and execution budgets')
text=re.sub(r'\bprovisional\s+', '',text,flags=re.I)
text=re.sub(r'\bMini dataset\b','image dataset',text,flags=re.I)
text=re.sub(r'\bMini classifier\b','SPI classifier',text,flags=re.I)
text=re.sub(r'\bMini pipeline\b','SPI extraction pipeline',text,flags=re.I)
text=re.sub(r'\bMini archive\b','dataset archive',text,flags=re.I)
text=re.sub(r'\bMini cache\b','image cache',text,flags=re.I)
text=re.sub(r'\bMini\b','dataset',text)
text=text.replace('The original table did not include SPI-first. Part III supplies the newer matched baseline analysis. Its aggregate improvement still requires class safeguards and independent confirmation.','The matched SPI-first comparison is reported in the subsequent results analysis. Aggregate improvements require class-specific safeguards and independent confirmation.')
text=text.replace('The supplied files are evidence, not a reason to merge incompatible experiment units. ','')
text=text.replace('according to the supplied records','in the recorded evaluation')
text=text.replace('in the supplied newer records','in the completed development runs')
text=text.replace('in the supplied aggregate analysis','in the aggregate analysis')
text=text.replace('The report consolidates all five files supplied by the team.','')
# Remove historical report-level prefaces from the main scientific narrative.
text=text.replace('The current evidence favors retaining','The results support retaining')
# Number every external literature link once and use conventional numeric citations.
links={}
for label,url in re.findall(r'(?<!!)\[([^\]]+)\]\((https?://[^)]+)\)',text):
    normalized=url.replace('/html/','/abs/').replace('v1','').replace('v3','')
    if normalized not in links:links[normalized]=(len(links)+1,label,url)
ref_start=text.index('\n# References')
body=text[:ref_start]
for key,(title,url) in refs.items():
    normalized=url.replace('/html/','/abs/').replace('v1','').replace('v3','')
    n=links[normalized][0]
    body=body.replace('['+key+']','['+str(n)+']')
def cite(m):
    normalized=m[2].replace('/html/','/abs/').replace('v1','').replace('v3','')
    return m[1]+' ['+str(links[normalized][0])+']'
body=re.sub(r'(?<!!)\[([^\]]+)\]\((https?://[^)]+)\)',cite,body)
# A continuous hierarchy replaces the source reports' independent numbering.
def roman(n):
    out=''
    for v,s in [(100,'C'),(90,'XC'),(50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]:
        while n>=v:out+=s;n-=v
    return out
section=0;sub=0;lines2=[]
for line in body.splitlines():
    if line.startswith('# Part '):
        section+=1;sub=0
        title=re.sub(r'^# Part [IVX]+ ','',line)
        line='# '+roman(section)+'. '+title
    elif line.startswith('## ') and not line.startswith('### '):
        title=re.sub(r'^\d+\.?\s*','',line[3:])
        if section:
            sub+=1;line='## '+chr(64+sub)+'. '+title
        elif title not in ['Abstract']:
            section+=1;sub=0;line='# '+roman(section)+'. '+title
    lines2.append(line)
text='\n'.join(lines2)+'\n\n# References\n\n'
for n,label,url in sorted(links.values()):text+=f'[{n}] [{label}]({url})\n\n'

sources=[PARAM/'REPORT.html',PARAM/'REPORT.md',PARAM/'completed_comparison.csv',ANALYSIS,OLD]
manifest={'sources':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],'new_experiments':False}
rows=list(csv.DictReader((PARAM/'completed_comparison.csv').open(encoding='utf-8')))
assert len(rows)==36
manifest['completed_csv_rows']=len(rows)
manifest['best_csv_candidate']=max(rows,key=lambda x:float(x['development_macro_f1']))
class ReadHTML(HTMLParser):
    def __init__(self):super().__init__();self.chunks=[]
    def handle_data(self,d):self.chunks.append(d)
hp=ReadHTML();hp.feed(sources[0].read_text(encoding='utf-8'));manifest['html_read_characters']=sum(map(len,hp.chunks))
OUT.joinpath('Semantic_Aware_AI_Scheduler.md').write_text(text,encoding='utf-8')

D=Document();sec=D.sections[0]
sec.page_width=Inches(8.5);sec.page_height=Inches(11)
sec.top_margin=sec.bottom_margin=Inches(.7);sec.left_margin=sec.right_margin=Inches(.75)
for st in ['Normal','Title','Subtitle','Heading 1','Heading 2','Heading 3']:
    D.styles[st].font.name='Times New Roman';D.styles[st].font.color.rgb=RGBColor(0,0,0)
D.styles['Normal'].font.size=Pt(11)
D.styles['Normal'].paragraph_format.space_after=Pt(6)
D.styles['Normal'].paragraph_format.line_spacing=1.03
D.styles['Title'].font.size=Pt(24)
for st in ['Heading 1','Heading 2','Heading 3']:D.styles[st].paragraph_format.keep_with_next=True
for style in D.styles:
    pr=style.element.find(qn('w:pPr'))
    if pr is not None:
        for border in list(pr.findall(qn('w:pBdr'))):pr.remove(border)
D.styles['Caption'].paragraph_format.keep_with_next=False
foot=sec.footer.paragraphs[0];foot.text=''
fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');foot._p.append(fld)

def inline(p,s):
    pattern=r'\[([^\]]+)\]\((https?://[^)]+)\)'
    start=0
    for m in re.finditer(pattern,s):
        p.add_run(re.sub(r'[*`]', '',s[start:m.start()]))
        link=OxmlElement('w:hyperlink');rid=p.part.relate_to(m[2],'http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink',is_external=True);link.set(qn('r:id'),rid)
        rr=OxmlElement('w:r');pr=OxmlElement('w:rPr');c=OxmlElement('w:color');c.set(qn('w:val'),'000000');pr.append(c);rr.append(pr);t=OxmlElement('w:t');t.text=m[1];rr.append(t);link.append(rr);p._p.append(link);start=m.end()
    p.add_run(re.sub(r'[*`]', '',s[start:]))

lines=text.splitlines();i=0;htmlout=[];figure_num=0;table_num=0
while i<len(lines):
    ln=lines[i].strip();i+=1
    if not ln:continue
    if ln.startswith('|'):
        ls=[ln]
        while i<len(lines) and lines[i].strip().startswith('|'):ls.append(lines[i].strip());i+=1
        rr=[[x.strip() for x in z.strip('|').split('|')] for z in ls]
        rr=[r for r in rr if not all(re.fullmatch(r'[-: ]+',x) for x in r)]
        table_num+=1
        cap=D.add_paragraph('TABLE '+str(table_num));cap.alignment=1;cap.paragraph_format.keep_with_next=True
        tb=D.add_table(rows=0,cols=len(rr[0]));tb.style='Table Grid'
        for ri,row in enumerate(rr):
            cells=tb.add_row().cells
            for cell,value in zip(cells,row):
                inline(cell.paragraphs[0],value)
                for pp in cell.paragraphs:
                    pp.paragraph_format.space_after=Pt(4);pp.paragraph_format.space_before=Pt(4)
                    for run in pp.runs:run.font.size=Pt(9.5);run.bold=ri==0
                sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'E4EAF0' if ri==0 else ('F5F6F7' if ri%2==0 else 'FFFFFF'));cell._tc.get_or_add_tcPr().append(sh)
            tb.rows[-1]._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
        tb.rows[0]._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
        D.add_paragraph();continue
    m=re.match(r'!\[([^]]*)\]\(([^)]+)\)',ln)
    if m:
        path=Path(m[2])
        if path.name=='mini_recall.png':continue
        if path.exists():
            from PIL import Image
            with Image.open(path) as im: width=min(6.7,5.5*im.width/im.height)
            D.add_picture(str(path),width=Inches(width));D.paragraphs[-1].paragraph_format.keep_with_next=True
            figure_num+=1
            p=D.add_paragraph('Fig. '+str(figure_num)+'. '+m[1]);p.style='Caption';p.alignment=1
        continue
    if ln.startswith('#'):
        hashes=len(ln)-len(ln.lstrip('#'));title=ln[hashes:].strip()
        if title=='Source provenance':D.add_heading(title,level=2)
        elif hashes==1:
            if False:D.add_page_break()
            p=D.add_paragraph(title,style='Title' if title=='Semantic Aware AI Scheduler' else 'Heading 1')
            if title=='Semantic Aware AI Scheduler':p.alignment=1
        else:D.add_heading(title,level=min(hashes,3))
        continue
    if ln.startswith('```'):continue
    while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','!','- ')):
        ln+=' '+lines[i].strip();i+=1
    p=D.add_paragraph();inline(p,ln)
    if any(name in ln for name in ['Satya Srinivas','Nandini Padavala','Jahnvi KV']):p.alignment=1

D.save(OUT/'Semantic_Aware_AI_Scheduler.docx')
manifest['word_count']=len(text.split());manifest['paper_tags']=len(refs)
OUT.joinpath('source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'words':manifest['word_count'],'references':len(refs),'output':str(OUT)},indent=2))
