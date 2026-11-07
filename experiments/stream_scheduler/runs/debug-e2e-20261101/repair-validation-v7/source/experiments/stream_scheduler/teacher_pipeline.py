"""Build causal candidate queries and independently specified teacher controls."""
import csv,hashlib,io,json,pickle,tarfile,urllib.request
from pathlib import Path
import numpy as np
from Evaluation.semantic_learning_env import Environment,Rule,make_episode
from Scheduler.semantic_learning_policy import fit_normalizer
from Scheduler.training.semantic_learning_models import Sample
from experiments.stream_scheduler.semantic_suite_data import load_pool

LABEL_URL="https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels.csv"
PERSON="/m/01g317"

def prepare(archive,output,text_vectors=None,mission_text=False,vector_extension=None):
    from PIL import Image
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    rows,features,vectors,profile,audit=load_pool(archive,out/"data",vector_extension)
    if mission_text:
        from experiments.stream_scheduler.semantic_suite_data import mission_vectors
        text_vectors=mission_vectors(out/"data")
    annotations=out/"openimages_human_validation.csv"
    if not annotations.exists():
        with urllib.request.urlopen(LABEL_URL,timeout=60) as response,annotations.open("wb") as f:
            total=0
            while True:
                chunk=response.read(1024*1024)
                if not chunk:break
                total+=len(chunk)
                if total>100*1024*1024:raise ValueError("annotation download exceeds preflight cap")
                f.write(chunk)
    support={}
    with annotations.open() as f:
        for r in csv.DictReader(f):
            if r["LabelName"]==PERSON:support[r["ImageID"]]=int(float(r["Confidence"]))
    controls=[r for r in rows if r["dataset"]=="open_images" and r["split"]=="train" and r["source_id"] in support]
    pos=[r for r in controls if support[r["source_id"]]==1]
    neg=[r for r in controls if support[r["source_id"]]==0]
    if len(pos)<8 or len(neg)<8:raise ValueError("insufficient positive/explicit-negative person controls")
    control_ids={r["image_id"] for r in pos[:8]+neg[:8]}
    requests=[];samples={};groups={};used=set(control_ids)
    tar=tarfile.open(archive);index={r["image_id"]:r for r in json.load(tar.extractfile("Mini_dataset/portable_image_index.json"))}
    image_dir=out/"images";image_dir.mkdir(exist_ok=True)
    def image(ident):
        rec=index[ident];path=image_dir/(rec["image_sha256"]+".jpg")
        if not path.exists():
            blob=tar.extractfile("Mini_dataset/"+rec["relative_path"]).read()
            if hashlib.sha256(blob).hexdigest()!=rec["image_sha256"]:raise ValueError("teacher image hash mismatch")
            path.write_bytes(blob)
        return str(path.resolve())
    choice_schema='Return JSON with choice ("A","B","tie","uncertain"), evidence (string), unknowns (list).'
    for i in range(8):
        images=[image(pos[i]["image_id"]),image(neg[i]["image_id"])]
        for swap in (False,True):
            requests.append({"id":"control-strict-"+str(i)+"-"+str(int(swap)),"task":"control","images":images[::-1] if swap else images,"roles":["Candidate A","Candidate B"],"prompt":"Mission: find visible people. Which image provides stronger evidence that a person is visibly present? Do not infer urgency. "+choice_schema,"expected":"B" if swap else "A","reference":{"url":LABEL_URL,"class":PERSON,"positive":pos[i]["source_id"],"explicit_negative":neg[i]["source_id"]}})
    blank=image_dir/"blank.png";Image.new("RGB",(224,224),"gray").save(blank)
    for i in range(8):
        img=image(pos[i]["image_id"])
        requests.append({"id":"control-tie-"+str(i),"task":"control","images":[img,img],"roles":["Candidate A; same time/history","Candidate B; same time/history"],"prompt":"Identical reports under identical context. Which should be selected on semantic grounds? "+choice_schema,"expected":"tie","reference":{"construction":"identical image and context"}})
        requests.append({"id":"control-unknown-"+str(i),"task":"control","images":[str(blank.resolve()),str(blank.resolve())],"roles":["Candidate A","Candidate B"],"prompt":"Which image establishes a building is about to collapse? If neither image permits that judgment, answer uncertain. "+choice_schema,"expected":"uncertain","reference":{"construction":"blank images; no building evidence"}})
    # The controls also exercise one-, two- and ten-image inference inside the existing32-call budget.
    for request in requests:
        if request["id"]=="control-unknown-0":
            request["images"]=request["images"][:1];request["roles"]=request["roles"][:1]
        if request["id"]=="control-tie-0":
            request["images"]+=request["images"]*4
            request["roles"]+=["Identical shared context; same history for A and B"]*8
    training_episodes=[]
    for role,n in (("train",28),("validation",12)):
        cohort=[r for r in rows if r["split"]==role and r["image_id"] not in used]
        for j in range(n):
            # Disjoint images per context prevent history components crossing roles or queries.
            chosen=cohort[j*12:(j+1)*12]
            if len(chosen)!=12:raise ValueError("insufficient disjoint teacher-context groups")
            ep=make_episode(chosen,features,vectors,text_vectors,count=12,ues=4,load=1.2,seed=900000+len(samples),repeat="none")
            training_episodes.append(ep) if role=="train" else None
            env=Environment(ep,"P");rule=Rule("random",j)
            for _ in range(3):
                if env.done:break
                env.step(rule.select(env.snapshot()))
            snap=env.snapshot()
            if len(snap.ready)<2:raise ValueError("teacher query lacks two ready candidates")
            snap=snap.pair_view([m.token for m in snap.ready[:2]])
            images=[];roles=[]
            for label,m,window in zip(("A","B"),snap.ready,snap.ue_windows):
                images.append(image(ep.metadata["assets"][m.token]))
                mission=ep.metadata["missions"][ep.metadata["profiles"][m.source]][1]
                roles.append("Candidate "+label+"; wait="+str(snap.now-m.arrival)+"; UE mission: "+mission)
                for h in window:
                    images.append(image(ep.metadata["assets"][h.token]));roles.append("Source-history-window observation for "+label+"; receiver arrival="+str(h.arrival)+"; decision time="+str(snap.now)+"; status="+("waiting" if h.token in snap.waiting_tokens else "inflight" if h.token in snap.in_flight_tokens else "processed"))
            for m,finish in snap.processed:
                images.append(image(ep.metadata["assets"][m.token]));roles.append("Previously processed source evidence; finish="+str(finish)+"; not verified reconstruction")
            if len(images)>10:raise ValueError("teacher context exceeds10 images")
            qid=role+"-"+str(j)
            samples[qid]=Sample(snap,np.zeros(2),np.full(2,-1),qid,target_source="qualified_Qwen_candidate_preferences")
            groups[qid]=role
            prompt="Receiver objective: early supported time-sensitive incidents and mission-useful additional visual evidence; avoid redundant reports. Select which CURRENT candidate A or B should be processed next, using only shown context. "+choice_schema+" Also rate four aspects for A and B: mission_usefulness, evidence_clarity, additional_information, condition_specificity. Each rating is a string0..4 or unknown. Follow the appended JSON schema exactly."
            for swap in (False,True):
                oriented=[x.replace("Candidate A","Candidate TEMP").replace("Candidate B","Candidate A").replace("Candidate TEMP","Candidate B").replace("for A;","for TEMP;").replace("for B;","for A;").replace("for TEMP;","for B;") if swap else x for x in roles]
                positions=list(range(len(images)))
                if swap:
                    second=next(k for k,r in enumerate(roles) if r.startswith("Candidate B"))
                    processed=next((k for k,r in enumerate(roles) if r.startswith("Previously processed")),len(roles))
                    positions=list(range(second,processed))+list(range(second))+list(range(processed,len(roles)))
                requests.append({"id":qid+"-"+str(int(swap)),"task":"candidate","context_id":qid,"split":role,"swapped":swap,"images":[images[k] for k in positions],"roles":[oriented[k] for k in positions],"prompt":prompt})
            used.update(r["image_id"] for r in chosen)
    from Scheduler.training.preference_reward import transition_path
    pre=fit_normalizer(training_episodes);prefixes={};replay_contexts={}
    for role,n in (("train",28),("validation",12)):
        cohort=[r for r in rows if r["split"]==role and r["image_id"] not in used]
        for j in range(n):
            chosen=cohort[j*12:(j+1)*12]
            if len(chosen)!=12:raise ValueError("insufficient disjoint prefix contexts")
            ep=make_episode(chosen,features,vectors,text_vectors,count=12,ues=4,load=1.2,seed=950000+len(prefixes),repeat="none")
            left=Environment(ep,"P");right=Environment(ep,"P");initial=left.snapshot()
            plans=[[],[]];paths=[[],[]]
            firsts=(initial.ready[0].token,initial.ready[2].token)
            for side,env in enumerate((left,right)):
                for step in range(2):
                    state=env.snapshot()
                    token=firsts[side] if step==0 else state.ready[0 if side==0 else min(1,len(state.ready)-1)].token
                    nxt,_,_=env.step(token)
                    paths[side].append(transition_path(state,token,env,pre,end=min(2.,env.now)))
                    plans[side].append({"selected":token,"dispatch":state.now,"finish":min(2.,env.now)})
            if left.now!=2 or right.now!=2:raise ValueError("prefix comparison endpoints differ")
            visible=[m for m in ep.messages if m.arrival<=2]
            if len(visible)>10:raise ValueError("prefix visible envelope exceeds10; do not truncate")
            images=[image(ep.metadata["assets"][m.token]) for m in visible]
            roles_text=["Evidence "+str(m.token)+"; complete arrival="+str(m.arrival)+"; mission="+ep.metadata["missions"][ep.metadata["profiles"][m.source]][1] for m in visible]
            qid="prefix-"+role+"-"+str(j)
            prefixes[qid]={"role":role,"left":paths[0],"right":paths[1],"start":0.,"end":2.,"plans":plans}
            for swap in (False,True):
                shown=plans[::-1] if swap else plans
                prompt="Both schedules start with the same available reports and finish at time2. A="+json.dumps(shown[0])+" B="+json.dumps(shown[1])+". Which exposes supported time-sensitive, mission-useful additional evidence earlier? Unselected evidence remains waiting. Do not reward picture count; both attempted two reports. "+choice_schema
                requests.append({"id":qid+"-"+str(int(swap)),"context_id":qid,"task":"prefix","split":role,"swapped":swap,"images":images,"roles":roles_text,"prompt":prompt})
            if role=="train" and j<24:
                replay_contexts[qid]=Sample(initial,np.zeros(len(initial.ready)),np.full(len(initial.ready),-1),qid,target_source="Qwen_replay_training_value")
                requests.append({"id":"replay-"+str(j),"task":"replay","context_id":qid,"split":"train","images":images,"roles":roles_text,"prompt":"Training experience: "+json.dumps(plans[0])+". Judge usefulness of this experience for learning to choose mission-useful/time-sensitive visual reports: high for a clear consequential selection error or informative conflict; neutral for ambiguous/ordinary experience; low for uninformative repetition. This is replay sampling guidance, not image urgency. Return JSON with value (low,neutral,high,uncertain), evidence(string), unknowns(list)."})
            used.update(r["image_id"] for r in chosen)
    with (out/"prefix_contexts.pickle").open("wb") as f:pickle.dump({"prefixes":prefixes,"replay":replay_contexts,"preprocessing":pre},f)
    tar.close()
    (out/"requests.jsonl").write_text("".join(json.dumps(r)+chr(10) for r in requests))
    with (out/"contexts.pickle").open("wb") as f:pickle.dump({"samples":samples,"roles":groups,"preprocessing":fit_normalizer(training_episodes),"input_profile":profile},f)
    (out/"manifest.json").write_text(json.dumps({"scope":"32 controls +80 candidate +80 prefix +24 replay =216 calls; no final-test teacher queries","calls":len(requests),"annotation_url":LABEL_URL,"annotation_sha256":hashlib.sha256(annotations.read_bytes()).hexdigest(),"controls":32,"profile":profile,"test_labels_unopened":True},indent=2))
    return out/"requests.jsonl"

def qualify(request_path,response_path):
    queries={r["id"]:r for r in map(json.loads,Path(request_path).read_text().splitlines())}
    answers={r["id"]:r for r in map(json.loads,Path(response_path).read_text().splitlines())}
    control=[q for q in queries.values() if q["task"]=="control"]
    if len(control)!=32:raise ValueError("all32 predefined controls required")
    valid=sum(answers.get(q["id"],{}).get("status")=="parsed" for q in control)/32
    accuracy={}
    for category in ("strict","tie","unknown"):
        subset=[q for q in control if q["id"].startswith("control-"+category+"-")]
        accuracy[category]=sum(answers.get(q["id"],{}).get("parsed",{}).get("choice")==q["expected"] for q in subset)/len(subset)
    return {"qualified":valid>=.95 and accuracy["strict"]>=.8 and accuracy["tie"]>=.75 and accuracy["unknown"]>=.75,"schema_valid_fraction":valid,"accuracy":accuracy,"scope":"visible-person relevance/identical/blank controls; not emergency-priority ground truth"}


def accepted_samples(prepared,teacher_output):
    prepared=Path(prepared);teacher_output=Path(teacher_output)
    gate=qualify(prepared/"requests.jsonl",teacher_output/"responses.jsonl")
    (teacher_output/"qualification.json").write_text(json.dumps(gate,indent=2))
    if not gate["qualified"]:raise ValueError("Teacher failed task-specific qualification: "+json.dumps(gate))
    with (prepared/"contexts.pickle").open("rb") as f:cache=pickle.load(f)
    responses={r["id"]:r for r in map(json.loads,(teacher_output/"responses.jsonl").read_text().splitlines())}
    roles={r:[] for r in ("train","validation","test")};aspects={r:[] for r in roles}
    opposite={"A":"B","B":"A","tie":"tie","uncertain":"uncertain"}
    rejected=[]
    for ident,sample in cache["samples"].items():
        if cache["roles"][ident]=="test":continue  # Locked teacher test is not used in model selection.
        a=responses.get(ident+"-0",{}).get("parsed",{});b=responses.get(ident+"-1",{}).get("parsed",{})
        x=a.get("choice");y=b.get("choice")
        if x not in opposite or y!=opposite[x] or x=="uncertain":
            rejected.append(ident);continue
        target={"A":1.,"B":0.,"tie":.5}[x]
        role=cache["roles"][ident]
        roles[role].append(Sample(sample.snapshot,sample.target,sample.tiers,ident,((0,1,target),),"qualified_Qwen_candidate_preferences"))
        ra=a.get("aspects",{});rb=b.get("aspects",{})
        if not isinstance(ra,dict) or not isinstance(rb,dict):continue
        def good(v):return isinstance(v,list) and len(v)==4 and all(type(x)==int and 0<=x<=4 for x in v)
        if not all(good(r.get(k)) for r in (ra,rb) for k in ("A","B")):continue
        da=np.mean(ra["A"])-np.mean(ra["B"]);db=np.mean(rb["B"])-np.mean(rb["A"])
        if np.sign(da)!=np.sign(db):continue
        at=1. if da>0 else 0. if da<0 else .5
        aspects[role].append(Sample(sample.snapshot,sample.target,sample.tiers,ident,((0,1,at),),"qualified_Qwen_four_aspect_preferences"))
    report={"accepted":{k:len(v) for k,v in roles.items()},"aspect_accepted":{k:len(v) for k,v in aspects.items()},"rejected":rejected,"scope":"silver teacher agreement, not independent urgency","gate":gate}
    (teacher_output/"accepted_label_report.json").write_text(json.dumps(report,indent=2))
    return cache,roles,aspects
