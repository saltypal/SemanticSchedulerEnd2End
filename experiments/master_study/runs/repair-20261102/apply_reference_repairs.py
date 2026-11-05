from pathlib import Path
root=Path.cwd()
# Task reliability is separate from easy control qualification.
p=root/"experiments/stream_scheduler/teacher_pipeline.py";t=p.read_text()
pos=t.index("\ndef accepted_samples")
t=t[:pos]+'''
def preference_reliability(request_path,response_path,task="candidate"):
    """Orientation agreement is necessary, never a correctness certificate."""
    queries=[json.loads(x) for x in Path(request_path).read_text().splitlines() if x.strip()]
    answers={a["id"]:a for a in map(json.loads,Path(response_path).read_text().splitlines())}
    opposite={"A":"B","B":"A","tie":"tie","uncertain":"uncertain"}
    pairs=[q for q in queries if q.get("task")==task and not q.get("swapped",False)]
    comparable=consistent=strict=0
    for q in pairs:
        ident=q["context_id"];a=answers.get(ident+"-0",{});b=answers.get(ident+"-1",{})
        if a.get("status")!="parsed" or b.get("status")!="parsed":continue
        comparable+=1;x=a.get("parsed",{}).get("choice");y=b.get("parsed",{}).get("choice")
        if x in opposite and y==opposite[x]:
            consistent+=1;strict+=x in ("A","B")
    fraction=consistent/len(pairs) if pairs else 0.
    return {"contexts":len(pairs),"parsed_pairs":comparable,"consistent_pairs":consistent,
            "strict_consistent":strict,"agreement":fraction,
            "qualified":len(pairs)>=8 and fraction>=.8,
            "scope":"presentation consistency only; independent semantic correctness still required"}

''' + t[pos:]
t=t.replace('if not gate["qualified"]:raise ValueError("Teacher failed task-specific qualification: "+json.dumps(gate))',
'''if not gate["qualified"]:raise ValueError("Teacher failed visual controls: "+json.dumps(gate))
    task_gate=preference_reliability(prepared/"requests.jsonl",teacher_output/"responses.jsonl")
    (teacher_output/"preference_reliability.json").write_text(json.dumps(task_gate,indent=2))
    if not task_gate["qualified"]:raise ValueError("Teacher failed scheduling preference consistency: "+json.dumps(task_gate))''')
t=t.replace('Also rate four aspects for A and B: mission_usefulness, evidence_clarity, additional_information, condition_specificity. Each rating is a string0..4 or unknown.',
'''Bind each image to its explicit candidate label before comparing. Compare CURRENT A and CURRENT B; history images are context, not candidates. Give concise evidence for both candidates and the consequential difference. If unsupported or equally useful, abstain or tie. Leave the optional aspects unknown; do not mix aspect grading with this choice task.''')
p.write_text(t)
# Give semantic evidence its own independent reference weights, never copy predicted R/N.
p=root/"Evaluation/semantic_learning_env.py";t=p.read_text()
t=t.replace("    metadata:dict\n","    metadata:dict\n    evidence_weights:tuple|None=None # evaluator-only published/controlled utility, never predicted fields\n")
t=t.replace("        n=len(self.messages)\n","""        n=len(self.messages)
        if self.evidence_weights is not None:
            if len(self.evidence_weights)!=n or any(not math.isfinite(w) or w<0 or w>1 for w in self.evidence_weights):
                raise ValueError('independent evidence weights must be finite0..1 for every offer')
            per_fact={}
            for fact,w in zip(self.fact_ids,self.evidence_weights):
                if fact in per_fact and per_fact[fact]!=w:raise ValueError('conflicting utility for one fact')
                per_fact[fact]=w
""",1)
t=t.replace("return float(sum(max(0.,min(t,first.get(f,float('inf')))-a) for f,a in offered.items()))",
"""weights={fact:1. if ep.evidence_weights is None else ep.evidence_weights[i] for i,fact in enumerate(ep.fact_ids)}
        return float(sum(weights[f]*max(0.,min(t,first.get(f,float('inf')))-a) for f,a in offered.items()))""")
t=t.replace("return np.array([float(self.episode.fact_ids[self.index[m.token]] not in self.covered) for m in s.ready])",
"""return np.array([(1. if self.episode.evidence_weights is None else self.episode.evidence_weights[self.index[m.token]])*
                float(self.episode.fact_ids[self.index[m.token]] not in self.covered) for m in s.ready])""")
p.write_text(t)
# An evidence objective is an explicit alternative, not an implicit change of D.
p=root/"experiments/stream_scheduler/train_real_suite.py";t=p.read_text()
t=t.replace("def evaluate(policy,episodes,output=None,probes=None):","def evaluate(policy,episodes,output=None,probes=None,task='D'):")
t=t.replace("env=Environment(episode,'D');latency=[]","env=Environment(episode,task);latency=[]")
t=t.replace("return -float(np.mean([r['J_D'] for r in results])),results",
"return -float(np.mean([r['J_D'] if task=='D' else r['L_E'] for r in results])),results")
p.write_text(t)
print("Reference weights, teacher reliability gates, and separate evidence evaluation added.")

