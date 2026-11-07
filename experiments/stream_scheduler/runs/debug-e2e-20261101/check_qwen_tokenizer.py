import json, torch
from transformers import AutoTokenizer
from experiments.stream_scheduler.teacher_output import prefix_constraint, ASPECT_NAMES
revision = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", revision=revision)
results = {}
for task in ("prefix","candidate","replay","tier"):
    row = {"choice":"tie","evidence":"Visible evidence.","unknowns":[]}
    if task=="candidate":row["aspects"]={"A":dict(zip(ASPECT_NAMES,("0","1","2","3"))),"B":dict.fromkeys(ASPECT_NAMES,"unknown")}
    if task=="replay":row={"value":"neutral","evidence":"Visible evidence.","unknowns":[]}
    if task=="tier":row={"value":"Low","evidence":"Visible evidence.","unknowns":[]}
    ids=tokenizer.encode("Teacher prompt: ",add_special_tokens=False)
    fn=prefix_constraint(tokenizer,task)
    fn(0,torch.tensor(ids))
    target=json.dumps(row);generated=[];prefix=""
    while prefix!=target:
        allowed=fn(0,torch.tensor(ids+generated))
        choices=[]
        for token in allowed:
            if token==tokenizer.eos_token_id:continue
            decoded=tokenizer.decode(generated+[token],skip_special_tokens=True)
            if len(decoded)>len(prefix) and target.startswith(decoded):choices.append((len(decoded),token,decoded))
        assert choices,(task,prefix)
        _,token,prefix=max(choices);generated.append(token)
    assert tokenizer.eos_token_id in fn(0,torch.tensor(ids+generated))
    results[task]={"tokens_checked":len(generated),"schema_answer_reachable_and_eos":True}
print(json.dumps({"official_tokenizer":tokenizer.name_or_path,"revision":revision,
                  "results":results,"scope":"CPU token-filter check; no VLM weights or inference"}),flush=True)
