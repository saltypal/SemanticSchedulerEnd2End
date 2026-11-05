"""Create a local notebook from measured pilot artifacts; never refit or infer silently."""
from pathlib import Path
import argparse
import nbformat as nbf


def build(root, source, destination):
    root=Path(root).resolve();source=Path(source).resolve()
    notebook=nbf.v4.new_notebook()
    cells=[nbf.v4.new_markdown_cell('''# parameters: measured laptop walkthrough

This notebook reads the real 180-parent CPU pilot. It shows predictions, causal
sender memory, a serialized header, queue decisions and independent annotation
coverage. It does **not** run a codec, establish natural cross-view identity, or
turn the exposed development comparison into a final test.

Run from the project Python environment at
`../SemanticSchedulerEnd2End/.venv/Scripts/python.exe`. Dependencies are the
existing project environment plus nbformat, nbclient, matplotlib and pandas.
Source modules live in parameters; the recorded run manifest identifies
the snapshot actually used for the pilot. Current replay code can differ.
Outputs: this executed notebook, sibling cell logs, and `walkthrough_figures/`.
No network access or new model inference is required.''')]
    def code(value):cells.append(nbf.v4.new_code_cell(value))
    def text(value):cells.append(nbf.v4.new_markdown_cell(value))
    code(f'''print("Setup: environment, repository and measured source run", flush=True)
import sys, json, platform, subprocess, time
from pathlib import Path
import importlib.metadata as metadata
ROOT=Path({str(root)!r}); SOURCE=Path({str(source)!r})
sys.path.insert(0,str(ROOT))
print("Actual interpreter:",sys.executable,"Python:",platform.python_version())
print("Repository:",ROOT,"Source run:",SOURCE)
for package in ("numpy","torch","transformers","scikit-learn","nbformat"):
    print(package,metadata.version(package))
print("Current revision:",subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())
manifest=json.loads((SOURCE/"manifest.json").read_text())
print("Pilot revision:",manifest.get("git_commit"),"Pilot Python:",manifest.get("python"))
import torch
print("Current CUDA available:",torch.cuda.is_available(),"Notebook uses cached CPU evidence only")
FIGURES=ROOT/"parameters"/"notebooks"/"walkthrough_figures"
FIGURES.mkdir(parents=True,exist_ok=True)
print("Setup complete; figure destination:",FIGURES)''')
    text('''## 1. Separate evidence from truth

The source image supplies frozen image features. Mission clauses supply text
features. The runtime extractor sees those predictions and past sender memory.
Published labels are used only by training/evaluation, never by extraction.
Missing labels are unknown, not negatives. These sources lack verified events
and capture times; no event-held-out or freshness claim follows.''')
    code('''print("Stage 1: audit annotations and cached prediction schema",flush=True)
import pandas as pd
records=json.loads((SOURCE/"data/records.json").read_text())
print("Parents:",len(records),"Groups:",len({r["group"] for r in records}))
print(pd.Series([r["source"] for r in records]).value_counts())
display(pd.DataFrame([r["labels"] for r in records]).apply(lambda x:x.value_counts(dropna=False)))
from parameters.extractor import VisualEvidence,SemanticParameterExtractor,SenderMemory
raw={}
for line in (SOURCE/"features/clip.jsonl").read_text().splitlines():
    row=json.loads(line);raw[row["image_id"]]=VisualEvidence.from_dict(row["evidence"])
print("Prediction record attributes:",list(next(iter(raw.values())).to_dict()))
print("Stage 1 complete")''')
    text('''## 2. Query memory, extract, serialize, then commit

The first observation has cold-start novelty. The same evidence after a
successful transmission should have zero embedding distance. Semantic-delta
novelty can also use changes in predicted mission conditions; it is not proof
that two photographs show the same physical asset. Memory is transmitter-side.
The receiver receives numerical fields, not the sender's whole knowledge base.''')
    code('''print("Stage 2: frozen heads and causal duplicate example",flush=True)
from parameters.contracts import MISSIONS,MethodSpecification
from parameters.learning import ClaimHeads
from parameters.serialization import HeaderCodec
directory=SOURCE/"metrics/clip"
spec=MethodSpecification(**json.loads((directory/"building_damage_specification.json").read_text()))
heads=ClaimHeads.from_dict(json.loads((directory/"claim_heads.json").read_text()))
enriched=heads.apply(raw)
extractor=SemanticParameterExtractor(spec)
memory=SenderMemory("example-sender","building_damage",capacity=spec.kb_capacity)
evidence=enriched[records[0]["image_id"]]
codec=HeaderCodec(spec.feature_names,spec.sha256,spec.quantization_bits)
for sequence in range(2):
    bundle=extractor.extract_evidence(evidence,MISSIONS["building_damage"],
        {"source_id":"example-sender","sequence":sequence,"image_token":str(sequence)},memory)
    header=codec.encode(bundle)
    print("Observation",sequence,"KB version before commit:",memory.version)
    print("Novelty:",bundle.parameters["novelty"].value)
    print("Header bytes:",len(header),"decoded fields:",codec.decode(header)["values"])
    memory.commit(evidence,sequence)
print("Selected fields:",spec.feature_names)
print("Stage 2 loaded saved producer/head/spec artifacts; no fitting")''')
    text('''## 3. Queue evaluation and why early evidence matters

For each successful decode opportunity, distinct annotated fact coverage is
`C_k = exposed fact weight / offered fact weight`. `A10` averages `C_1` through
`C_10`, so earlier useful evidence receives more credit. An exact repeat adds no
new atom. Native image atoms measure distinct annotated observations, not
distinct real-world buildings. The cached pilot assumes successful service;
codec reconstruction and decoder failure experiments are separate work.''')
    code('''print("Stage 3: independent numerical metric reference",flush=True)
from Evaluation.decoder_semantic_benchmark import coverage_metrics
facts={"A":{"building-1"},"B":{"building-1"},"C":{"building-2"}}
weights={"building-1":1.,"building-2":1.}
for order in (["A","B","C"],["A","C","B"]):
    result=coverage_metrics(order,facts,weights,k=3)
    print(order,result)
print("Hand calculation: repeat-first coverage .5,.5,1 => mean 2/3; diverse-first .5,1,1 => mean 5/6")
print("Stage 3 complete")''')
    text('''## 4. Actual provisional results

These are development-selected values on a single grouped split. They are
optimistic selection results, not locked test effects. Only four MEDIC parents
carry positive affected-people labels, so that stress result is especially weak.
Spatial evidence, useful task complexity, geography and trusted temporal
metadata are not validated by their absence in this pilot.''')
    code('''print("Stage 4: measured comparisons and costs",flush=True)
import matplotlib.pyplot as plt
rows=[]
for backbone in ("clip","siglip"):
    results=json.loads((SOURCE/"metrics"/backbone/"summary.json").read_text())
    for mission,result in results.items():
        chosen=result.get("selected")
        if chosen:
            rows.append({"backbone":backbone,"mission":mission,"A10":chosen["mean_A10"],
                         "groups":str(chosen["groups"]),"header_bytes":chosen["header_bytes"]})
table=pd.DataFrame(rows);display(table)
fig,ax=plt.subplots(figsize=(9,4))
table.pivot(index="mission",columns="backbone",values="A10").plot.bar(ax=ax)
ax.set_ylim(0,1);ax.set_ylabel("Development-selected A10");ax.set_title("Provisional cached CPU pilot; not final evaluation")
fig.tight_layout();fig.savefig(FIGURES/"development_A10.png",dpi=160);plt.show()
print((SOURCE/"PILOT_REPORT.md").read_text())
print("Stage 4 complete; figure saved")''')
    text('''## 5. What remains before freezing a deployable specification

Run the corrected serialized-header replay, verify saved-head single/batch CLI
behaviour, then nested grouped selection on fresh supported data. Compare strong
baselines with explicit information profiles. Any unavailable parameter remains
unresolved. Two-GPU ownership and numerical parity must pass before a cloud
extraction campaign. A final test is opened only after selection is frozen.''')
    code('''print("Walkthrough complete. This notebook inspected cached evidence only.")
print("Measured source:",SOURCE)
print("Figures:",FIGURES)
print("See parameters/PLAN.md and KAGGLE_RUNS.md for current execution state.")''')
    notebook.cells=cells
    notebook.metadata.kernelspec={"display_name":"parameters project Python","language":"python","name":"python3"}
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    nbf.write(notebook,destination)
    return destination


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(build(Path(__file__).resolve().parents[1],args.source,args.output))
