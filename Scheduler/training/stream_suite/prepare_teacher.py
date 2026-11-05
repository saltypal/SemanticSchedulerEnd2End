import argparse,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
s=importlib.util.spec_from_file_location("boot",Path(__file__).with_name("runtime_bootstrap.py"));b=importlib.util.module_from_spec(s);s.loader.exec_module(b);b.install(ROOT)
from Scheduler.training.stream_suite.teacher_pipeline import prepare
p=argparse.ArgumentParser();p.add_argument("--archive",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--mission-text",action="store_true")
p.add_argument("--vector-extension",type=Path)
p.add_argument("--mission-vectors",type=Path)
a=p.parse_args()
from Scheduler.training.stream_suite.semantic_suite_data import load_mission_vectors
text_vectors=load_mission_vectors(a.mission_vectors) if a.mission_vectors else None
print("START teacher request preparation",a.output,flush=True);print("DONE",prepare(a.archive,a.output,text_vectors=text_vectors,mission_text=a.mission_text,vector_extension=a.vector_extension),flush=True)
