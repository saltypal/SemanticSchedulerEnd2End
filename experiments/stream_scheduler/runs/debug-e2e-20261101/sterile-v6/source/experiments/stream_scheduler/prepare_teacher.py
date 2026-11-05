import argparse,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
s=importlib.util.spec_from_file_location("boot",Path(__file__).with_name("runtime_bootstrap.py"));b=importlib.util.module_from_spec(s);s.loader.exec_module(b);b.install(ROOT)
from experiments.stream_scheduler.teacher_pipeline import prepare
p=argparse.ArgumentParser();p.add_argument("--archive",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--mission-text",action="store_true")
p.add_argument("--vector-extension",type=Path)
a=p.parse_args();print("START teacher request preparation",a.output,flush=True);print("DONE",prepare(a.archive,a.output,mission_text=a.mission_text,vector_extension=a.vector_extension),flush=True)
