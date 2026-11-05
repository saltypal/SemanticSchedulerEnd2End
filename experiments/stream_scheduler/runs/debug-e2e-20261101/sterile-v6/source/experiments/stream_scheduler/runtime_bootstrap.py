"""Isolated training namespace: reuse canonical modules without optional vision imports."""
from pathlib import Path
import sys
import types


def install(root=None):
    root=Path(root or Path(__file__).resolve().parents[2])
    for name in ('Evaluation','Scheduler','Scheduler.training'):
        if name not in sys.modules:
            module=types.ModuleType(name);module.__path__=[str(root.joinpath(*name.split('.')))];sys.modules[name]=module
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    return root
