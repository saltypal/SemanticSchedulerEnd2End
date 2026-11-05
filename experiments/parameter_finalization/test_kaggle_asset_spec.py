"""Regression checks for the exact emitted Kaggle asset-verification code."""
import ast
import hashlib
import json
from pathlib import Path
import nbformat

ROOT=Path(__file__).resolve().parents[2]
PACKAGE=ROOT/'experiments/parameter_finalization/runs/notebook9_20261028/submission_v10_fixed'

def test_emitted_tokenizer_asset_checks_match_verified_local_bytes():
    notebook=nbformat.read(PACKAGE/'notebook.ipynb',as_version=4)
    tree=ast.parse(notebook.cells[2].source)
    preflight=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='DEPENDENCY_PREFLIGHT')
    declaration=next(n for n in ast.parse(preflight).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='REMOTE_CLIP_FILES')
    env={'SOURCE':json.loads((PACKAGE/'source_manifest.json').read_text())}
    exec(compile(ast.Module(body=[declaration],type_ignores=[]),'emitted_asset_spec','exec'),env)
    for name,(size,digest) in env['REMOTE_CLIP_FILES'].items():
        raw=(ROOT/'models/priority_classifiers/clip'/name).read_bytes()
        assert len(digest)==64
        assert size==len(raw)
        assert hashlib.sha256(raw).hexdigest()==digest

def test_packaged_notebook_schema_and_python_compile():
    notebook=nbformat.read(PACKAGE/'notebook.ipynb',as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type=='code':compile(cell.source,'packaged_notebook','exec')
    assert (PACKAGE/'notebook.ipynb').stat().st_size<900000
