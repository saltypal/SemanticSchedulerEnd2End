"""Exact-snapshot discovery across Kaggle's nested input mount layouts."""
import hashlib
from pathlib import Path
import pytest
from parameters.dataset_pipeline import discover_input_root

def make_input(folder):
 folder.mkdir(parents=True)
 (folder/'input_manifest.json').write_bytes(b'verified snapshot')
 (folder/'panel.json').write_text('[]')
 (folder/'features.npz').write_bytes(b'fixture')
 return hashlib.sha256(b'verified snapshot').hexdigest()

def test_nested_kaggle_mount_resolves_exact_snapshot(tmp_path):
 folder=tmp_path/'datasets'/'owner'/'slug'/'versions'/'1'
 digest=make_input(folder)
 assert discover_input_root(tmp_path,digest)==folder.resolve()

def test_ambiguous_mount_rejected(tmp_path):
 digest=make_input(tmp_path/'first');make_input(tmp_path/'second')
 with pytest.raises(ValueError,match='found 2'):discover_input_root(tmp_path,digest)

def test_wrong_snapshot_and_incomplete_snapshot_rejected(tmp_path):
 digest=make_input(tmp_path/'input')
 with pytest.raises(ValueError,match='found 0'):discover_input_root(tmp_path,'a'*64)
 (tmp_path/'input'/'features.npz').unlink()
 with pytest.raises(ValueError,match='incomplete'):discover_input_root(tmp_path,digest)
