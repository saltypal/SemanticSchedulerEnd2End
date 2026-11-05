"""A new producer must create new vectors, never rename an old feature space."""
import json
import numpy as np
from experiments.parameter_finalization.test_dataset_pipeline import make_dummy_inputs,DummyProvider
from parameters.dataset_pipeline import extract_runtime_features
from parameters.contracts import MissionSpec
from parameters.extractor import embedding_signature,VisualEvidence

def test_runtime_features_preserve_identity_and_new_producer(tmp_path):
 records=make_dummy_inputs(tmp_path/'input')
 mission=MissionSpec('test','Look.',(('visible','image'),))
 path=extract_runtime_features(records,DummyProvider(),mission,tmp_path/'new',batch_size=3)
 with np.load(path,allow_pickle=False) as cache:
  assert cache['image_ids'].tolist()==[r['image_id'] for r in records]
  assert cache['image_sha256'].tolist()==[r['image_sha256'] for r in records]
  assert cache['embeddings'].shape==(8,512)
  assert str(cache['feature_signature'].item())==embedding_signature(VisualEvidence(provenance={'backbone':'synthetic_fixture'}))
 assert json.loads((tmp_path/'new/provenance.json').read_text())['image_count']==8
 assert (tmp_path/'input/model.json').exists()
