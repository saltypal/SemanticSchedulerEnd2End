import numpy as np
import torch
from parameters.spi_model_selection import mlp_fit

def test_epoch_logging_preserves_fitted_weights():
 x=np.tile(np.eye(4,dtype=np.float32),(4,1));y=np.tile(np.arange(4),4)
 reference,_=mlp_fit(x,y,True,17,epochs=2)
 history=[];logged,_=mlp_fit(x,y,True,17,epochs=2,history=history)
 assert len(history)==2
 for a,b in zip(reference.parameters(),logged.parameters()):assert torch.equal(a,b)


def test_existing_run_is_preserved_and_budget_is_bounded(tmp_path):
 import pytest
 from experiments.spi_classifier_repair.cnn_compare import run
 out=tmp_path/'existing';out.mkdir();(out/'status.json').write_text('preserve me')
 with pytest.raises(FileExistsError):run([],[],out)
 assert (out/'status.json').read_text()=='preserve me'
 assert not (out/'traceback.txt').exists()
 with pytest.raises(ValueError,match='budget'):run([],[],tmp_path/'bad',seconds=1501)
 assert not (tmp_path/'bad').exists()
