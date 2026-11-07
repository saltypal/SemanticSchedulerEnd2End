import sys
from pathlib import Path
import numpy as np
ROOT=Path("/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End")
sys.path.insert(0,str(ROOT))
import project_compat; project_compat.install()
from Evaluation.semantic_learning_env import Message,Snapshot
from Scheduler.semantic_learning_policy import NumpyRanker,digest
def test_tree_threshold_equality_and_adjacent_float():
    pre={"profile_identity":{"sha256":"fixture"},"width":387,"continuous":[0],"median":[0.]*387,"mean":[0.]*387,"scale":[1.]*387}
    tree={"split_feature":1,"threshold":0.5,"decision_type":"<=","default_left":True,"left_child":{"leaf_value":1.0},"right_child":{"leaf_value":2.0}}
    a={"schema":"semantic_scene_ranker_v1","kind":"trees","arm":"lambdamart","preprocessing":pre,"input_profile":pre["profile_identity"],"objective_version":"D_v1","reward_version":"none_supervised","trees":[tree]}
    a["sha256"]=digest(a); policy=NumpyRanker(a)
    def score(v):
        x=np.zeros(387,np.float32); x[1]=v
        m=Message(1,1,1,0.,1,x,profile_id="fixture")
        s=Snapshot((m,),(),((),),(),0.,0)
        return policy.raw(s)[0]
    assert score(0.5)==1.0
    assert score(float(np.nextafter(np.float32(0.5),np.float32(1))))==2.0
