"""Offline local pilot: use the explicit checksum-verified OpenAI CLIP cache."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--image',required=True)
    p.add_argument('--output',required=True);p.add_argument('--flip',action='store_true');a=p.parse_args()
    # Keep future library configuration initialization inside this experiment.
    os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'experiments/parameter_finalization/runs/local_detector_setup_20261028/ultralytics_config'))
    import clip
    weights=ROOT/'models/clip_openai/ViT-B-32.pt'
    expected='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
    if not weights.is_file() or hashlib.sha256(weights.read_bytes()).hexdigest()!=expected:
        raise ValueError('explicit verified OpenAI CLIP text-encoder cache required; network disabled')
    original=clip.load
    def local_load(name,*args,**kwargs):
        if name!='ViT-B/32':raise ValueError('unexpected text encoder requested: '+str(name))
        return original(str(weights),*args,**kwargs)
    from experiments.parameter_finalization.local_detector_experiment import worker
    with patch.object(clip,'load',side_effect=local_load):
        worker('yolo_world',a.image,a.output,a.flip)


if __name__=='__main__':main()
