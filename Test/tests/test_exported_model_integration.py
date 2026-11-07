"""Active scheduler codec integration: real SQ-GAN bundle, digital bytes only."""
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from PIL import Image
from Codec.integration.sqgan_payload import build_scheduler_codec

@pytest.fixture(scope="module")
def codec():
    # No skip/fallback: the selected codec's assets must actually load.
    return build_scheduler_codec()

@pytest.fixture(scope="module")
def encoded(codec):
    source=Path(__file__).resolve().parents[2]/"experiments/sqgan_scheduler/runs/bundle-switch-20261102/input.png"
    with Image.open(source) as image:
        x=torch.from_numpy(np.asarray(image.convert("RGB")).copy()).permute(2,0,1)[None].float()/255
    semantic=codec.semantic_map(x)
    return x,codec.encode(x,semantic)

def test_fine_tuned_component_state_is_strictly_loaded(codec):
    assert len(codec.validation["components"])==10
    assert all(item["strict"] and item["tensor_count"]>0 for item in codec.validation["components"])
    assert codec.representation=="digital_bytes"

def test_digital_packet_roundtrip_and_corruption(codec,encoded):
    x,payload=encoded
    assert isinstance(payload,bytes)
    y=codec.decode(payload)
    assert y.shape==x.shape and torch.isfinite(y).all()
    corrupted=bytearray(payload);corrupted[-1]^=1
    with pytest.raises(ValueError,match="CRC"):codec.decode(bytes(corrupted))

def test_real_payload_receiver_dispatch_completion_and_failure(codec,encoded):
    from dataclasses import replace
    from Test.tests.test_decoder_service_lifecycle import service,message
    receiver=service()
    original=message(1)
    receiver.enqueue(replace(original,payload_ref=encoded[1]))
    job=receiver.dispatch(0.)
    result=codec.execute(receiver,job)
    assert result.shape==(1,3,256,512)
    assert receiver.completions[-1].outcome=="succeeded" and len(receiver.history)==1
    corrupted=bytearray(encoded[1]);corrupted[-1]^=1
    next_message=message(2,arrival=receiver.completions[-1].finished_at)
    receiver.enqueue(replace(next_message,payload_ref=bytes(corrupted)))
    job=receiver.dispatch(next_message.arrival_time)
    with pytest.raises(ValueError,match="CRC"):codec.execute(receiver,job)
    assert receiver.completions[-1].outcome=="failed" and len(receiver.history)==1
