"""Runs only when a Kaggle-produced artifact directory is supplied explicitly."""

import os

import pytest


MODEL_DIR = os.environ.get("STAGE1_MODEL_DIR")
pytestmark = pytest.mark.skipif(not MODEL_DIR, reason="set STAGE1_MODEL_DIR to a Kaggle-exported models/SwinJSCC directory")


def test_full_and_split_tensor_state_artifacts_validate_and_load() -> None:
    from EncDecPipeline.Models.SwinJSCC.model_loader import load_stage1_swinjscc

    adapter = load_stage1_swinjscc(MODEL_DIR, device="cpu")
    encoder, decoder = adapter._require_built()
    assert encoder is not None
    assert decoder is not None


def test_encoder_only_then_decoder_only_round_trip() -> None:
    import torch

    from EncDecPipeline.Models.SwinJSCC.encoder_runtime import EncoderRuntime
    from EncDecPipeline.Models.SwinJSCC.decoder_runtime import DecoderRuntime
    from INFRA.Artifacts import ImageArtifact
    from EncDecPipeline.Models.SwinJSCC.model_loader import load_stage1_swinjscc

    adapter = load_stage1_swinjscc(MODEL_DIR, device="cpu")
    image = ImageArtifact(tensor=torch.rand(1, 3, 256, 256), sample_ids=["integration"], source="test")
    latent = EncoderRuntime(adapter)(image, snr_db=10, rate=96)
    reconstruction = DecoderRuntime(adapter)(latent, snr_db=10)
    assert reconstruction.tensor.shape == image.tensor.shape
