"""Adapter that exposes the compatible DeepJSCC implementation through the project contract."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from Channels.channel_utils import require_torch
from INFRA.Artifacts import ImageArtifact, LatentArtifact, TransmissionArtifact
from INFRA.Interfaces import EncDecInterface
from INFRA.Registries import ENCDEC_REGISTRY


class DeepJSCCAdapter(EncDecInterface):
    def __init__(
        self,
        checkpoint_path,
        source_root,
        c=19,
        device=None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.source_root = Path(source_root)
        self.c = int(c)
        self.device = device
        self.encoder = None
        self.decoder = None

    def build(self, device=None):
        torch = require_torch()

        target_device = (
            device
            or self.device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        model_path = self.source_root / "old_model_6.py"

        if not model_path.exists():
            raise FileNotFoundError(
                f"DeepJSCC model file not found: {model_path}"
            )

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"DeepJSCC checkpoint not found: {self.checkpoint_path}"
            )

        source_root = str(self.source_root.resolve())

        if source_root not in sys.path:
            sys.path.insert(0, source_root)

        spec = importlib.util.spec_from_file_location(
            "external_deepjscc_old_model_6",
            model_path,
        )

        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"Could not load DeepJSCC module from {model_path}"
            )

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.encoder = module._Encoder(c=self.c)
        self.decoder = module._Decoder(c=self.c)

        state_dict = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        encoder_state = {
            key[len("encoder."):]: value
            for key, value in state_dict.items()
            if key.startswith("encoder.")
        }

        decoder_state = {
            key[len("decoder."):]: value
            for key, value in state_dict.items()
            if key.startswith("decoder.")
        }

        if not encoder_state:
            raise RuntimeError(
                "No encoder weights found in DeepJSCC checkpoint."
            )

        if not decoder_state:
            raise RuntimeError(
                "No decoder weights found in DeepJSCC checkpoint."
            )

        self.encoder.load_state_dict(encoder_state)
        self.decoder.load_state_dict(decoder_state)

        self.encoder.to(target_device)
        self.decoder.to(target_device)

        self.encoder.eval()
        self.decoder.eval()

        self.device = target_device

        return self

    def _require_built(self):
        if self.encoder is None or self.decoder is None:
            raise RuntimeError(
                "DeepJSCCAdapter has not been built. Call .build() first."
            )

        return self.encoder, self.decoder

    def encode(
        self,
        image: ImageArtifact,
        **options: object,
    ) -> LatentArtifact:
        torch = require_torch()

        encoder, _ = self._require_built()

        # The repository stores ImageArtifact tensors in [0, 1].
        # old_model_6 expects input in [0, 255] and performs /255 internally.
        image_tensor = image.tensor.to(self.device) * 255.0

        with torch.no_grad():
            latent = encoder(image_tensor)

        return LatentArtifact(
            tensor=latent,
            rate_mask=None,
            source_model="DeepJSCC",
            rate_tokens=self.c,
            metadata={
                "sample_ids": list(image.sample_ids),
                "latent_shape": tuple(latent.shape),
                "c": self.c,
                "input_scale": "repo_[0,1]_converted_to_[0,255]_for_old_model_6",
            },
        )

    def decode(
        self,
        latent: LatentArtifact,
        **options: object,
    ) -> ImageArtifact:
        torch = require_torch()

        _, decoder = self._require_built()

        received = latent.tensor.to(self.device)

        with torch.no_grad():
            reconstruction = decoder(received)

        # old_model_6 returns the reconstructed image in [0, 255].
        # The repository ImageArtifact contract uses [0, 1].
        reconstruction = reconstruction / 255.0

        # Numerical safety: keep the reconstruction inside the valid
        # image range expected by the rest of the evaluation pipeline.
        reconstruction = torch.clamp(reconstruction, 0.0, 1.0)

        return ImageArtifact(
            tensor=reconstruction,
            sample_ids=list(
                latent.metadata.get("sample_ids", [])
            ),
            source="DeepJSCCDecoder",
            metadata={
                "c": self.c,
                "output_scale": "[0,1]",
                "decoder_native_output_scale": "[0,255]",
            },
        )

    def reconstruct_received(
        self,
        transmission: TransmissionArtifact,
        latent: LatentArtifact,
    ) -> ImageArtifact:
        received_latent = LatentArtifact(
            tensor=transmission.received,
            rate_mask=latent.rate_mask,
            source_model=latent.source_model,
            rate_tokens=latent.rate_tokens,
            metadata={
                **latent.metadata,
                "snr_db": transmission.snr_db,
            },
        )

        return self.decode(
            received_latent,
            snr_db=transmission.snr_db,
        )


def register_deepjscc() -> None:
    if not ENCDEC_REGISTRY.contains("deepjscc"):
        ENCDEC_REGISTRY.register(
            "deepjscc",
            DeepJSCCAdapter,
        )