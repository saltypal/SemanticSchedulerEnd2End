from __future__ import annotations

import torch

from INFRA.Artifacts.latent_artifact import LatentArtifact
from INFRA.Artifacts.transmission_artifact import TransmissionArtifact
from INFRA.Interfaces.channel_interface import ChannelInterface


class DeepJSCCChannelWrapper(ChannelInterface):
    """
    Historical DeepJSCC AWGN channel.

    Unlike the repository's generic AWGN channel, the original DeepJSCC
    implementation operates directly on the real-valued latent tensor.

    For AWGN:

        signal_power = mean(z^2)
        noise_power = signal_power / 10^(SNR/10)

        received = z + N(0, noise_power)
    """

    def __init__(self, channel=None):
        # Kept for compatibility with the previous constructor:
        # DeepJSCCChannelWrapper(AWGNChannel())
        self.channel = channel

    def transmit(
        self,
        latent: LatentArtifact,
        snr_db: float,
        **options,
    ) -> TransmissionArtifact:

        z = latent.tensor

        if z.dim() not in (3, 4):
            raise ValueError(
                f"DeepJSCC latent must be 3D or 4D, got {tuple(z.shape)}"
            )

        if z.dim() == 4:
            k = z[0].numel()

            signal_power = (
                torch.sum(
                    torch.abs(z).square(),
                    dim=(1, 2, 3),
                    keepdim=True,
                )
                / k
            )

        else:
            k = z.numel()

            signal_power = (
                torch.sum(torch.abs(z).square()) / k
            )

        noise_power = signal_power / (10 ** (snr_db / 10))

        noise = torch.randn_like(z) * torch.sqrt(noise_power)

        received = z + noise

        if z.dim() == 4:
            channel_uses_per_image = [
                z[i].numel()
                for i in range(z.shape[0])
            ]
        else:
            channel_uses_per_image = [z.numel()]

        return TransmissionArtifact(
            transmitted=z,
            received=received,
            channel_name="deepjscc_historical_awgn",
            snr_db=snr_db,
            channel_uses_per_image=channel_uses_per_image,
            payload_bits_per_image=None,
            metadata={
                "snr_definition": "signal_power_to_noise_power",
                "channel_semantics": "historical_deepjscc",
                "real_latent_direct_noise": True,
                "complex_mapping": False,
                "qpsk": False,
            },
        )