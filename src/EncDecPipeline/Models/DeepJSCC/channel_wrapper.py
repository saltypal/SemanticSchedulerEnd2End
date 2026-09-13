"""DeepJSCC-specific channel compatibility wrapper."""

from __future__ import annotations

from typing import Any

from Channels.channel_utils import require_torch
from INFRA.Artifacts import LatentArtifact, TransmissionArtifact
from INFRA.Interfaces import ChannelInterface


class DeepJSCCChannelWrapper(ChannelInterface):
    """Adapt odd-sized DeepJSCC latents to the existing channel contract.

    The wrapped channel implementation is never modified. One zero-valued
    element is added only when the latent contains an odd number of real
    values, allowing real-to-complex channel mapping. The padding is removed
    before the received latent is returned to the decoder.
    """

    def __init__(self, channel: ChannelInterface) -> None:
        self.channel = channel

    def transmit(
        self,
        latent: LatentArtifact,
        snr_db: float,
        **options: object,
    ) -> TransmissionArtifact:
        torch = require_torch()

        original_shape = tuple(latent.tensor.shape)
        batch_size = int(latent.tensor.shape[0])
        flattened = latent.tensor.reshape(batch_size, -1)

        original_values = int(flattened.shape[1])
        padding_added = original_values % 2

        if padding_added:
            padding = torch.zeros(
                batch_size,
                padding_added,
                dtype=flattened.dtype,
                device=flattened.device,
            )
            channel_tensor = torch.cat((flattened, padding), dim=1)
        else:
            channel_tensor = flattened

        channel_latent = LatentArtifact(
            tensor=channel_tensor,
            rate_mask=None,
            source_model=latent.source_model,
            rate_tokens=latent.rate_tokens,
            metadata={
                **latent.metadata,
                "deepjscc_original_shape": original_shape,
                "deepjscc_padding_values": padding_added,
            },
        )

        transmission = self.channel.transmit(
            channel_latent,
            snr_db,
            **options,
        )

        received = transmission.received.reshape(batch_size, -1)

        if padding_added:
            received = received[:, :-padding_added]

        received = received.reshape(original_shape)

        return TransmissionArtifact(
            transmitted=latent.tensor,
            received=received,
            channel_name=transmission.channel_name,
            snr_db=transmission.snr_db,
            channel_uses_per_image=transmission.channel_uses_per_image,
            payload_bits_per_image=transmission.payload_bits_per_image,
            metadata={
                **transmission.metadata,
                "deepjscc_padding_values": padding_added,
                "deepjscc_original_shape": original_shape,
                "deepjscc_channel_shape": tuple(channel_tensor.shape),
            },
        )