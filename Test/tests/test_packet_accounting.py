from EncDecPipeline.Codebooks.codebook_codec import CodebookPacketHeader, pack_fixed_width_indices, unpack_fixed_width_indices
from Evaluation.Communication.channel_uses import qpsk_channel_uses


def test_k256_packet_has_eight_bit_indices_and_full_rate_mask_per_image() -> None:
    header = CodebookPacketHeader(
        codebook_id="swinjscc_tx_k256",
        codebook_size=256,
        latent_shape=(1, 256, 320),
        active_channels=96,
        index_bits=8,
        rate_mask_bits=320,
    )
    assert header.index_count == 256
    assert header.payload_bits == 256 * 8 + 320 + 128 + 32


def test_fixed_width_index_codec_round_trip() -> None:
    indices = [0, 1, 17, 255, 128]
    packed = pack_fixed_width_indices(indices, width=8)
    assert unpack_fixed_width_indices(packed, count=len(indices), width=8) == indices


def test_qpsk_accounting_uses_one_complex_symbol_per_two_coded_bits() -> None:
    assert qpsk_channel_uses(2048) == 1024
