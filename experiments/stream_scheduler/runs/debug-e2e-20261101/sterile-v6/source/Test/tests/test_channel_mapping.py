import pytest

torch = pytest.importorskip("torch")

from Channels.channel_utils import complex_to_real_per_image, real_to_complex_per_image


def test_complex_symbol_pairing_never_crosses_image_boundaries() -> None:
    features = torch.tensor(
        [
            [[1.0, 2.0, 10.0, 20.0]],
            [[3.0, 4.0, 30.0, 40.0]],
        ]
    )
    symbols, shape = real_to_complex_per_image(features)
    assert torch.equal(symbols[0], torch.tensor([1.0 + 10.0j, 2.0 + 20.0j]))
    assert torch.equal(symbols[1], torch.tensor([3.0 + 30.0j, 4.0 + 40.0j]))
    assert torch.equal(complex_to_real_per_image(symbols, shape), features)


def test_per_image_mapping_matches_independently_sharded_batches() -> None:
    features = torch.randn(4, 8, 16)
    all_symbols, _ = real_to_complex_per_image(features)
    sharded_symbols = torch.cat([real_to_complex_per_image(features[:2])[0], real_to_complex_per_image(features[2:])[0]])
    assert torch.equal(all_symbols, sharded_symbols)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_data_parallel_shards_preserve_per_image_symbols() -> None:
    features = torch.randn(4, 8, 16, device="cuda:0")
    all_symbols, _ = real_to_complex_per_image(features)
    shard_zero, _ = real_to_complex_per_image(features[:2])
    shard_one, _ = real_to_complex_per_image(features[2:])
    assert torch.equal(all_symbols.cpu(), torch.cat((shard_zero, shard_one)).cpu())
