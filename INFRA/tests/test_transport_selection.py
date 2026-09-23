import pytest


def test_masked_rayleigh_counts_active_symbols_per_image():
    torch = pytest.importorskip('torch')
    from INFRA.Artifacts import LatentArtifact
    from INFRA.Channels.rayleigh import RayleighChannel
    values = torch.ones(2, 4, 4)
    mask = torch.zeros_like(values)
    mask[:, :, :2] = 1
    latent = LatentArtifact(tensor=values, source_model='test', rate_mask=mask)
    transmission = RayleighChannel().transmit(latent, 10)
    assert transmission.channel_uses_per_image == [4, 4]
    assert torch.all(transmission.transmitted[:, :, 2:] == 0)
    assert torch.all(transmission.received[:, :, 2:] == 0)


def test_unimplemented_channel_rejected_before_run():
    from INFRA.configuration import resolve_config
    with pytest.raises(ValueError, match='Rician'):
        resolve_config(overrides={'transport': {'name': 'rician'}})
