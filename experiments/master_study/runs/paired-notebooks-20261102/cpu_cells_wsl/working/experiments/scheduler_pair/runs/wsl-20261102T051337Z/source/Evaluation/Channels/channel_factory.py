"""Evaluation-owned channel selection; Rician remains an explicit placeholder."""

from Evaluation.Channels.awgn import AWGNChannel
from Evaluation.Channels.rayleigh import RayleighChannel
from Evaluation.Channels.rician import RicianChannel
from Evaluation.Registries import CHANNEL_REGISTRY


def register_builtin_channels() -> None:
    entries = {"awgn": AWGNChannel, "rayleigh_perfect_csi": RayleighChannel, "rician": RicianChannel}
    for name, constructor in entries.items():
        if not CHANNEL_REGISTRY.contains(name):
            CHANNEL_REGISTRY.register(name, constructor)


def create_channel(name: str):
    register_builtin_channels()
    return CHANNEL_REGISTRY.create(name)
