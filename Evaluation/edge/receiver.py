"""Receiver-side state: a delivered-only memory of header keys and the duplicate model.

Only headers of images that were actually decoded and delivered enter the memory, so a policy can
never learn from images it has not served. Reuses ``contracts.Header`` only.
"""
from collections import deque
from dataclasses import dataclass, replace
from math import exp


@dataclass(frozen=True)
class DupModel:
    """Logistic duplicate model: p = sigmoid(a + b1*d_sem/64 + b2*d_perc/64)."""
    a: float = 4.0
    b1: float = -14.0
    b2: float = -14.0


def _sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + exp(-x))
    e = exp(x)
    return e / (1.0 + e)


class Receiver:
    """Ring memory of delivered (sem_key, perc_key) pairs plus duplicate/fidelity estimates."""

    def __init__(self, size=512, model=None, u_fn=None, fusion=None):
        if size < 1:
            raise ValueError("memory size must be >= 1")
        self.model = model or DupModel()
        self.u_fn = u_fn
        self.fusion = fusion   # D19: StsFusion or None
        self.memory = deque(maxlen=int(size))

    def min_distances(self, header):
        """Min Hamming distance (semantic, perceptual) to memory; None when memory is empty."""
        if not self.memory:
            return None
        d_sem = min((header.sem_key ^ s).bit_count() for s, _ in self.memory)
        d_perc = min((header.perc_key ^ p).bit_count() for _, p in self.memory)
        return d_sem, d_perc

    def dup_prob(self, header):
        d = self.min_distances(header)
        if d is None:
            return 0.0
        m = self.model
        return _sigmoid(m.a + m.b1 * d[0] / 64.0 + m.b2 * d[1] / 64.0)

    def belief(self, header):
        """Header whose tier_probs is the SPI+STS fused belief (unchanged without fusion or a 13-value sts)."""
        if self.fusion is None or len(header.sts) != 13:
            return header
        return replace(header, tier_probs=tuple(self.fusion(header.tier_probs, header.sts)))

    def u_hat(self, header):
        return 1.0 if self.u_fn is None else float(self.u_fn(header))

    def deliver(self, header):
        self.memory.append((header.sem_key, header.perc_key))
