from types import SimpleNamespace

import numpy as np

from Evaluation.edge.contracts import Decision, Header, Policy, QueuedJob
from parameters.edge import bitdepth as bd


def _h(p, rel=0.6, sts=()):
    return Header(capture_time=0.0, arrival_time=0.0, payload_bytes=464, mission=0, codec_id=1, m_index=0,
                  tier_probs=tuple(p), relevance=rel, sem_key=1, perc_key=2, sts=tuple(sts))


def test_requantise_only_requested_fields():
    h = _h((0.51, 0.49, 0.0, 0.0), rel=0.6123, sts=[0.3] * 13)
    q = bd.requantise_fields(h, 2, ("spi",))
    assert q.relevance == h.relevance and q.sts == h.sts and abs(sum(q.tier_probs) - 1) < 1e-9
    assert bd.requantise_fields(h, 2, ("relevance",)).relevance in (0.0, 1 / 3, 2 / 3, 1.0)


class ArgmaxCritical(Policy):
    """Picks the job with the highest Critical probability (a decision that low SPI bits can flip)."""
    def select(self, state):
        best = max(state.queue, key=lambda j: (j.header.tier_probs[0], -j.id))
        return Decision(batch=(best.id,))


def test_shadow_counts_flips_and_primary_drives():
    sp = bd.ShadowPolicy(ArgmaxCritical(), ArgmaxCritical(), 1, ("spi",))
    sp.reset(None, np.random.default_rng(0))
    q = (QueuedJob(0, _h((0.40, 0.2, 0.2, 0.2)), 1.0), QueuedJob(1, _h((0.45, 0.2, 0.2, 0.15)), 1.0))
    state = SimpleNamespace(queue=q)
    import dataclasses
    state = dataclasses.make_dataclass("S", [("queue", tuple)])(q)
    d = sp.select(state)
    assert d.batch == (1,)                      # primary on full precision picks job 1 (0.45 > 0.40)
    assert sp.epochs == 1 and sp.flips == 1     # at 1 bit both round to 0 -> shadow tie-breaks to job 0
    sp8 = bd.ShadowPolicy(ArgmaxCritical(), ArgmaxCritical(), 8, ("spi",))
    sp8.reset(None, np.random.default_rng(0))
    sp8.select(state)
    assert sp8.flips == 0


def test_smallest_passing():
    rows = [{"fields": "spi", "bits": b, "passes": b >= 4} for b in (2, 3, 4, 8)]
    assert bd.smallest_passing(rows) == {"spi": 4}
