from dataclasses import dataclass

import numpy as np

from Evaluation.edge import regimes


@dataclass(frozen=True)
class H:
    capture_time: float


@dataclass(frozen=True)
class J:
    header: H


def test_awgn_delay_is_bits_over_capacity_and_no_loss():
    out, lost = regimes.apply_link([J(H(10.0))] * 5, {"model": "awgn", "snr_db": 0, "bandwidth_hz": 10000},
                                   np.random.default_rng(0))
    assert lost == 0 and abs((10.0 - out[0].header.capture_time) - 8 * 528 / 10000.0) < 1e-9


def test_rayleigh_loses_more_at_low_snr():
    jobs = [J(H(0.0))] * 4000
    lost = {s: regimes.apply_link(jobs, {"model": "rayleigh", "snr_db": s}, np.random.default_rng(1))[1] for s in (0, 20)}
    assert lost[0] > lost[20] > 0


def test_no_link_is_identity():
    jobs = [J(H(1.0))]
    assert regimes.apply_link(jobs, None, np.random.default_rng(0)) == (jobs, 0)
    assert "rayleigh_snr10" in regimes.load_config("sim")["scenarios"]
