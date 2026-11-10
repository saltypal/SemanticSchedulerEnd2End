import numpy as np
import pytest

from Evaluation.edge.contracts import LatencyProfile, ValueSpec
from Evaluation.edge.edge_config import load_config


def test_latency_profile_cdf_and_quantiles():
    p = LatencyProfile({1: [1.0, 2.0, 3.0, 4.0], 2: [2.0, 2.0]})
    assert p.cdf(1, 0.5) == 0 and p.cdf(1, 2.0) == 0.5 and p.cdf(1, 9) == 1.0
    assert p.mean(1) == 2.5 and p.batch_sizes == (1, 2)
    assert 3.0 <= p.p95(1) <= 4.0
    assert p.sample(2, np.random.default_rng(0)) == 2.0
    with pytest.raises(KeyError):
        p.mean(8)
    with pytest.raises(ValueError):
        LatencyProfile({1: []})


def test_value_spec_time_utility_shape():
    spec = ValueSpec.from_config(load_config("sim"))
    assert spec.g(0, 0.0) == 1.0 and spec.g(0, 5.0) == 1.0
    assert spec.g(0, 7.5) == pytest.approx(0.5)      # halfway between D=5 and H=10
    assert spec.g(0, 10.0) == 0.0 and spec.g(0, 99.0) == 0.0
    assert spec.hard_deadline(3) == 600.0
