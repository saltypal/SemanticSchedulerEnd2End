import numpy as np
import pytest

from parameters.edge import sts


def test_majority_vote():
    seg = np.zeros((256, 512), dtype=int)
    seg[:16, :16] = 5
    seg[:16, :10] = 7          # 160 of 256 pixels are class 7 -> majority
    g = sts.majority_vote_grid(seg)
    assert g.shape == (16, 32) and g[0, 0] == 7 and g[0, 1] == 0


def test_sts_hand_checked():
    grid = np.full((16, 32), 10)           # sky everywhere
    grid[0, 0:4] = 11                      # 4 person tokens (idx 0..3)
    grid[15, 0:2] = 13                     # 2 vehicle tokens
    kept = [0, 1, 2, 3, 100, 101]          # 4 person + 2 sky (row 3)
    v = sts.compute_sts(grid, kept)
    assert len(v) == 13 and all(0 <= x <= 1 for x in v)
    assert v[0] == pytest.approx(4 / 6) and v[5] == pytest.approx(2 / 6) and v[1] == 0
    assert v[7] == pytest.approx(1.0) and v[8] == 0          # person fully kept, vehicles not kept
    assert v[9] == 0 and v[10] == 0                           # no building / road in the scene
    assert v[11] > 0 and v[12] == pytest.approx(0.0)          # one coarse cell -> zero entropy
    assert sts.compute_sts(grid, [100])[11:] == [0.0, 0.0]


def test_sts_bad_index():
    with pytest.raises(ValueError):
        sts.compute_sts(np.zeros((16, 32), dtype=int), [512])
