import numpy as np

from parameters.edge import features as F


def test_simhash_determinism_and_stats():
    rng = np.random.default_rng(0)
    v = rng.standard_normal(512)
    assert F.simhash64(v, 1) == F.simhash64(v.copy(), 1)
    assert F.hamming(F.simhash64(v, 1), F.simhash64(v, 1)) == 0
    assert F.hamming(F.simhash64(v, 1), F.simhash64(-v, 1)) == 64
    d = [F.hamming(F.simhash64(rng.standard_normal(512), 1), F.simhash64(rng.standard_normal(512), 1))
         for _ in range(200)]
    assert 28 < np.mean(d) < 36
    assert F.simhash64(v, 1) != F.simhash64(v, 2)


def test_phash_brightness_and_difference():
    rng = np.random.default_rng(1)
    img = rng.random((64, 96)) * 150
    assert F.hamming(F.phash64(img), F.phash64(img + 40)) <= 2
    other = rng.random((64, 96)) * 150
    assert F.hamming(F.phash64(img), F.phash64(other)) > 10


def test_relevance_and_fake_embedder():
    a = np.array([1.0, 0, 0])
    assert F.relevance(a, a) == 1.0 and F.relevance(a, -a) == 0.0
    e = F.FakeEmbedder(16)(["x", "y"])
    assert e.shape == (2, 16) and np.allclose(e, F.FakeEmbedder(16)(["x", "y"]))
