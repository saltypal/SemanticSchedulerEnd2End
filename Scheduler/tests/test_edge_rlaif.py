"""RLAIF pipeline with fake scorers and synthetic preferences."""
import json

import numpy as np

from Scheduler.training.edge import rlaif


def test_prompts_mention_options():
    assert "A)" in rlaif.pointwise_prompt() and "D)" in rlaif.pointwise_prompt()
    assert "Image A" in rlaif.pairwise_prompt()


def test_collect_pointwise_with_fake_scorer():
    def scorer(prompt, images):
        lp = np.full(4, -2.0)
        lp[images[0] % 4] = -0.1
        return lp
    recs = rlaif.collect_pointwise([("a", 5), ("b", 2)], scorer)
    assert [r["label"] for r in recs] == [1, 2]


def test_pairwise_drops_inconsistent_and_reports_rate():
    def scorer(prompt, images):
        a, b = images
        if a == 99 or b == 99:                 # position-biased: always answers "A"
            return np.array([0.0, -5.0])
        return np.array([0.0, -5.0]) if a > b else np.array([-5.0, 0.0])
    pairs = [("x", 3, "y", 1), ("y", 1, "x", 3), ("p", 99, "q", 1), ("r", 2, "s", 7)]
    kept, rate = rlaif.collect_pairwise(pairs, scorer)
    assert rate == 0.75 and len(kept) == 3
    assert kept[0]["winner"] == "x" and kept[2]["winner"] == "s"


def test_bradley_terry_recovers_linear_preference():
    rng = np.random.default_rng(0)
    w = np.array([2.0, -1.0, 0.0, 0.5])
    a, b = rng.normal(size=(300, 4)), rng.normal(size=(300, 4))
    win = np.where(((a - b) @ w > 0)[:, None], a, b)
    lose = np.where(((a - b) @ w > 0)[:, None], b, a)
    m = rlaif.BradleyTerryRewardModel(4)
    hist = rlaif.fit_reward_model(m, win, lose, epochs=300)
    assert hist[-1] < hist[0]
    learned = m.reward_head.weight.detach().numpy().ravel()
    cos = learned @ w / (np.linalg.norm(learned) * np.linalg.norm(w))
    assert cos > 0.95
    fn = rlaif.reward_fn_from_model(m)
    assert isinstance(fn(a[0]), float) and fn(a).shape == (300,)


def test_tier_head_and_save_load(tmp_path):
    rng = np.random.default_rng(1)
    px = rng.normal(size=(80, 4))
    py = (px[:, 0] > 0).astype(int)
    m = rlaif.BradleyTerryRewardModel(4, hidden=8)
    rlaif.fit_reward_model(m, px[:5], px[5:10], px, py, epochs=100)
    pred = m.forward(__import__("torch").as_tensor(px, dtype=__import__("torch").float32))[1].argmax(1).numpy()
    assert (pred == py).mean() > 0.85
    m.save(tmp_path / "m.pt")
    m2 = rlaif.BradleyTerryRewardModel.load(tmp_path / "m.pt")
    np.testing.assert_allclose(m2.reward(px), m.reward(px), atol=1e-6)


def test_rows_value_fn():
    m = rlaif.BradleyTerryRewardModel(2)
    fn = rlaif.rows_value_fn(rlaif.reward_fn_from_model(m), {7: np.array([1.0, 2.0])})
    assert isinstance(fn({"id": 7}), float)


def test_resumable_collection_skips_done_ids(tmp_path):
    out = tmp_path / "o.jsonl"
    calls = []

    def scorer(prompt, images):
        calls.append(images[0])
        return np.array([-1.0, -2.0, -3.0, -4.0])
    items = [("i%d" % k, k) for k in range(5)]
    assert rlaif.collect_to_jsonl(items[:3], scorer, str(out)) == 3
    assert rlaif.collect_to_jsonl(items, scorer, str(out)) == 2
    assert calls == [0, 1, 2, 3, 4]
    keys = [json.loads(l)["key"] for l in out.read_text().splitlines()]
    assert keys == ["i%d" % k for k in range(5)]


def test_pairwise_jsonl_records_consistency(tmp_path):
    out = tmp_path / "p.jsonl"
    scorer = lambda prompt, images: np.array([0.0, -5.0])   # always "A": inconsistent
    assert rlaif.collect_to_jsonl([("a", 1, "b", 2)], scorer, str(out), mode="pairwise") == 1
    rec = json.loads(out.read_text())
    assert rec["consistent"] is False and rec["winner"] is None
