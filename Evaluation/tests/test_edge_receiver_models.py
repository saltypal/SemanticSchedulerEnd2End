import numpy as np

from Evaluation.edge import receiver_models as rm
from Evaluation.edge.contracts import Header
from Evaluation.edge.receiver import Receiver


def flip(rng, key, n):
    for b in rng.choice(64, n, replace=False):
        key ^= 1 << int(b)
    return key


def make_pairs(seed=0, n=150):
    rng = np.random.default_rng(seed)
    sem, perc, pairs = {}, {}, []
    for i in range(n):
        base_s, base_p = int(rng.integers(0, 2 ** 63)), int(rng.integers(0, 2 ** 63))
        sem[2 * i], perc[2 * i] = base_s, base_p
        dup = i % 2 == 0
        sem[2 * i + 1] = flip(rng, base_s, 3 if dup else 32)
        perc[2 * i + 1] = flip(rng, base_p, 3 if dup else 32)
        pairs.append({"id_a": 2 * i, "id_b": 2 * i + 1, "label": "duplicate" if dup else "distinct"})
    return pairs, sem, perc


def test_dup_model_and_auc(tmp_path):
    pairs, sem, perc = make_pairs()
    model, auc = rm.fit_dup_model(pairs, sem, perc)
    assert auc > 0.95
    assert rm.dup_scores(model, [3], [3])[0] > 0.8 > 0.2 > rm.dup_scores(model, [32], [32])[0]
    y = [1 if p["label"] == "duplicate" else 0 for p in pairs]
    sc = rm.dup_scores(model, [(sem[p["id_a"]] ^ sem[p["id_b"]]).bit_count() for p in pairs],
                       [(perc[p["id_a"]] ^ perc[p["id_b"]]).bit_count() for p in pairs])
    out = rm.per_event_auc(sc, y, ["x", "y"] * (len(y) // 2))
    assert set(out) == {"x", "y"}
    rm.save_models(tmp_path / "m.json", dup=model)
    assert rm.load_models(tmp_path / "m.json")[0] == model


def test_u_hat_as_receiver_fn(tmp_path):
    rng = np.random.default_rng(1)
    n = 300
    sts = rng.normal(size=(n, 13))
    probs = np.full((n, 4), 0.25)
    m = rng.integers(0, 2, n)
    y = (sts[:, 0] > 0).astype(int)
    u = rm.fit_u_hat(sts, probs, m, y)

    def mk(v):
        return Header(0, 0, 1, 0, 0, 0, (.25,) * 4, 0.0, 0, 0, tuple([v] + [0.0] * 12))
    assert u(mk(2.0)) > 0.8 > 0.2 > u(mk(-2.0))
    assert Receiver(u_fn=u).u_hat(mk(2.0)) > 0.5
    rm.save_models(tmp_path / "u.json", u_hat=u)
    u2 = rm.load_models(tmp_path / "u.json")[1]
    assert abs(u2(mk(1.0)) - u(mk(1.0))) < 1e-9
