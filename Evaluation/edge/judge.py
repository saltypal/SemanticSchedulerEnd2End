"""Fidelity judge: linear probe on frozen embeddings; u = judge(recon) == judge(original).

Also DINOv2 embedding extraction (lazy transformers import) and a human audit sheet / agreement helper.
"""
import csv

import numpy as np


class LinearProbeJudge:
    """Class-balanced logistic regression on embedding arrays."""

    def __init__(self, C=1.0, seed=0):
        self.C, self.seed, self.clf = C, seed, None

    def fit(self, emb, labels):
        from sklearn.linear_model import LogisticRegression
        self.clf = LogisticRegression(C=self.C, class_weight="balanced", max_iter=1000, random_state=self.seed)
        self.clf.fit(np.asarray(emb, dtype=float), np.asarray(labels))
        return self

    def _need(self):
        if self.clf is None:
            raise ValueError("judge not fitted")

    def predict(self, emb):
        self._need()
        return self.clf.predict(np.asarray(emb, dtype=float))

    def predict_proba(self, emb):
        self._need()
        return self.clf.predict_proba(np.asarray(emb, dtype=float))

    def save(self, path):
        import joblib
        joblib.dump(self.clf, path)

    @classmethod
    def load(cls, path):
        import joblib
        j = cls()
        j.clf = joblib.load(path)
        return j


def fidelity(judge, emb_original, emb_recon):
    """0/1 array: 1 where the judge's class agrees on original and reconstruction."""
    a, b = judge.predict(emb_original), judge.predict(emb_recon)
    if len(a) != len(b):
        raise ValueError("embedding count mismatch")
    return (a == b).astype(int)


def extract_dino_embeddings(paths_or_images, model_name="facebook/dinov2-base", batch_size=32, device="cpu"):
    """CLS embeddings (N, D) from paths, PIL images or uint8 HxWx3 arrays. Lazy heavy imports; not used in tests."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    def as_pil(x):
        if hasattr(x, "convert"):
            return x
        if isinstance(x, np.ndarray):
            return Image.fromarray(x)
        return Image.open(x)

    items = list(paths_or_images)
    out = []
    for i in range(0, len(items), batch_size):
        inp = proc(images=[as_pil(x).convert("RGB") for x in items[i:i + batch_size]], return_tensors="pt").to(device)
        with torch.inference_mode():
            out.append(model(**inp).last_hidden_state[:, 0].float().cpu().numpy())
    return np.concatenate(out)


def audit_sheet(pairs, n, rng, out_csv):
    """Write n random (original, reconstruction[, u]) rows with an empty human_label column."""
    pairs = list(pairs)
    idx = rng.choice(len(pairs), size=min(n, len(pairs)), replace=False)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["original", "reconstruction", "u", "human_label"])
        for i in sorted(idx):
            p = pairs[i]
            w.writerow([p[0], p[1], p[2] if len(p) > 2 else "", ""])
    return len(idx)


def agreement(audit_csv):
    """Fraction of labelled rows where u equals human_label; None if nothing labelled."""
    with open(audit_csv, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["human_label"].strip() != "" and r["u"].strip() != ""]
    if not rows:
        return None
    return float(np.mean([int(float(r["u"])) == int(float(r["human_label"])) for r in rows]))
