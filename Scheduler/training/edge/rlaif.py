"""Qwen-based AI feedback (RLAIF) pipeline: prompts, collection, Bradley-Terry reward model, CLI.

Heavy libraries (torch, transformers, PIL) are imported lazily. ``scorer`` is an injected callable
``scorer(prompt, images) -> log-probs`` over the answer options (4 tier letters for pointwise,
2 for pairwise "A"/"B"), so tests use fakes; ``QwenScorer`` (Qwen2.5-VL) is built only by the CLI.
Pairwise queries are asked in BOTH orders and only order-consistent answers are kept.
The reward model maps header-visible job features to a scalar reward and a 4-way tier head;
``reward_fn_from_model`` plugs it in place of the true value in reward attribution.
"""
import argparse
import json
import os

import numpy as np

from Evaluation.edge.contracts import TIERS

LETTERS = ("A", "B", "C", "D")


def pointwise_prompt(tiers=TIERS):
    opts = "\n".join("%s) %s" % (LETTERS[i], t) for i, t in enumerate(tiers))
    return ("This image was received by an emergency-response team. How urgent is it to act on what it "
            "shows?\n%s\nAnswer with a single letter." % opts)


def pairwise_prompt(tiers=TIERS):
    return ("Two disaster images, Image A (first) and Image B (second), are waiting in a queue. Which one "
            "should be processed first because it is more urgent for rescue and response (tiers: %s)? "
            "Answer with a single letter, A or B." % ", ".join(tiers))


def collect_pointwise(images, scorer, tiers=TIERS):
    """``images``: iterable of (image_id, image). Returns records with class log-probs and argmax label."""
    prompt, out = pointwise_prompt(tiers), []
    for image_id, image in images:
        lp = np.asarray(scorer(prompt, [image]), dtype=float)
        if lp.shape != (len(tiers),):
            raise ValueError("pointwise scorer must return %d log-probs" % len(tiers))
        out.append({"image_id": image_id, "logp": lp.tolist(), "label": int(lp.argmax())})
    return out


def ask_pair(scorer, img_a, img_b):
    """Returns (winner_first_order, winner_swapped_order) as 0 if img_a wins else 1."""
    prompt = pairwise_prompt()
    first = int(np.argmax(scorer(prompt, [img_a, img_b])))
    swapped = int(np.argmax(scorer(prompt, [img_b, img_a])))
    return first, 1 - swapped


def collect_pairwise(pairs, scorer):
    """``pairs``: iterable of (id_a, img_a, id_b, img_b). Returns (kept records, consistency_rate)."""
    kept, total = [], 0
    for id_a, img_a, id_b, img_b in pairs:
        total += 1
        w1, w2 = ask_pair(scorer, img_a, img_b)
        if w1 == w2:
            kept.append({"id_a": id_a, "id_b": id_b, "winner": id_a if w1 == 0 else id_b,
                         "loser": id_b if w1 == 0 else id_a})
    return kept, (len(kept) / total if total else 0.0)


class BradleyTerryRewardModel:
    """r(x) = trunk(x) -> scalar; tier head (4-way) shares the trunk. hidden=0 gives a linear model."""

    def __init__(self, in_dim, hidden=0, n_tiers=4):
        from torch import nn
        self.in_dim, self.hidden, self.n_tiers = in_dim, hidden, n_tiers
        feat = in_dim if hidden == 0 else hidden
        self.trunk = nn.Identity() if hidden == 0 else nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.reward_head = nn.Linear(feat, 1)
        self.tier_head = nn.Linear(feat, n_tiers)

    def parameters(self):
        for m in (self.trunk, self.reward_head, self.tier_head):
            yield from m.parameters()

    def forward(self, x):
        h = self.trunk(x)
        return self.reward_head(h).squeeze(-1), self.tier_head(h)

    def reward(self, X):
        import torch
        with torch.no_grad():
            r, _ = self.forward(torch.as_tensor(np.asarray(X), dtype=torch.float32))
        return r.numpy()

    def save(self, path):
        import torch
        torch.save({"in_dim": self.in_dim, "hidden": self.hidden, "n_tiers": self.n_tiers,
                    "trunk": self.trunk.state_dict(), "reward": self.reward_head.state_dict(),
                    "tier": self.tier_head.state_dict()}, path)

    @classmethod
    def load(cls, path):
        import torch
        d = torch.load(path, map_location="cpu")
        m = cls(d["in_dim"], d["hidden"], d["n_tiers"])
        m.trunk.load_state_dict(d["trunk"])
        m.reward_head.load_state_dict(d["reward"])
        m.tier_head.load_state_dict(d["tier"])
        return m


def fit_reward_model(model, win_x, lose_x, point_x=None, point_y=None, epochs=300, lr=0.05,
                     tier_weight=1.0, seed=0):
    """Loss = mean -log sigmoid(r_win - r_lose) + tier_weight * CE(tier head, pointwise labels)."""
    import torch
    torch.manual_seed(seed)
    f = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)
    wx, lx = f(win_x), f(lose_x)
    have_pt = point_x is not None and len(point_x) > 0
    if have_pt:
        px, py = f(point_x), torch.as_tensor(np.asarray(point_y), dtype=torch.long)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for _ in range(epochs):
        loss = torch.nn.functional.softplus(-(model.forward(wx)[0] - model.forward(lx)[0])).mean()
        if have_pt:
            loss = loss + tier_weight * torch.nn.functional.cross_entropy(model.forward(px)[1], py)
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(float(loss))
    return hist


def reward_fn_from_model(model):
    """Callable job_features (F,) or (N,F) -> reward (float or array)."""
    def fn(features):
        x = np.asarray(features, dtype=float)
        r = model.reward(x.reshape(-1, x.shape[-1]))
        return float(r[0]) if x.ndim == 1 else r
    return fn


def rows_value_fn(reward_fn, features_by_id):
    """Adapt a feature-level reward to the ``value_fn(row)`` used by attribute_rewards."""
    return lambda row: float(reward_fn(features_by_id[row["id"]]))


# ---------------------------------------------------------------- resumable collection / CLI
def done_ids(path):
    ids = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ids.add(json.loads(line)["key"])
    return ids


def collect_to_jsonl(items, scorer, out_path, mode="pointwise", tiers=TIERS):
    """Append one JSON line per item, skipping keys already in ``out_path``. Returns number written.

    pointwise items: (id, image); pairwise items: (id_a, img_a, id_b, img_b); key is id or "a|b".
    Pairwise lines also record ``consistent``; inconsistent pairs are logged with winner=None.
    """
    seen, n = done_ids(out_path), 0
    with open(out_path, "a", encoding="utf-8") as f:
        for it in items:
            if mode == "pointwise":
                key = str(it[0])
                if key in seen:
                    continue
                rec = collect_pointwise([it], scorer, tiers)[0]
            else:
                key = "%s|%s" % (it[0], it[2])
                if key in seen:
                    continue
                kept, _ = collect_pairwise([it], scorer)
                rec = kept[0] if kept else {"id_a": it[0], "id_b": it[2], "winner": None, "loser": None}
                rec["consistent"] = bool(kept)
            rec["key"] = key
            f.write(json.dumps(rec) + "\n")
            f.flush()
            n += 1
    return n


class QwenScorer:
    """Qwen2.5-VL answer-letter scorer. Constructed only by the CLI (heavy imports)."""

    def __init__(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct", load_4bit=False):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        kw = {"torch_dtype": torch.float16, "device_map": "auto"}
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kw).eval()
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.torch = torch

    def __call__(self, prompt, images):
        from PIL import Image
        imgs = [Image.open(i).convert("RGB") if isinstance(i, str) else i for i in images]
        content = [{"type": "image"} for _ in imgs] + [{"type": "text", "text": prompt}]
        text = self.proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)
        inputs = self.proc(text=[text], images=imgs, return_tensors="pt").to(self.model.device)
        n = 4 if "single letter." in prompt and "A or B" not in prompt else 2
        ids = [self.proc.tokenizer.encode(l, add_special_tokens=False)[0] for l in LETTERS[:n]]
        with self.torch.no_grad():
            logits = self.model(**inputs).logits[0, -1]
        return self.torch.log_softmax(logits[ids].float(), dim=0).cpu().numpy()


def _read_list(path, width):
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 1 and width == 2:
                parts = [os.path.splitext(os.path.basename(parts[0]))[0], parts[0]]
            if len(parts) >= width:
                yield tuple(parts[:width])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rlaif")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--images-list", required=True, help="pointwise: 'id<TAB>path' lines; pairwise: 'ida<TAB>patha<TAB>idb<TAB>pathb'")
    c.add_argument("--out", required=True)
    c.add_argument("--mode", choices=("pointwise", "pairwise"), default="pointwise")
    c.add_argument("--shard", default="0/1", help="i/n")
    c.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    c.add_argument("--4bit", dest="load_4bit", action="store_true")
    a = ap.parse_args(argv)
    i, n = (int(x) for x in a.shard.split("/"))
    items = [it for k, it in enumerate(_read_list(a.images_list, 2 if a.mode == "pointwise" else 4)) if k % n == i]
    scorer = QwenScorer(a.model_id, a.load_4bit)
    print("wrote", collect_to_jsonl(items, scorer, a.out, a.mode))


if __name__ == "__main__":
    main()
