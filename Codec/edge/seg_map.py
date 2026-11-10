"""Map an ADE20K SegFormer label set onto the project's 20-channel SQ-GAN class map.

Reads ``config/edge/seg_classes.yaml``. ``project_logits`` merges source classes by summing
softmax probabilities per project class (then log), so the output behaves like 20-class logits
and an argmax over it gives ids in 0..19. Stdlib + PyYAML at import; torch only inside functions.
This file has no package-relative imports so a subprocess can load it by path.
"""
from pathlib import Path

import yaml

N_CLASSES = 20
NULL_ID = 19
PINNED_IDS = {6: "traffic_light", 7: "traffic_sign", 19: "null"}   # trainer assumptions (sqgan_train.py:19,84)


def default_path():
    return Path(__file__).resolve().parents[2] / "config" / "edge" / "seg_classes.yaml"


def load_class_map(path=None):
    """Load and validate the YAML; return the dict."""
    cfg = yaml.safe_load(Path(path or default_path()).read_text(encoding="utf-8"))
    validate(cfg)
    return cfg


def validate(cfg):
    classes = cfg["classes"]
    if [c["id"] for c in classes] != list(range(N_CLASSES)):
        raise ValueError("classes must have ids 0..19 in order (19 classes + null)")
    names = [c["name"] for c in classes]
    if len(set(names)) != N_CLASSES:
        raise ValueError("duplicate class names")
    for i, name in PINNED_IDS.items():
        if names[i] != name:
            raise ValueError("class %d must be %r (trainer assumption), got %r" % (i, name, names[i]))
    rows = cfg["ade_to_project"]
    if [r["id"] for r in rows] != list(range(int(cfg["teacher_label_count"]))):
        raise ValueError("ade_to_project must list every source id once, in order")
    bad = sorted({r["to"] for r in rows} - set(names))
    if bad:
        raise ValueError("mapping targets not in classes: %s" % bad)


def build_index(cfg):
    """Return a list: source id -> project class id."""
    ids = {c["name"]: c["id"] for c in cfg["classes"]}
    return [ids[r["to"]] for r in cfg["ade_to_project"]]


def check_teacher_labels(cfg, id2label):
    """Raise if the teacher's id2label names differ from the table (guards a silently wrong mapping)."""
    want = {r["id"]: r["ade"] for r in cfg["ade_to_project"]}
    got = {int(k): str(v) for k, v in id2label.items()}
    if len(got) != len(want):
        raise ValueError("teacher has %d labels, table has %d" % (len(got), len(want)))
    diff = [(i, want[i], got.get(i)) for i in want if got.get(i, "").split(",")[0].strip().lower() != want[i]]
    if diff:
        raise ValueError("teacher label names differ from seg_classes.yaml, first: %s" % diff[:5])


def project_logits(logits, index):
    """(B, S, H, W) source logits -> (B, 20, H, W) log of merged probabilities."""
    import torch
    idx = torch.as_tensor(index, dtype=torch.long, device=logits.device)
    if logits.shape[1] != idx.numel():
        raise ValueError("expected %d source channels, got %d" % (idx.numel(), logits.shape[1]))
    probs = torch.softmax(logits.float(), dim=1)
    out = probs.new_zeros(logits.shape[0], N_CLASSES, *logits.shape[2:])
    out.index_add_(1, idx, probs)
    return torch.log(out.clamp_min(1e-12))


def wrap_teacher(teacher, cfg):
    """Make ``teacher(...).logits`` return 20-channel logits WITHOUT changing its state_dict keys.

    The bundle reader (Codec/integration/sqgan_payload.py:93) loads ``teacher.*`` into a plain
    SegformerForSemanticSegmentation, so only ``forward`` is replaced on the instance.
    """
    check_teacher_labels(cfg, teacher.config.id2label)
    index = build_index(cfg)
    inner = teacher.forward

    def forward(*args, **kwargs):
        out = inner(*args, **kwargs)
        out.logits = project_logits(out.logits, index)
        return out

    teacher.forward = forward
    return teacher
