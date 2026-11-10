"""Per-image codec TABLE rows: image_id, payload_bytes, sts, fidelity, codec (+ optional lpips).

Everything model-specific is injected as callables so tests use fakes:
  encode_fn(image)->(payload_bytes, kept_idx, seg_map); decode_fn(payload)->recon;
  judge_fn(original, recon)->0/1; sts_fn(seg_map, kept_idx)->13 floats; lpips_fn optional.
``encode_with_kept_tokens`` is the real-codec encode_fn (masker hook, same pattern as
Codec.integration.sqgan_prefeatures). ``run_table`` is resumable, flushes each row, and shards by hash parity.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path


def _find_index(output):
    if isinstance(output, dict):
        return output.get("sample_index")
    if hasattr(output, "sample_index"):
        return output.sample_index
    if isinstance(output, (tuple, list)):
        for o in output:
            r = _find_index(o)
            if r is not None:
                return r
    return None


@contextlib.contextmanager
def masker_capture(codec):
    """Record the masker's sample_index (kept token indices) during one encode."""
    holder = {}

    def record(module, inputs, output):
        idx = _find_index(output)
        if idx is not None:
            holder["kept"] = idx.detach().cpu().reshape(-1).tolist() if hasattr(idx, "detach") else list(idx)
    handle = codec.model.masker.register_forward_hook(record)
    try:
        yield holder
    finally:
        handle.remove()


def encode_with_kept_tokens(codec, image, semantic=None):
    """Return (packet_bytes, kept_token_indices, segmentation_map[H,W] ints)."""
    if semantic is None:
        semantic = codec.semantic_map(image)
    with masker_capture(codec) as holder:
        payload = codec.encode(image, semantic)
    if "kept" not in holder:
        raise RuntimeError("masker did not expose sample_index")
    seg = semantic.argmax(1)[0].cpu().numpy()
    return payload, holder["kept"], seg


def build_rows(items, encode_fn, decode_fn, judge_fn, sts_fn, codec_tag="v0", lpips_fn=None):
    """Yield one row per (image_id, image)."""
    for image_id, image in items:
        payload, kept, seg = encode_fn(image)
        recon = decode_fn(payload)
        row = {"image_id": image_id, "payload_bytes": len(payload),
               "sts": [float(x) for x in sts_fn(seg, kept)],
               "fidelity": int(judge_fn(image, recon)), "codec": codec_tag}
        if lpips_fn is not None:
            row["lpips"] = float(lpips_fn(image, recon))
        yield row


def in_shard(image_id, shard):
    i, n = shard
    return int(hashlib.sha256(str(image_id).encode()).hexdigest(), 16) % n == i


def _done_ids(path):
    p = Path(path)
    if not p.exists():
        return set()
    return {json.loads(l)["image_id"] for l in p.read_text().splitlines() if l.strip()}


def run_table(items, out_jsonl, shard=(0, 1), **kw):
    """Resumable: skips ids already in out_jsonl; flushes each row. Returns rows written."""
    done = _done_ids(out_jsonl)
    todo = [(k, im) for k, im in items if in_shard(k, shard) and k not in done]
    n = 0
    with open(out_jsonl, "a") as f:
        for row in build_rows(todo, **kw):
            f.write(json.dumps(row) + "\n")
            f.flush()
            n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build codec TABLE rows with the real SQ-GAN codec")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1", help="i/n split by image-id hash parity")
    ap.add_argument("--judge", required=True, help="LinearProbeJudge file")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    shard = tuple(int(x) for x in a.shard.split("/"))
    import numpy as np
    import torch
    from PIL import Image
    from Codec.integration.sqgan_payload import SQGANDigitalCodec
    from Evaluation.edge.judge import LinearProbeJudge, extract_dino_embeddings
    from parameters.edge.sts import compute_sts
    codec = SQGANDigitalCodec(device=a.device)
    judge = LinearProbeJudge.load(a.judge)
    paths = sorted(p for p in Path(a.images_dir).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))

    def load(p):
        arr = np.asarray(Image.open(p).convert("RGB").resize((512, 256)), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)[None]

    def to_np(t):
        return (t[0].permute(1, 2, 0).detach().cpu().numpy().clip(0, 1) * 255).astype("uint8")

    def judge_fn(orig, rec):
        e = extract_dino_embeddings([to_np(orig), to_np(rec)], "facebook/dinov2-base", 2, a.device)
        return int(judge.predict(e[:1])[0] == judge.predict(e[1:])[0])
    items = ((p.stem, load(p)) for p in paths)
    run_table(items, a.out, shard, encode_fn=lambda im: encode_with_kept_tokens(codec, im), decode_fn=codec.decode,
              judge_fn=judge_fn, sts_fn=compute_sts)


if __name__ == "__main__":
    main()
