"""Real parity smoke for the single-pass pre-mask SQ-GAN feature export.

Runs the verified teammate bundle end to end on the local CPU and checks, per
real image: deterministic baseline encode, byte-identical payload from
``encode_with_features`` with exactly one encoder call, and pixel-identical
reconstructions from both payloads on the same backend. The observed pre-mask
grid shape is recorded against the handoff's expected 1x256x16x32.

Usage (staged bundle and source root live outside the repository):

    python -m Codec.integration.prefeature_parity_smoke \
        --source-root <staged-source-root> --bundle <staged-bundle> \
        --image <path> [--image <path> ...] --output <receipt.json>
"""
import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from time import perf_counter

EXPECTED_FEATURE_SHAPE = [1, 256, 16, 32]
PINNED_CHECKPOINT_SHA256 = \
    "706396402d86e84276393ce3fe39f0b0b1b545ff024c0ecae3b8ab72d22f3d6d"


def elapsed(start):
    return round((perf_counter() - start) * 1000, 3)


def load_tensor(path):
    """Aspect-preserving fit into the 512x256 canvas used by the native pilots."""
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    with Image.open(path) as raw:
        source = ImageOps.exif_transpose(raw).convert("RGB")
        content = ImageOps.contain(source, (512, 256), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (512, 256), (127, 127, 127))
    left = (512 - content.width) // 2
    top = (256 - content.height) // 2
    canvas.paste(content, (left, top))
    array = np.array(canvas)
    return torch.from_numpy(array).permute(2, 0, 1)[None].float() / 255


def run(args):
    from Codec.integration.sqgan_payload import SQGANDigitalCodec
    from Codec.integration.sqgan_prefeatures import encode_with_features
    import torch

    torch.set_num_threads(args.threads)
    if not args.source_root.is_dir():
        raise FileNotFoundError("staged source root required: %s" % args.source_root)
    if not (args.bundle / "bundle.json").is_file():
        raise FileNotFoundError("staged bundle required: %s" % args.bundle)
    # The pinned teammate package resolves its upstream fetch root from this
    # variable; point it at the staged source root, never at live checkouts.
    os.environ.setdefault("SEMANTIC_LOSSLESS_ROOT", str(args.source_root))

    clock = perf_counter()
    codec = SQGANDigitalCodec(bundle=args.bundle, source_root=args.source_root,
                              device="cpu")
    load_ms = elapsed(clock)
    if codec.checkpoint_sha256 != PINNED_CHECKPOINT_SHA256:
        raise ValueError("bundle checkpoint does not match the pinned Trial 2 bytes")

    rows = []
    all_passed = True
    for image_path in args.image:
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        image = load_tensor(image_path)
        clock = perf_counter()
        semantic = codec.semantic_map(image)
        teacher_ms = elapsed(clock)

        clock = perf_counter()
        baseline_a = codec.encode(image, semantic)
        encode_a_ms = elapsed(clock)
        clock = perf_counter()
        baseline_b = codec.encode(image, semantic)
        encode_b_ms = elapsed(clock)
        deterministic = baseline_a == baseline_b

        clock = perf_counter()
        result = encode_with_features(codec, image, semantic)
        encode_features_ms = elapsed(clock)
        features = result["features"]

        clock = perf_counter()
        reconstruction_baseline = codec.decode(baseline_a)
        decode_baseline_ms = elapsed(clock)
        clock = perf_counter()
        reconstruction_features = codec.decode(result["payload"])
        decode_features_ms = elapsed(clock)

        pixels_identical = bool(torch.equal(reconstruction_baseline,
                                            reconstruction_features))
        payload_identical = result["payload"] == baseline_a
        shape = result["feature_shape"]
        row = {
            "image": str(image_path),
            "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "payload_bytes": len(baseline_a),
            "payload_sha256": hashlib.sha256(baseline_a).hexdigest(),
            "deterministic_baseline_encode": deterministic,
            "payload_identical_encode_with_features": payload_identical,
            "encoder_calls_during_export": result["encoder_calls"],
            "pixel_identical_reconstruction": pixels_identical,
            "reconstruction_sha256": hashlib.sha256(
                reconstruction_baseline.detach().cpu().numpy().tobytes()).hexdigest(),
            "reconstruction_finite": bool(torch.isfinite(reconstruction_baseline).all()),
            "feature_shape": shape,
            "feature_dtype": result["feature_dtype"],
            "feature_shape_matches_handoff_expectation": shape == EXPECTED_FEATURE_SHAPE,
            "features_finite": bool(torch.isfinite(features).all()),
            "teacher_ms": teacher_ms,
            "encode_baseline_ms": encode_a_ms,
            "encode_determinism_check_ms": encode_b_ms,
            "encode_with_features_ms": encode_features_ms,
            "decode_baseline_ms": decode_baseline_ms,
            "decode_features_ms": decode_features_ms,
        }
        row["passed"] = bool(
            deterministic and payload_identical and result["encoder_calls"] == 1
            and pixels_identical and row["reconstruction_finite"]
            and row["features_finite"] and row["feature_dtype"].startswith("torch.float"))
        all_passed = all_passed and row["passed"]
        rows.append(row)
        if args.grids_out:
            import numpy as np
            args.grids_out.mkdir(parents=True, exist_ok=True)
            stem = image_path.stem
            np.save(args.grids_out / (stem + ".npy"),
                    features.detach().cpu().numpy().astype("float32"))
        print("PASS" if row["passed"] else "FAIL", image_path.name,
              "payload_bytes=%d" % row["payload_bytes"],
              "feature_shape=%s" % shape, flush=True)

    receipt = {
        "check": "trial2_test1_prefeature_parity",
        "scope": "Pinned SQ-GAN single-pass pre-mask export, local CPU, real images.",
        "device": "cpu",
        "torch_version": torch.__version__,
        "torch_threads": args.threads,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "source_commit": codec.model.metadata.get("source_commit"),
        "checkpoint_sha256": codec.checkpoint_sha256,
        "bundle_run_id": json.loads((args.bundle / "bundle.json").read_text())["run_id"],
        "codec_inputs": {"bundle": str(args.bundle), "source_root": str(args.source_root)},
        "expected_feature_shape": EXPECTED_FEATURE_SHAPE,
        "codec_load_ms": load_ms,
        "images": rows,
        "summary": {
            "images": len(rows),
            "passed": sum(1 for row in rows if row["passed"]),
            "all_passed": all_passed,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("parity smoke %s -> %s"
          % ("PASSED" if all_passed else "FAILED", args.output), flush=True)
    return 0 if all_passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--image", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--grids-out", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    sys.exit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
