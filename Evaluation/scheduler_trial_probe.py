"""Measure a development-only source/header/JPEG reader comparison."""

import argparse
import hashlib
import json
import platform
import struct
from pathlib import Path
from statistics import median
from time import perf_counter_ns

from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

from Codec.Baselines.JPEG.jpeg_codec import JPEGCodec


HEADER = struct.Struct(">BQ")


def elapsed_ms(start):
    return (perf_counter_ns() - start) / 1_000_000


def appearance_hash(image):
    pixels = image.convert("L").resize((9, 8), Image.Resampling.BILINEAR).tobytes()
    signature = 0
    for row in range(8):
        for col in range(8):
            index = row * 9 + col
            signature = (signature << 1) | (pixels[index] > pixels[index + 1])
    return signature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-manifest", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = json.loads(args.sample_manifest.read_text(encoding="utf-8-sig"))
    if args.output.exists():
        parser.error("output already exists")
    torch.set_num_threads(2)
    processor = AutoImageProcessor.from_pretrained(args.model_dir, local_files_only=True, use_fast=False)
    model = AutoModelForImageClassification.from_pretrained(args.model_dir, local_files_only=True).cpu().eval()
    label = next(i for i, name in model.config.id2label.items() if name == "roads_damage")
    codec = JPEGCodec(55)

    def score(image):
        with torch.inference_mode():
            return model(**processor(images=image, return_tensors="pt")).logits[0, label].sigmoid().item()

    score(Image.new("RGB", (224, 224)))
    records = []
    for row in rows:
        path = args.sample_manifest.parent / Path(row["image_path"].replace("\\", "/")).name
        raw = path.read_bytes()
        started = perf_counter_ns()
        with Image.open(path) as opened:
            exif = opened.getexif()
            capture = bool(exif.get(306) or exif.get_ifd(34665).get(36867))
            has_gps = bool(exif.get_ifd(34853))
            source = opened.convert("RGB")
        source_score = score(source)
        header = HEADER.pack(round(source_score * 255), appearance_hash(source))
        sender_ms = elapsed_ms(started)
        header_score, signature = HEADER.unpack(header)
        started = perf_counter_ns()
        payload = codec.encode(source)
        encode_ms = elapsed_ms(started)
        started = perf_counter_ns()
        decoded = codec.decode(payload)
        decode_ms = elapsed_ms(started)
        started = perf_counter_ns()
        decoded_score = score(decoded)
        reader_ms = elapsed_ms(started)
        records.append(dict(id=row["item_id"], raw_sha256=hashlib.sha256(raw).hexdigest(),
                            native_road_damage="roads_damage" in row["positive_labels"],
                            source=source_score, header=header_score / 255, decoded=decoded_score,
                            appearance_hash=f"{signature:016x}", has_capture_time=capture, has_gps=has_gps,
                            sender_ms=sender_ms, encode_ms=encode_ms, decode_ms=decode_ms,
                            reader_ms=reader_ms, header_bytes=len(header), payload_bytes=len(payload)))
    confusion = {}
    for path in ("source", "header", "decoded"):
        confusion[path] = {
            f"truth_{int(truth)}_pred_{int(pred)}": sum(
                r["native_road_damage"] == truth and (r[path] >= 0.5) == pred for r in records
            )
            for truth in (False, True) for pred in (False, True)
        }
    with (args.model_dir / "model.safetensors").open("rb") as stream:
        model_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    result = dict(
        scope="Historical, class-stratified development sample. Native visible road damage. JPEG control.",
        codec="Repository JPEGCodec quality=55 4:4:4",
        model_sha256=model_hash,
        sample_manifest_sha256=hashlib.sha256(args.sample_manifest.read_bytes()).hexdigest(),
        source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runtime=dict(os=platform.platform(), python=platform.python_version(), torch=torch.__version__, threads=2),
        threshold=0.5, confusion=confusion,
        medians={key: median(r[key] for r in records) for key in
                 ("sender_ms", "encode_ms", "decode_ms", "reader_ms", "payload_bytes")},
        records=records,
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()

