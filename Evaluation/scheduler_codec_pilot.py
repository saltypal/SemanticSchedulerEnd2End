"""Measure native codec jobs on a declared development manifest."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
from PIL import Image, ImageOps

from Codec.Baselines.JPEG.jpeg_codec import JPEGCodec
from Codec.integration.sqgan_payload import SQGANDigitalCodec, sha256
from Evaluation.scheduler_codec_replay import Observation, decode_cue, encode_cue, road_retention


PINNED_SHA = "706396402d86e84276393ce3fe39f0b0b1b545ff024c0ecae3b8ab72d22f3d6d"


def elapsed(start):
    return (perf_counter_ns() - start) / 1e6


def run(args):
    rows = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    if not rows or any(row.get("source_split") != "validation" for row in rows):
        raise ValueError("only explicitly declared development validation rows are accepted")
    if len({row["item_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate development identity")
    if args.output.exists():
        raise ValueError("output already exists")
    if args.codec == "sqgan":
        if not args.bundle or not args.source_root or not (args.bundle / "bundle.json").is_file():
            raise FileNotFoundError("verified SQ-GAN bundle and source root required")
        if sha256(args.bundle / "last.pt") != PINNED_SHA:
            raise ValueError("checkpoint does not match the selected SQ-GAN trial")
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    torch.set_num_threads(2)
    reader = AutoModelForImageClassification.from_pretrained(args.reader, local_files_only=True).eval()
    processor = AutoImageProcessor.from_pretrained(args.reader, local_files_only=True, use_fast=False)
    label = next(i for i, name in reader.config.id2label.items() if name == "roads_damage")
    codec = (SQGANDigitalCodec(bundle=args.bundle, source_root=args.source_root)
             if args.codec == "sqgan" else JPEGCodec(55))

    @torch.inference_mode()
    def score(picture):
        inputs = processor(images=picture, return_tensors="pt")
        return float(reader(**inputs).logits[0, label].sigmoid())

    def stages(path):
        start = perf_counter_ns()
        with Image.open(path) as raw:
            source = raw.convert("RGB")
            content = ImageOps.contain(source, (512, 256), Image.Resampling.BILINEAR)
        image = Image.new("RGB", (512, 256), (127, 127, 127))
        left, top = (512 - content.width) // 2, (256 - content.height) // 2
        box = (left, top, left + content.width, top + content.height)
        image.paste(content, (left, top))
        cue, cue_ms, teacher_ms = None, 0., 0.
        if args.codec == "sqgan":
            tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1)[None].float() / 255
            clock = perf_counter_ns()
            semantic = codec.semantic_map(tensor)
            teacher_ms = elapsed(clock)
            if str(codec._teacher.config.id2label[0]).casefold() != "road":
                raise ValueError("pinned teacher no longer maps class 0 to road")
            clock = perf_counter_ns()
            payload = codec.encode(tensor, semantic)
            encode_ms = elapsed(clock)
            clock = perf_counter_ns()
            image_positions, _ = codec.packet.decode_packet(payload)[0]
            road = np.zeros((256, 512), dtype=bool)
            road[top:box[3], left:box[2]] = semantic[0, 0, top:box[3], left:box[2]].cpu().numpy()
            cue = decode_cue(encode_cue(road_retention(road, image_positions)))
            cue_ms = elapsed(clock)
        else:
            clock = perf_counter_ns()
            payload = codec.encode(image)
            encode_ms = elapsed(clock)
        sender_ms = elapsed(start)
        clock = perf_counter_ns()
        try:
            reconstructed = codec.decode(payload)
        except (ValueError, RuntimeError) as exc:
            return payload, sender_ms, elapsed(clock), 0., None, cue, cue_ms, teacher_ms, encode_ms, str(exc), source, content
        decode_ms = elapsed(clock)
        clock = perf_counter_ns()
        if args.codec == "sqgan":
            pixels = (reconstructed[0].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round()
            reconstructed = Image.fromarray(pixels.astype(np.uint8))
        prediction = score(reconstructed.crop(box))
        return payload, sender_ms, decode_ms, elapsed(clock), prediction, cue, cue_ms, teacher_ms, encode_ms, None, source, content

    stages(args.manifest.parent / rows[0]["image_path"])  # warm components and reader
    observations = []
    for row in rows:
        path = args.manifest.parent / row["image_path"]
        payload, sender, decode, read, prediction, cue, cue_ms, teacher, encode, error, source, content = stages(path)
        observation = Observation(row["item_id"], hashlib.sha256(payload).hexdigest(), sender, len(payload),
                                  decode, read, None if prediction is None else prediction >= 0.5,
                                  "roads_damage" in row["positive_labels"], row.get("spi"),
                                  row.get("reference_critical"), cue)
        observations.append(dict(observation=asdict(observation), reader_score=prediction,
                                 teacher_ms=teacher, encode_ms=encode, cue_ms=cue_ms, error=error,
                                 source_sha256=sha256(path), event_group=row.get("event_group"),
                                 source_reader_score=score(source), resized_reader_score=score(content)))
    receipt = dict(codec=args.codec, scope="Native visible road damage, development only. CPU, two torch threads.",
                   reader_checkpoint_sha256=sha256(args.reader / "model.safetensors"), threshold=0.5,
                   preprocessing="Aspect-preserving fit in 512x256 gray canvas; crop padding before reading.",
                   manifest_sha256=sha256(args.manifest), sqgan_checkpoint_sha256=PINNED_SHA if args.codec == "sqgan" else None,
                   cue_bytes=1 if args.codec == "sqgan" else 0, rows=observations)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--reader", required=True, type=Path)
    parser.add_argument("--codec", choices=("sqgan", "jpeg"), default="sqgan")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
