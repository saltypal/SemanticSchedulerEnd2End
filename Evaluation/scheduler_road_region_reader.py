"""Qualify a frozen road-tile reader before spending compute on native decoding."""

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics import average_precision_score

from Codec.integration.sqgan_payload import SQGANDigitalCodec, sha256
from Evaluation.scheduler_codec_pilot import PINNED_SHA
from Evaluation.scheduler_frozen_reader_screen import READER_REVISION


def fit(image):
    content = ImageOps.contain(image, (512, 256), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (512, 256), (127,) * 3)
    x, y = (512 - content.width) // 2, (256 - content.height) // 2
    box = (x, y, x + content.width, y + content.height)
    canvas.paste(content, box)
    return canvas, box


def road_boxes(mask):
    """Rank nine overlapping half-size tiles by road-pixel count."""
    height, width = mask.shape
    if min(height, width) < 4 or not mask.any():
        return []
    tiles = []
    for j in range(3):
        for i in range(3):
            left, top = i * width // 4, j * height // 4
            right, bottom = (i + 2) * width // 4, (j + 2) * height // 4
            count = int(mask[top:bottom, left:right].sum())
            if count:
                tiles.append((-count, j, i, (left, top, right, bottom)))
    return [tile[-1] for tile in sorted(tiles)[:3]]


class RoadReader:
    def __init__(self, reader, bundle):
        import torch
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification
        from transformers import SegformerForSemanticSegmentation

        self.torch = torch
        torch.set_num_threads(2)
        self.processor = AutoImageProcessor.from_pretrained(reader, local_files_only=True, use_fast=False)
        self.model = AutoModelForImageClassification.from_pretrained(reader, local_files_only=True).eval()
        self.damage = next(i for i, name in self.model.config.id2label.items() if name == "roads_damage")
        config = json.loads((bundle / "portable_config.json").read_text())
        cfg = AutoConfig.from_pretrained(config["teacher_model"], revision=config["teacher_revision"],
                                         local_files_only=True)
        if str(cfg.id2label[0]).casefold() != "road":
            raise ValueError("teacher class zero must be road")
        self.teacher = SegformerForSemanticSegmentation(cfg)
        saved = torch.load(bundle / "last.pt", map_location="cpu", weights_only=True, mmap=True)
        prefix = "teacher."
        self.teacher.load_state_dict({k[len(prefix):]: v for k, v in saved["state_dict"].items()
                                      if k.startswith(prefix)}, strict=True)
        self.teacher.eval()

    def read(self, image, include_global=False):
        torch = self.torch
        canvas, (x, y, right, bottom) = fit(image)
        width, height = right - x, bottom - y
        tensor = torch.from_numpy(np.array(canvas)).permute(2, 0, 1)[None].float() / 255
        mean = tensor.new_tensor([.485, .456, .406])[None, :, None, None]
        std = tensor.new_tensor([.229, .224, .225])[None, :, None, None]
        with torch.inference_mode():
            logits = self.teacher(pixel_values=(tensor - mean) / std).logits
            ids = torch.nn.functional.interpolate(logits, size=(256, 512), mode="bilinear",
                                                   align_corners=False).argmax(1)
            boxes = road_boxes((ids[0, y:bottom, x:right] == 0).numpy())
            crops = [image.crop((a * image.width // width, b * image.height // height,
                                 math.ceil(c * image.width / width), math.ceil(d * image.height / height)))
                     for a, b, c, d in boxes]
            inputs = [image, *crops] if include_global or not crops else crops
            scores = self.model(**self.processor(images=inputs, return_tensors="pt")).logits[:, self.damage].sigmoid()
        global_score = float(scores[0]) if include_global else None
        tile_score = float(scores[1:].max() if include_global and crops else scores.max())
        return tile_score, global_score, len(boxes)


def summarize(rows, score, threshold):
    truth = [r["positive"] for r in rows]
    scores = [r[score] for r in rows]
    return dict(tp=sum(t and s >= threshold for t, s in zip(truth, scores)),
                fp=sum(not t and s >= threshold for t, s in zip(truth, scores)),
                ap=float(average_precision_score(truth, scores)))


def calibrate(rows, score):
    negatives = sorted((r[score] for r in rows if not r["positive"]), reverse=True)
    return float(np.nextafter(negatives[2], np.inf))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("reader", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--source-screen", type=Path)
    parser.add_argument("--codec-root", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.reader.name != READER_REVISION or sha256(args.bundle / "last.pt") != PINNED_SHA:
        raise ValueError("unexpected frozen checkpoint")
    inputs = json.loads(args.manifest.read_bytes())
    if any(r["source_split"] != "validation" for r in inputs):
        raise ValueError("development validation images required")
    identities = {r["source_sha256"] for r in inputs}
    if len(identities) != len(inputs):
        raise ValueError("duplicate inputs")
    source_screen = json.loads(args.source_screen.read_bytes()) if args.source_screen else None
    if source_screen and (not source_screen["passed"] or not args.codec_root
                          or source_screen["manifest_sha256"] != sha256(args.manifest)):
        raise ValueError("paired codec run requires a passing source receipt for the same inputs")
    thresholds = None
    if args.calibration:
        calibration = json.loads(args.calibration.read_bytes())
        if identities & {r["sha256"] for r in calibration["rows"]}:
            raise ValueError("calibration and evaluation images overlap")
        thresholds = calibration["thresholds"]
    if source_screen:
        thresholds = source_screen["thresholds"]
    reader = RoadReader(args.reader, args.bundle)
    codec = SQGANDigitalCodec(bundle=args.bundle, source_root=args.codec_root) if source_screen else None
    rows = []
    for item in inputs:
        path = args.manifest.parent / item["image_path"]
        if sha256(path) != item["source_sha256"]:
            raise ValueError("source hash mismatch")
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        row = dict(id=item["item_id"], sha256=item["source_sha256"],
                   positive="roads_damage" in item["positive_labels"])
        if codec:
            canvas, box = fit(image)
            tensor = reader.torch.from_numpy(np.array(canvas)).permute(2, 0, 1)[None].float() / 255
            start = perf_counter()
            payload = codec.encode(tensor)
            sender = (perf_counter() - start) * 1000
            start = perf_counter()
            reconstruction = codec.decode(payload)
            decode = (perf_counter() - start) * 1000
            pixels = (reconstruction[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round()
            image = Image.fromarray(pixels.astype(np.uint8)).crop(box)
            row.update(payload_bytes=len(payload), sender_ms=sender, decode_ms=decode)
        start = perf_counter()
        tile, whole, tiles = reader.read(image, include_global=True)
        row.update(tile=tile, whole=whole, tiles=tiles, reader_ms=(perf_counter() - start) * 1000)
        rows.append(row)
        print(f"Completed {len(rows)}/{len(inputs)}", flush=True)
    thresholds = thresholds or {key: calibrate(rows, key) for key in ("tile", "whole")}
    metrics = {key: summarize(rows, key, thresholds[key]) for key in thresholds}
    min_tp = (2 * sum(r["positive"] for r in rows) + 2) // 3
    passed = bool(args.calibration or codec) and metrics["tile"]["tp"] >= min_tp and metrics["tile"]["fp"] <= 2
    result = dict(stage="reconstruction" if codec else "source" if args.calibration else "calibration",
                  manifest_sha256=sha256(args.manifest), reader_revision=READER_REVISION,
                  reader_sha256=sha256(args.reader / "model.safetensors"), sqgan_checkpoint_sha256=PINNED_SHA,
                  thresholds=thresholds, metrics=metrics, minimum_true_positives=min_tp,
                  maximum_false_positives=2, passed=passed, rows=rows)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
