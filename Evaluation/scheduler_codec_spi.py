"""Attach provisional, archived SPI predictions to measured development jobs."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter_ns

import numpy as np
from PIL import Image, ImageOps

from Codec.integration.sqgan_payload import sha256


ARCHIVE = "experiments/mini_dataset/runs/mini_spi_20261030_v1/downloads/SPI_classifier"


def archive_file(name):
    return subprocess.check_output(["git", "show", f"7a1e4b1:{ARCHIVE}/{name}"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("clip_snapshot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    import torch
    from transformers import AutoImageProcessor, CLIPModel

    receipt = json.loads(args.receipt.read_text())
    if sha256(args.manifest) != receipt["manifest_sha256"]:
        raise ValueError("SPI source manifest does not match measured payloads")
    rows = {r["item_id"]: r for r in json.loads(args.manifest.read_text(encoding="utf-8-sig"))}
    identity = json.loads(archive_file("encoder.json"))
    checkpoint = args.clip_snapshot / "pytorch_model.bin"
    if args.clip_snapshot.name != identity["revision"] or sha256(checkpoint) != identity["weights_sha256"][checkpoint.name]:
        raise ValueError("CLIP checkpoint identity mismatch")
    head_raw = archive_file("head.json")
    head = json.loads(head_raw)
    if head["classes"] != [0, 1, 2, 3]:
        raise ValueError("unexpected archived SPI class order")
    weights, bias = np.array(head["coef"]), np.array(head["intercept"])
    torch.set_num_threads(2)
    processor = AutoImageProcessor.from_pretrained(args.clip_snapshot, local_files_only=True, use_fast=False)
    processor_hash = hashlib.sha256(json.dumps(processor.to_dict(), sort_keys=True, default=str).encode()).hexdigest()
    model = CLIPModel.from_pretrained(args.clip_snapshot, local_files_only=True, use_safetensors=False).eval()

    @torch.inference_mode()
    def classify(image):
        padded = ImageOps.pad(image, (224, 224), method=Image.Resampling.BICUBIC, color=(127,) * 3)
        pixels = processor(images=[padded], return_tensors="pt", do_resize=False,
                           do_center_crop=False)["pixel_values"]
        vision = model.vision_model(pixel_values=pixels)
        vector = torch.nn.functional.normalize(model.visual_projection(vision.pooler_output).float(), dim=-1)
        logits = weights @ vector[0].cpu().numpy() + bias
        return 3 - int(np.argmax(logits))

    classify(Image.new("RGB", (224, 224)))
    for record in receipt["rows"]:
        observation = record["observation"]
        path = args.manifest.parent / rows[observation["identity"]]["image_path"]
        if sha256(path) != record["source_sha256"]:
            raise ValueError("SPI source bytes do not match the codec pilot")
        start = perf_counter_ns()
        with Image.open(path) as image:
            observation["spi"] = classify(ImageOps.exif_transpose(image).convert("RGB"))
        record["spi_sender_ms"] = (perf_counter_ns() - start) / 1e6
        observation["sender_ms"] += record["spi_sender_ms"]
    receipt["spi"] = dict(scope="Provisional archived classifier. Predicted SPI only. No urgency ground truth.",
                          checkpoint_sha256=identity["weights_sha256"][checkpoint.name],
                          head_sha256=hashlib.sha256(head_raw).hexdigest(),
                          processor_sha256=processor_hash,
                          archived_processor_sha256=identity["processor_sha256"],
                          unaugmented_receipt_sha256=sha256(args.receipt))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
