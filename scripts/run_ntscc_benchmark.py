"""Evaluate an existing NTSCC reconstruction using repository metrics.

This is a modular evaluation wrapper.
It does not modify the existing NTSCC implementation,
DeepJSCC implementation, or repository benchmark scripts.
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import json
import sys
from pathlib import Path

from PIL import Image

from EncDecPipeline.Preprocessing.image_loader import pil_to_tensor
from Evaluation.Image.psnr import psnr
from Evaluation.Image.ssim import ssim


def main() -> None:
    # ---------------------------------------------------------
    # Command-line arguments
    #
    # Usage:
    # python scripts\run_ntscc_benchmark.py ^
    #   "reference.png" ^
    #   "ntscc_reconstruction.png" ^
    #   "ntscc_results.json"
    # ---------------------------------------------------------

    if len(sys.argv) < 4:
        raise SystemExit(
            "Usage:\n"
            "python scripts\\run_ntscc_benchmark.py "
            "\"reference.png\" "
            "\"ntscc_reconstruction.png\" "
            "\"ntscc_results.json\""
        )

    reference_path = Path(sys.argv[1])
    reconstruction_path = Path(sys.argv[2])
    metadata_path = Path(sys.argv[3])

    if not reference_path.exists():
        raise FileNotFoundError(
            f"Reference image not found:\n{reference_path}"
        )

    if not reconstruction_path.exists():
        raise FileNotFoundError(
            f"NTSCC reconstruction not found:\n{reconstruction_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"NTSCC metadata not found:\n{metadata_path}"
        )

    # ---------------------------------------------------------
    # Load images.
    # ---------------------------------------------------------

    reference = Image.open(reference_path).convert("RGB")
    reconstruction = Image.open(reconstruction_path).convert("RGB")

    # The NTSCC runner already generated the reconstruction at
    # the evaluation resolution. Resize the reference to exactly
    # the reconstruction dimensions for metric evaluation.
    if reference.size != reconstruction.size:
        reference = reference.resize(
            reconstruction.size,
            Image.Resampling.BICUBIC,
        )

    reference_tensor = pil_to_tensor(reference).unsqueeze(0)
    reconstruction_tensor = pil_to_tensor(reconstruction).unsqueeze(0)

    # ---------------------------------------------------------
    # Repository image-quality metrics.
    # ---------------------------------------------------------

    psnr_value = float(
        psnr(
            reference_tensor,
            reconstruction_tensor,
        )
    )

    ssim_value = float(
        ssim(
            reference_tensor,
            reconstruction_tensor,
        )
    )

    # ---------------------------------------------------------
    # Read communication metrics produced by NTSCC itself.
    # ---------------------------------------------------------

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    snr_db = float(metadata["snr_db"])
    cbr = float(metadata["cbr"])
    bpp_y = float(metadata["bpp_y"])
    bpp_z = float(metadata["bpp_z"])

    input_shape = metadata["input_shape"]

    height = int(input_shape[-2])
    width = int(input_shape[-1])

    source_elements = int(metadata["source_elements"])

    # NTSCC's CBR is defined relative to source elements.
    # Therefore use the same definition rather than inventing
    # a different channel-use convention.
    channel_uses = int(round(cbr * height * width))

    # ---------------------------------------------------------
    # Print results.
    # ---------------------------------------------------------

    print()
    print("NTSCC REAL IMAGE EVALUATION")
    print("--------------------------------")

    print(f"Reference: {reference_path}")
    print(f"Reconstruction: {reconstruction_path}")
    print(f"Input shape: {input_shape}")
    print(f"Reconstruction size: {reconstruction.size}")

    print()
    print("CHANNEL")
    print("-------")
    print("Channel: AWGN")
    print(f"SNR (dB): {snr_db}")

    print()
    print("COMMUNICATION METRICS")
    print("---------------------")
    print(f"Source elements: {source_elements:,}")
    print(f"Channel Uses: {channel_uses:,}")
    print(f"CBR: {cbr:.6f}")
    print(f"BPP-y: {bpp_y:.6f}")
    print(f"BPP-z: {bpp_z:.6f}")

    print()
    print("IMAGE QUALITY METRICS")
    print("---------------------")
    print(f"PSNR (dB): {psnr_value:.6f}")
    print(f"SSIM: {ssim_value:.6f}")

    print()
    print("RESULT SUMMARY")
    print("--------------")
    print(
        f"NTSCC | "
        f"SNR={snr_db:g} dB | "
        f"Channel Uses={channel_uses:,} | "
        f"CBR={cbr:.6f} | "
        f"PSNR={psnr_value:.4f} dB | "
        f"SSIM={ssim_value:.4f}"
    )

    print()
    print("NTSCC evaluation OK")


if __name__ == "__main__":
    main()