"""Compare RAW+LDPC, JPEG+LDPC, and DeepJSCC on one image.

This is an evaluation wrapper around the existing repository components.

Existing teammate implementations are not modified.

Comparison:
    RAW      -> framing -> LDPC -> QPSK -> AWGN -> RAW image
    JPEG     -> JPEG -> framing -> LDPC -> QPSK -> AWGN -> JPEG image
    DeepJSCC -> neural encoder -> repository AWGN -> neural decoder

Metrics:
    - channel uses
    - source/transmitted bits where applicable
    - CBR
    - PSNR
    - SSIM
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from Baselines.DigitalPHY.channel_coding import LDPCQPSKAWGN
from Baselines.JPEG.jpeg_codec import JPEGCodec
from Baselines.JPEG.jpeg_pipeline import encode_jpeg_for_transport
from EncDecPipeline.Models.DeepJSCC.adapter import DeepJSCCAdapter
from EncDecPipeline.Models.DeepJSCC.channel_wrapper import (
    DeepJSCCChannelWrapper,
)
from EncDecPipeline.Preprocessing.image_loader import pil_to_tensor
from Evaluation.Image.psnr import psnr
from Evaluation.Image.ssim import ssim
from INFRA.Artifacts import ImageArtifact


# ============================================================================
# Compatibility helper
# ============================================================================

def bytes_to_bits(payload: bytes) -> np.ndarray:
    """Convert bytes to a big-endian bit array.

    The current repository's bit_utils.py does not expose this helper,
    although jpeg_pipeline.py imports it. We keep the helper local to this
    benchmark rather than modifying the existing teammate implementation.
    """
    return np.unpackbits(
        np.frombuffer(payload, dtype=np.uint8),
        bitorder="big",
    )


# ============================================================================
# Paths
# ============================================================================

CHECKPOINT = Path(
    r"D:\OneDrive - Amrita vishwa vidyapeetham\Desktop\image_semcom"
    r"\Deep-JSCC-PyTorch\out\imagenet_10_0.33_200.00_32_19.pth"
)

SOURCE_ROOT = Path(
    r"D:\OneDrive - Amrita vishwa vidyapeetham\Desktop\image_semcom"
    r"\Deep-JSCC-PyTorch"
)

DEFAULT_IMAGE = Path(
    r"D:\OneDrive - Amrita vishwa vidyapeetham\Desktop\image_semcom"
    r"\Deep-JSCC-PyTorch\demo\kodim08.png"
)


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_SNR_DB = 10.0
DEFAULT_JPEG_QUALITY = 75

LDPC_INFORMATION_BITS = 1024
LDPC_CODEWORD_BITS = 2048
LDPC_ITERATIONS = 20


# ============================================================================
# Image / metric helpers
# ============================================================================

def image_to_raw_bytes(image: Image.Image) -> bytes:
    """Convert an RGB image to raw RGB bytes."""
    rgb = image.convert("RGB")
    return np.asarray(
        rgb,
        dtype=np.uint8,
    ).tobytes()


def raw_bytes_to_image(
    payload: bytes,
    size: tuple[int, int],
) -> Image.Image:
    """Reconstruct an RGB image from raw RGB bytes."""
    width, height = size

    expected_bytes = width * height * 3

    if len(payload) != expected_bytes:
        raise ValueError(
            f"RAW payload has {len(payload)} bytes; "
            f"expected {expected_bytes} bytes for "
            f"{width}x{height} RGB."
        )

    array = np.frombuffer(
        payload,
        dtype=np.uint8,
    ).reshape(
        height,
        width,
        3,
    )

    return Image.fromarray(
        array,
        mode="RGB",
    )


def evaluate_images(
    reference: Image.Image,
    reconstruction: Image.Image,
) -> tuple[float, float]:
    """Evaluate PSNR and SSIM using the repository implementations."""

    reference_tensor = pil_to_tensor(
        reference,
    ).unsqueeze(0)

    reconstruction_tensor = pil_to_tensor(
        reconstruction,
    ).unsqueeze(0)

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

    return psnr_value, ssim_value


def print_result(result: dict) -> None:
    """Print one normalized benchmark result."""

    print()
    print(result["name"])
    print("-" * len(result["name"]))

    print(
        f"Success:          {result['success']}"
    )

    if result.get("source_bits") is not None:
        print(
            f"Source bits:      "
            f"{result['source_bits']:,}"
        )

    if result.get("transmitted_bits") is not None:
        print(
            f"Transmitted bits: "
            f"{result['transmitted_bits']:,}"
        )

    if result.get("channel_uses") is not None:
        print(
            f"Channel uses:     "
            f"{result['channel_uses']:,}"
        )

    if result.get("cbr") is not None:
        print(
            f"CBR:              "
            f"{result['cbr']:.6f}"
        )

    if result.get("psnr_db") is not None:
        print(
            f"PSNR (dB):        "
            f"{result['psnr_db']:.6f}"
        )

    if result.get("ssim") is not None:
        print(
            f"SSIM:             "
            f"{result['ssim']:.6f}"
        )

    if result.get("reason"):
        print(
            f"Reason:           "
            f"{result['reason']}"
        )


# ============================================================================
# RAW + LDPC + QPSK + AWGN
# ============================================================================

def evaluate_raw(
    image: Image.Image,
    digital: LDPCQPSKAWGN,
    snr_db: float,
    device: str,
) -> dict:
    """Evaluate RAW RGB bytes through the existing digital baseline."""

    raw_payload = image_to_raw_bytes(
        image,
    )

    source_bits = bytes_to_bits(
        raw_payload,
    )

    result = digital.transmit(
        source_bits=source_bits,
        esn0_db=snr_db,
        device=device,
    )

    cbr = result.information_bits / (
        image.width
        * image.height
        * 24
    )

    if (
        not result.frame_success
        or result.recovered_bits is None
    ):
        return {
            "name": "RAW + LDPC + QPSK",
            "success": False,
            "channel_uses": result.channel_uses,
            "source_bits": result.information_bits,
            "transmitted_bits": result.coded_bits,
            "cbr": cbr,
            "psnr_db": None,
            "ssim": None,
            "reason": result.failure_reason,
        }

    recovered_bytes = np.packbits(
        result.recovered_bits,
        bitorder="big",
    ).tobytes()

    reconstructed = raw_bytes_to_image(
        recovered_bytes,
        size=(
            image.width,
            image.height,
        ),
    )

    psnr_value, ssim_value = evaluate_images(
        image,
        reconstructed,
    )

    return {
        "name": "RAW + LDPC + QPSK",
        "success": True,
        "channel_uses": result.channel_uses,
        "source_bits": result.information_bits,
        "transmitted_bits": result.coded_bits,
        "cbr": cbr,
        "psnr_db": psnr_value,
        "ssim": ssim_value,
        "reason": "",
    }


# ============================================================================
# JPEG + LDPC + QPSK + AWGN
# ============================================================================

def evaluate_jpeg(
    image: Image.Image,
    digital: LDPCQPSKAWGN,
    snr_db: float,
    device: str,
    quality: int,
) -> dict:
    """Evaluate JPEG bytes through the existing digital baseline."""

    jpeg_stream = encode_jpeg_for_transport(
        image=image,
        quality=quality,
    )

    source_bits = np.asarray(
        jpeg_stream.bits,
        dtype=np.uint8,
    ).reshape(-1)

    result = digital.transmit(
        source_bits=source_bits,
        esn0_db=snr_db,
        device=device,
    )

    original_image_bits = (
        image.width
        * image.height
        * 24
    )

    cbr = result.information_bits / original_image_bits

    if (
        not result.frame_success
        or result.recovered_bits is None
    ):
        return {
            "name": (
                f"JPEG Q{quality} + LDPC + QPSK"
            ),
            "success": False,
            "channel_uses": result.channel_uses,
            "source_bits": result.information_bits,
            "transmitted_bits": result.coded_bits,
            "cbr": cbr,
            "psnr_db": None,
            "ssim": None,
            "reason": result.failure_reason,
        }

    recovered_bytes = np.packbits(
        result.recovered_bits,
        bitorder="big",
    ).tobytes()

    reconstructed = JPEGCodec.decode(
        recovered_bytes,
    )

    psnr_value, ssim_value = evaluate_images(
        image,
        reconstructed,
    )

    return {
        "name": (
            f"JPEG Q{quality} + LDPC + QPSK"
        ),
        "success": True,
        "channel_uses": result.channel_uses,
        "source_bits": result.information_bits,
        "transmitted_bits": result.coded_bits,
        "cbr": cbr,
        "psnr_db": psnr_value,
        "ssim": ssim_value,
        "reason": "",
    }


# ============================================================================
# DeepJSCC
# ============================================================================

def evaluate_deepjscc(
    image: Image.Image,
    snr_db: float,
) -> dict:
    """Evaluate the verified historical DeepJSCC implementation."""

    tensor = pil_to_tensor(
        image,
    ).unsqueeze(0)

    artifact = ImageArtifact(
        tensor=tensor,
        sample_ids=["benchmark"],
        source="DeepJSCCBenchmark",
    )

    # This is the exact historical model/checkpoint configuration.
    #
    # The model's internal channel object is required by old_model_6.py
    # during construction, but the actual transmission in this repository
    # is performed by the injected repository channel below.
    encdec = DeepJSCCAdapter(
        checkpoint_path=CHECKPOINT,
        source_root=SOURCE_ROOT,
        c=19,
        device="cpu",
        channel_type="AWGN",
        snr=200,
    ).build()

    # Encode using the verified notebook model.
    latent = encdec.encode(
        artifact,
    )

    # Use the repository's existing AWGN implementation.
    channel = DeepJSCCChannelWrapper()
    transmission = channel.transmit(
        latent,
        snr_db=snr_db,
    )

    # Decode the received latent using the verified notebook decoder.
    reconstruction = encdec.reconstruct_received(
        transmission,
        latent,
    )

    psnr_value = float(
        psnr(
            artifact.tensor,
            reconstruction.tensor,
        )
    )

    ssim_value = float(
        ssim(
            artifact.tensor,
            reconstruction.tensor,
        )
    )

    latent_values = int(
        latent.tensor.reshape(
            latent.tensor.shape[0],
            -1,
        ).shape[1]
    )

    padding_values = latent_values % 2

    padded_values = (
        latent_values
        + padding_values
    )

    complex_channel_uses = (
        padded_values // 2
    )

    source_values = int(
        tensor.shape[1]
        * tensor.shape[2]
        * tensor.shape[3]
    )

    notebook_style_cbr = (
        latent_values / source_values
    )

    return {
        "name": "DeepJSCC",
        "success": True,
        "channel_uses": complex_channel_uses,
        "source_bits": None,
        "transmitted_bits": None,
        "cbr": notebook_style_cbr,
        "psnr_db": psnr_value,
        "ssim": ssim_value,
        "reason": "",
        "latent_real_values": latent_values,
        "padding_values": padding_values,
        "latent_shape": tuple(
            latent.tensor.shape
        ),
    }


# ============================================================================
# Summary
# ============================================================================

def print_summary(
    results: list[dict],
) -> None:
    """Print a compact comparison table."""

    print()
    print("=" * 82)
    print("COMMUNICATION COMPARISON")
    print("=" * 82)

    header = (
        f"{'Method':<30}"
        f"{'Success':<10}"
        f"{'Channel Uses':>16}"
        f"{'CBR':>12}"
        f"{'PSNR':>10}"
        f"{'SSIM':>10}"
    )

    print(header)
    print("-" * 82)

    for result in results:
        channel_uses = result.get(
            "channel_uses"
        )

        cbr = result.get("cbr")
        psnr_value = result.get(
            "psnr_db"
        )
        ssim_value = result.get(
            "ssim"
        )

        channel_text = (
            f"{channel_uses:,}"
            if channel_uses is not None
            else "-"
        )

        cbr_text = (
            f"{cbr:.6f}"
            if cbr is not None
            else "-"
        )

        psnr_text = (
            f"{psnr_value:.4f}"
            if psnr_value is not None
            else "-"
        )

        ssim_text = (
            f"{ssim_value:.4f}"
            if ssim_value is not None
            else "-"
        )

        print(
            f"{result['name']:<30}"
            f"{str(result['success']):<10}"
            f"{channel_text:>16}"
            f"{cbr_text:>12}"
            f"{psnr_text:>10}"
            f"{ssim_text:>10}"
        )

    print("=" * 82)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    """Run the benchmark."""

    image_path = (
        Path(sys.argv[1])
        if len(sys.argv) >= 2
        else DEFAULT_IMAGE
    )

    snr_db = (
        float(sys.argv[2])
        if len(sys.argv) >= 3
        else DEFAULT_SNR_DB
    )

    jpeg_quality = (
        int(sys.argv[3])
        if len(sys.argv) >= 4
        else DEFAULT_JPEG_QUALITY
    )

    digital_device = (
        sys.argv[4]
        if len(sys.argv) >= 5
        else "cuda:0"
    )

    # ------------------------------------------------------------------
    # Validate paths
    # ------------------------------------------------------------------

    if not image_path.exists():
        raise FileNotFoundError(
            f"Input image not found:\n{image_path}"
        )

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"DeepJSCC checkpoint not found:\n{CHECKPOINT}"
        )

    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(
            f"DeepJSCC source directory not found:\n{SOURCE_ROOT}"
        )

    image = Image.open(
        image_path,
    ).convert("RGB")

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    print()
    print("=" * 82)
    print("SEMANTIC COMMUNICATION BENCHMARK")
    print("=" * 82)
    print(
        f"Image:            {image_path}"
    )
    print(
        f"Resolution:       "
        f"{image.width} x {image.height}"
    )
    print(
        f"SNR / EsN0:       "
        f"{snr_db} dB"
    )
    print(
        f"JPEG quality:     "
        f"{jpeg_quality}"
    )
    print(
        f"Digital device:   "
        f"{digital_device}"
    )
    print("=" * 82)

    results: list[dict] = []

    # ------------------------------------------------------------------
    # DeepJSCC
    # ------------------------------------------------------------------

    print()
    print("Running DeepJSCC...")

    deepjscc_result = evaluate_deepjscc(
        image=image,
        snr_db=snr_db,
    )

    results.append(
        deepjscc_result
    )

    print_result(
        deepjscc_result,
    )

    print(
        f"Latent shape:     "
        f"{deepjscc_result['latent_shape']}"
    )

    print(
        f"Latent real vals: "
        f"{deepjscc_result['latent_real_values']:,}"
    )

    print(
        f"Padding values:   "
        f"{deepjscc_result['padding_values']}"
    )

    # ------------------------------------------------------------------
    # RAW + JPEG
    # ------------------------------------------------------------------

    print()
    print("Running RAW/JPEG digital baselines...")

    try:
        digital = LDPCQPSKAWGN(
            information_block_bits=(
                LDPC_INFORMATION_BITS
            ),
            codeword_bits=(
                LDPC_CODEWORD_BITS
            ),
            iterations=(
                LDPC_ITERATIONS
            ),
        )

        raw_result = evaluate_raw(
            image=image,
            digital=digital,
            snr_db=snr_db,
            device=digital_device,
        )

        results.append(
            raw_result
        )

        print_result(
            raw_result,
        )

        jpeg_result = evaluate_jpeg(
            image=image,
            digital=digital,
            snr_db=snr_db,
            device=digital_device,
            quality=jpeg_quality,
        )

        results.append(
            jpeg_result
        )

        print_result(
            jpeg_result,
        )

    except RuntimeError as error:
        print()
        print(
            "RAW/JPEG digital baselines unavailable."
        )
        print(
            f"Reason: {error}"
        )
        print(
            "DeepJSCC result above is still valid."
        )

    # ------------------------------------------------------------------
    # Final comparison
    # ------------------------------------------------------------------

    print_summary(
        results,
    )

    print()
    print("Benchmark completed.")


if __name__ == "__main__":
    main()