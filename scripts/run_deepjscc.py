"""Run the verified DeepJSCC implementation through the existing pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

from Channels.awgn import AWGNChannel
from EncDecPipeline.Models.DeepJSCC.adapter import DeepJSCCAdapter
from EncDecPipeline.Models.DeepJSCC.channel_wrapper import DeepJSCCChannelWrapper
from EncDecPipeline.Preprocessing.image_loader import pil_to_tensor
from Evaluation.image_evaluator import ImageQualityEvaluator
from INFRA.Artifacts import ImageArtifact
from INFRA.Core.pipeline_manager import PipelineManager


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


def main() -> None:
    # ---------------------------------------------------------
    # Command-line arguments
    #
    # Usage:
    # python scripts\run_deepjscc.py
    #
    # Or:
    # python scripts\run_deepjscc.py "path\to\image.png" 10
    # ---------------------------------------------------------

    image_path = (
        Path(sys.argv[1])
        if len(sys.argv) >= 2
        else DEFAULT_IMAGE
    )

    snr_db = (
        float(sys.argv[2])
        if len(sys.argv) >= 3
        else 10.0
    )

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"DeepJSCC checkpoint not found:\n{CHECKPOINT}"
        )

    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(
            f"DeepJSCC source directory not found:\n{SOURCE_ROOT}"
        )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Input image not found:\n{image_path}"
        )

    # ---------------------------------------------------------
    # Load the original image.
    # ---------------------------------------------------------

    image = Image.open(image_path).convert("RGB")

    # This follows the repo preprocessing convention:
    # RGB uint8 image -> float tensor in [0, 1].
    tensor = pil_to_tensor(image).unsqueeze(0)

    # ---------------------------------------------------------
    # Create the ImageArtifact expected by the existing
    # semantic communication pipeline.
    # ---------------------------------------------------------

    artifact = ImageArtifact(
        tensor=tensor,
        sample_ids=[image_path.stem],
        source="DeepJSCC",
    )

    # ---------------------------------------------------------
    # Build the verified DeepJSCC implementation.
    #
    # The adapter internally uses:
    #   old_model_6.py
    #   c=19
    #   checkpoint from the original DeepJSCC implementation
    #
    # We keep the original notebook's model configuration.
    # ---------------------------------------------------------

    encdec = DeepJSCCAdapter(
        checkpoint_path=CHECKPOINT,
        source_root=SOURCE_ROOT,
        c=19,
        device="cpu",
        channel_type="AWGN",
        snr=200,
    ).build()

    # ---------------------------------------------------------
    # Reuse the EXISTING repository AWGN implementation.
    #
    # DeepJSCCChannelWrapper only handles the odd number of
    # latent real values. AWGN itself is not modified.
    # ---------------------------------------------------------

    channel = DeepJSCCChannelWrapper(
        AWGNChannel()
    )

    # ---------------------------------------------------------
    # Reuse the EXISTING repository evaluator.
    # ---------------------------------------------------------

    evaluator = ImageQualityEvaluator()

    # ---------------------------------------------------------
    # Use the existing PipelineManager.
    # ---------------------------------------------------------

    pipeline = PipelineManager(
        encdec=encdec,
        channel=channel,
        evaluators=[evaluator],
    )

    # ---------------------------------------------------------
    # Run the complete pipeline.
    # ---------------------------------------------------------

    result = pipeline.run_image(
        image=artifact,
        snr_db=snr_db,
    )

    # ---------------------------------------------------------
    # Extract communication information.
    # ---------------------------------------------------------

    latent = encdec.encode(artifact)

    latent_values = int(
        latent.tensor.reshape(latent.tensor.shape[0], -1).shape[1]
    )

    padding_values = latent_values % 2

    padded_values = latent_values + padding_values

    complex_channel_uses = padded_values // 2

    original_pixels = (
        tensor.shape[2] * tensor.shape[3]
    )

    # Notebook-style CBR:
    # latent real-valued elements / RGB source elements.
    notebook_cbr = latent_values / (
        3 * original_pixels
    )

    # ---------------------------------------------------------
    # Print results.
    # ---------------------------------------------------------

    print()
    print("DEEPJSCC REAL IMAGE EVALUATION")
    print("--------------------------------")
    print(f"Image: {image_path}")
    print(f"Input shape: {tuple(tensor.shape)}")
    print(
        "Input range:",
        float(tensor.min()),
        "to",
        float(tensor.max()),
    )

    print(
        "Latent shape:",
        tuple(latent.tensor.shape),
    )

    print(
        "Reconstruction shape:",
        tuple(result.reconstruction.tensor.shape),
    )

    print(
        "Reconstruction range:",
        float(result.reconstruction.tensor.min()),
        "to",
        float(result.reconstruction.tensor.max()),
    )

    print()
    print("CHANNEL")
    print("-------")
    print("Channel: awgn")
    print(f"SNR (dB): {snr_db}")

    print()
    print("COMMUNICATION METRICS")
    print("---------------------")
    print(
        f"Latent real values per image: {latent_values:,}"
    )
    print(
        f"Padding values: {padding_values}"
    )
    print(
        f"Padded real values: {padded_values:,}"
    )
    print(
        f"Actual complex channel uses: {complex_channel_uses:,}"
    )
    print(
        f"Notebook-style CBR: {notebook_cbr:.6f}"
    )

    print()
    print("IMAGE QUALITY METRICS")
    print("---------------------")

    if result.evaluations:
        metrics = result.evaluations[0].metrics

        print(
            f"PSNR (dB): {metrics['psnr_db']:.6f}"
        )
        print(
            f"SSIM: {metrics['ssim']:.6f}"
        )

    print()
    print("RESULT SUMMARY")
    print("--------------")

    if result.evaluations:
        metrics = result.evaluations[0].metrics

        print(
            f"DeepJSCC | "
            f"SNR={snr_db} dB | "
            f"Channel Uses={complex_channel_uses:,} | "
            f"CBR={notebook_cbr:.6f} | "
            f"PSNR={metrics['psnr_db']:.4f} dB | "
            f"SSIM={metrics['ssim']:.4f}"
        )

    print()
    print("DeepJSCC evaluation OK")


if __name__ == "__main__":
    main()