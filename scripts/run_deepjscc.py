"""Run DeepJSCC on a real image through the existing communication pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

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


def main() -> None:
    from PIL import Image

    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage:\n"
            "python scripts\\run_deepjscc.py <image_path> [snr_db]\n\n"
            "Example:\n"
            "python scripts\\run_deepjscc.py "
            '"D:\\path\\to\\image.jpg" 10'
        )

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    snr_db = float(sys.argv[2]) if len(sys.argv) >= 3 else 10.0

    # ---------------------------------------------------------
    # Build DeepJSCC
    # ---------------------------------------------------------

    encdec = DeepJSCCAdapter(
        checkpoint_path=CHECKPOINT,
        source_root=SOURCE_ROOT,
        c=19,
        device="cpu",
    ).build()

    # ---------------------------------------------------------
    # Build the existing AWGN channel through our wrapper
    # ---------------------------------------------------------

    channel = DeepJSCCChannelWrapper(
        AWGNChannel()
    )

    # ---------------------------------------------------------
    # Load the real image
    # ---------------------------------------------------------

    image = Image.open(image_path).convert("RGB")

    tensor = pil_to_tensor(image).unsqueeze(0)

    artifact = ImageArtifact(
        tensor=tensor,
        sample_ids=[image_path.stem],
        source=str(image_path),
    )

    # ---------------------------------------------------------
    # Explicitly execute the communication stages.
    #
    # This uses the SAME components as PipelineManager,
    # but lets this runner retain the actual transmission
    # artifact for communication accounting.
    # ---------------------------------------------------------

    latent = encdec.encode(
        artifact
    )

    transmission = channel.transmit(
        latent,
        snr_db,
    )

    reconstruction = encdec.reconstruct_received(
        transmission,
        latent,
    )

    # ---------------------------------------------------------
    # Existing evaluator
    # ---------------------------------------------------------

    evaluator = ImageQualityEvaluator()

    evaluation = evaluator.evaluate(
        artifact,
        reconstruction,
    )

    # ---------------------------------------------------------
    # Actual communication metrics
    # ---------------------------------------------------------

    latent_values = latent.tensor[0].numel()

    padding_values = int(
        transmission.metadata.get(
            "deepjscc_padding_values",
            0,
        )
    )

    channel_uses = transmission.channel_uses_per_image[0]

    height = tensor.shape[-2]
    width = tensor.shape[-1]

    cbr = channel_uses / (
        3 * height * width
    )

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    print()
    print("DEEPJSCC REAL IMAGE EVALUATION")
    print("--------------------------------")

    print("Image:", image_path)
    print("Input shape:", tuple(tensor.shape))
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
        tuple(reconstruction.tensor.shape),
    )

    print(
        "Reconstruction range:",
        float(reconstruction.tensor.min()),
        "to",
        float(reconstruction.tensor.max()),
    )

    print()
    print("CHANNEL")
    print("-------")
    print("Channel:", transmission.channel_name)
    print("SNR (dB):", transmission.snr_db)

    print()
    print("COMMUNICATION METRICS")
    print("---------------------")
    print(
        "Latent real values per image:",
        latent_values,
    )
    print(
        "Padding values:",
        padding_values,
    )
    print(
        "Actual complex channel uses:",
        channel_uses,
    )
    print(
        "CBR:",
        cbr,
    )

    print()
    print("IMAGE QUALITY METRICS")
    print("---------------------")
    print(
        "PSNR (dB):",
        evaluation.metrics["psnr_db"],
    )
    print(
        "SSIM:",
        evaluation.metrics["ssim"],
    )

    print()
    print("RESULT SUMMARY")
    print("--------------")
    print(
        f"DeepJSCC | "
        f"SNR={transmission.snr_db:.1f} dB | "
        f"Channel Uses={channel_uses} | "
        f"CBR={cbr:.6f} | "
        f"PSNR={evaluation.metrics['psnr_db']:.4f} dB | "
        f"SSIM={evaluation.metrics['ssim']:.4f}"
    )

    print()
    print("DeepJSCC evaluation OK")


if __name__ == "__main__":
    main()