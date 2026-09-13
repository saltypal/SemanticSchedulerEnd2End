"""Validate DeepJSCC construction through the project registry."""

from __future__ import annotations

from pathlib import Path

from EncDecPipeline.Models.DeepJSCC.adapter import register_deepjscc
from INFRA.Registries import ENCDEC_REGISTRY


CHECKPOINT = Path(
    r"D:\OneDrive - Amrita vishwa vidyapeetham\Desktop\image_semcom"
    r"\Deep-JSCC-PyTorch\out\imagenet_10_0.33_200.00_32_19.pth"
)

SOURCE_ROOT = Path(
    r"D:\OneDrive - Amrita vishwa vidyapeetham\Desktop\image_semcom"
    r"\Deep-JSCC-PyTorch"
)


def main() -> None:
    register_deepjscc()

    model = ENCDEC_REGISTRY.create(
        "deepjscc",
        checkpoint_path=CHECKPOINT,
        source_root=SOURCE_ROOT,
        c=19,
        device="cpu",
    )

    model.build()

    print("DEEPJSCC REGISTRY CONSTRUCTION TEST")
    print("-----------------------------------")
    print("Registry names:", ENCDEC_REGISTRY.names())
    print("Model type:", type(model).__name__)
    print("Device:", model.device)
    print("Checkpoint exists:", CHECKPOINT.exists())
    print("Registry construction OK")


if __name__ == "__main__":
    main()