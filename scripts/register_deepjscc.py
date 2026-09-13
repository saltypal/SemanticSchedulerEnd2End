"""Register and validate the DeepJSCC component."""

from __future__ import annotations

from EncDecPipeline.Models.DeepJSCC.adapter import register_deepjscc
from INFRA.Registries import ENCDEC_REGISTRY


def main() -> None:
    register_deepjscc()

    print("DEEPJSCC REGISTRY TEST")
    print("----------------------")
    print("Registered:", ENCDEC_REGISTRY.contains("deepjscc"))

    if not ENCDEC_REGISTRY.contains("deepjscc"):
        raise RuntimeError("DeepJSCC was not registered.")

    print("Registry integration OK")


if __name__ == "__main__":
    main()