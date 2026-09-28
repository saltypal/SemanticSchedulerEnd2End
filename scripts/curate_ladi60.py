"""Draft a real LADI v2 60-image manifest and image contact sheet candidates.

The generated manifest is intentionally marked pending. Inspect related pairs,
enter evidence sufficiency and reviewer, then validate before training.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-split", choices=("train", "validation", "test"),
                        default="validation")
    parser.add_argument("--pool-per-stratum", type=int, default=10)
    parser.add_argument("--revision", default=None,
                        help="Pinned Hugging Face dataset SHA; resolved from main if omitted")
    parser.add_argument("--local-model-only", action="store_true")
    arguments = parser.parse_args()

    from datasets import Image, load_dataset
    from huggingface_hub import HfApi
    from INFRA.ladi_manifest import DATASET_ID, draft_ladi60_manifest, render_contact_sheet
    from SemParam.clip_backbone import CLIPBackbone

    revision = arguments.revision or HfApi().dataset_info(DATASET_ID).sha
    dataset = load_dataset(DATASET_ID, split=arguments.source_split,
                           revision=revision, streaming=True)
    dataset = dataset.cast_column("image", Image(decode=False))
    backend = CLIPBackbone(local_files_only=arguments.local_model_only)
    manifest = draft_ladi60_manifest(
        dataset, arguments.output_dir, dataset_revision=revision,
        source_split=arguments.source_split, embedding_backend=backend,
        pool_per_stratum=arguments.pool_per_stratum,
    )
    contact_sheet = render_contact_sheet(manifest, arguments.output_dir / "ladi60_review.png")
    print(f"Draft manifest: {manifest}")
    print(f"Contact sheet: {contact_sheet}")
    print("Review all 30 proposed image pairs and enter reviewer, sufficiency and any override rationale.")
    print("Training will reject this draft until review_status is 'reviewed' for every row.")


if __name__ == "__main__":
    main()
