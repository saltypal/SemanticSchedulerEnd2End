import hashlib

import numpy as np
from PIL import Image
import pytest
from scripts.build_ladi_review_page import build_review_page

from INFRA.ladi_manifest import (
    NATIVE_LABELS, STRATA, draft_ladi60_manifest,
    validate_ladi60_manifest,
)


class FakeBackend:
    def encode_image(self, path):
        digest = hashlib.sha256(path.read_bytes()).digest()
        return np.asarray([digest[0] + 1, digest[1] + 1, digest[2] + 1],
                          dtype=np.float32)


def test_ladi_draft_preserves_multilabel_fields_and_requires_review(tmp_path):
    rows = []
    for stratum_index, stratum in enumerate(STRATA):
        for image_index in range(10):
            image = Image.new("RGB", (2, 2),
                              (stratum_index * 30, image_index * 20, image_index + 1))
            labels = {name: False for name in NATIVE_LABELS}
            labels["roads_any" if stratum == "infrastructure_control" else stratum] = True
            rows.append({"image": image, **labels})
    manifest = draft_ladi60_manifest(
        rows, tmp_path, dataset_revision="fixture-sha", source_split="train",
        embedding_backend=FakeBackend(), pool_per_stratum=10,
    )
    reviewed_rows = validate_ladi60_manifest(manifest, require_reviewed=False)
    assert len(reviewed_rows) == 60
    assert len({row["evidence_group_id"] for row in reviewed_rows}) == 30
    with pytest.raises(ValueError, match="Human review"):
        validate_ladi60_manifest(manifest)
    review_page = build_review_page(manifest, tmp_path / "review.html")
    assert review_page.read_text(encoding="utf-8").count('class="group"') == 30
    with pytest.raises(ValueError, match="Human review"):
        validate_ladi60_manifest(manifest)
