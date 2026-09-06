import json

import pytest

from EncDecPipeline.Models.SwinJSCC.checkpoint_manager import MANIFEST_FILENAME, Stage1ArtifactManifest


def test_manifest_rejects_missing_required_fields(tmp_path) -> None:
    (tmp_path / MANIFEST_FILENAME).write_text(json.dumps({"format_version": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing fields"):
        Stage1ArtifactManifest.load_and_validate(tmp_path, strict_runtime=False, validate_source=False)
