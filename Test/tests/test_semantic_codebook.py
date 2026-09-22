from __future__ import annotations

import numpy as np

from CodebookParam.semantic_codebook import CONTENT_PARAMETER_NAMES, SemanticParameterCodebook
from CodebookParam.semantic_codebook import register_semantic_parameter_codebook
from INFRA.Registries import CODEBOOK_REGISTRY, PARAMETER_CODEBOOK_REGISTRY


def test_semantic_parameter_codebook_is_deterministic_and_separate_from_communication_vq(tmp_path) -> None:
    representations = [np.full((1, 4, 3), fill_value=index, dtype=np.float32) for index in range(10)]
    target_rows = [
        {
            "relevance": index / 10.0,
            "reliability": min(1.0, (index + 1) / 10.0),
            "task_importance": 0.5,
        }
        for index in range(10)
    ]
    hashes = [f"hash-{index}" for index in range(10)]

    first = SemanticParameterCodebook.train(
        representations,
        target_rows,
        task_profile_id="emergency_fire",
        training_image_hashes=hashes,
        random_state=7,
    )
    second = SemanticParameterCodebook.train(
        representations,
        target_rows,
        task_profile_id="emergency_fire",
        training_image_hashes=hashes,
        random_state=7,
    )

    assert first.content_hash() == second.content_hash()
    imported_before_prediction = set(__import__("sys").modules)
    prediction = first.predict(representations[0], task_profile_id="emergency_fire")
    assert tuple(prediction.values) == CONTENT_PARAMETER_NAMES
    assert "transformers" not in set(__import__("sys").modules).difference(imported_before_prediction)

    artifact = first.save(tmp_path / "semantic_parameter_demo.npz")
    loaded = SemanticParameterCodebook.load(artifact)
    assert loaded.predict(representations[0], task_profile_id="emergency_fire").values == prediction.values


def test_semantic_parameter_codebook_uses_its_own_registry() -> None:
    register_semantic_parameter_codebook()

    assert "swin_semantic_parameter_codebook" in PARAMETER_CODEBOOK_REGISTRY.names()
    assert "swin_semantic_parameter_codebook" not in CODEBOOK_REGISTRY.names()
