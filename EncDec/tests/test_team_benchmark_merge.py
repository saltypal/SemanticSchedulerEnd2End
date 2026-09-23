"""Ensure teammate research imports and modular asset checks coexist."""
import pytest


def test_historical_research_imports_share_canonical_classes():
    from EncDec.Research.model_wrappers import DeepJSCCAdapter
    from semcom_benchmark.model_wrappers import DeepJSCCAdapter as HistoricalAdapter
    from EncDec.Evaluation import repaired_notebook_models as canonical
    from Evaluation import repaired_notebook_models as historical

    assert HistoricalAdapter is DeepJSCCAdapter
    assert historical is canonical


def test_native_deepjscc_rejects_unhydrated_checkpoint_before_model_loading(tmp_path):
    from EncDec.Evaluation.repaired_notebook_models import build_team_benchmark_codecs

    source = tmp_path / "source"
    source.mkdir()
    pointer = tmp_path / "checkpoint.pth"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:" + "0" * 64 + "\nsize 100\n",
        encoding="utf-8",
    )
    config = {
        "implementation": "deepjscc",
        "device": "cpu",
        "benchmark": {"snr_db": 10},
        "seed": 42,
        "dependencies": {"deepjscc": {"deep_source_root": str(source), "deep_checkpoint": str(pointer)}},
    }

    with pytest.raises(RuntimeError, match="Git LFS asset has not been downloaded"):
        build_team_benchmark_codecs(config)


def test_native_ntscc_requires_only_its_own_checkpoint(tmp_path):
    from EncDec.Evaluation.repaired_notebook_models import build_team_benchmark_codecs

    source = tmp_path / "source"
    source.mkdir()
    checkpoint = tmp_path / "missing.pth"
    config = {
        "implementation": "ntscc",
        "device": "cpu",
        "benchmark": {"snr_db": 10},
        "seed": 42,
        "dependencies": {"ntscc": {"ntscc_source_root": str(source), "ntscc_checkpoint": str(checkpoint)}},
    }

    with pytest.raises(FileNotFoundError) as failure:
        build_team_benchmark_codecs(config)
    assert str(checkpoint) in str(failure.value)
    assert "Deep-JSCC-PyTorch" not in str(failure.value)
