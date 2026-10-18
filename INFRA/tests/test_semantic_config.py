from dataclasses import replace
from pathlib import Path

import pytest

from INFRA.semantic_config import load_scheduler_config, load_ue_profiles


ROOT = Path(__file__).resolve().parents[2]


def test_locked_demo_configuration_matches_five_ue_profiles():
    config = load_scheduler_config(ROOT / "INFRA/config/semantic_scheduler_demo.yaml")
    profiles = load_ue_profiles(ROOT / "INFRA/config/ue_profiles.yaml")
    assert config.image_count == 60 and config.jobs_per_episode == 60
    assert len(profiles) == config.ue_count


def test_invalid_root_options_fail_before_execution():
    config = load_scheduler_config(ROOT / "INFRA/config/semantic_scheduler_demo.yaml")
    with pytest.raises(ValueError, match="locked"):
        replace(config, jobs_per_episode=25)
    with pytest.raises(ValueError, match="probability"):
        replace(config, robustness_transport_success_probability=1.1)
    with pytest.raises(ValueError, match="deadline_mode"):
        replace(config, deadline_mode="unknown")
