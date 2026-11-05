"""Frozen task rubric and UE traffic profiles for class-order experiments.

These labels are evaluator facts, not DQN input. They express one disaster
triage task and are not universal importance labels for LADI images.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from Evaluation.semantic_streams import UEProfile


@dataclass(frozen=True, slots=True)
class ClassPriority:
    priority: int  # 0 is most important; 3 is least important.
    urgency: int   # 3 is most time-sensitive; 0 is least time-sensitive.


CLASS_RUBRIC: Mapping[str, ClassPriority] = {
    "flooding_structures": ClassPriority(0, 3),
    "buildings_affected_or_greater": ClassPriority(1, 2),
    "roads_damage": ClassPriority(1, 3),
    "bridges_any": ClassPriority(1, 2),
    "flooding_any": ClassPriority(1, 2),
    "debris_any": ClassPriority(2, 2),
    "roads_any": ClassPriority(2, 2),
    "buildings_any": ClassPriority(2, 1),
    "buildings_minor_or_greater": ClassPriority(2, 1),
    "trees_damage": ClassPriority(2, 1),
    "trees_any": ClassPriority(3, 1),
    "water_any": ClassPriority(3, 0),
}


# These describe traffic generation, not semantic importance. Class assignment
# is balanced across the UEs by Evaluation.semantic_streams.build_episode.
PRESENTATION_UE_PROFILES = (
    UEProfile("ue_drone", "periodic", interval_ms=20.0, jitter_ms=3.0),
    UEProfile("ue_burst_camera", "burst", interval_ms=60.0, burst_size=3),
    UEProfile("ue_responder", "exponential", interval_ms=25.0),
    UEProfile("ue_survey", "periodic", interval_ms=40.0, jitter_ms=10.0),
    UEProfile("ue_relay", "burst", interval_ms=45.0, burst_size=2),
)


def validate_manifest_rubric(rows: list[dict[str, object]]) -> None:
    """Reject an experiment if the declared rubric drifts from its manifest."""
    if {str(row["selection_stratum"]) for row in rows} != set(CLASS_RUBRIC):
        raise ValueError("Manifest strata do not match the class-priority rubric")
    for row in rows:
        expected = CLASS_RUBRIC[str(row["selection_stratum"])]
        if (int(row["base_priority"]), int(row["urgency"])) != (
                expected.priority, expected.urgency):
            raise ValueError(f"Class rubric differs for image {row['image_id']}")
