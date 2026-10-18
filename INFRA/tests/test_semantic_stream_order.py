from dataclasses import replace

from INFRA.semantic_contracts import (
    EpisodeJobSpec, FeatureValue, LatentJobSpec, SchedulingFeatureVectorV2,
)
from INFRA.semantic_streams import UEProfile, build_episode, with_task_coupled_deadlines


def test_stratum_balancing_does_not_sort_the_arrival_stream():
    value = FeatureValue(1, True, "fixture", "v2")
    features = SchedulingFeatureVectorV2(
        value, FeatureValue.unavailable("fixture", "v2"),
        value, value, value, (1.0, 0.0),
    )
    candidate = LatentJobSpec(
        "base", "base", "ue", "task", "base", 0, 100,
        10, "simulated_units", features,
    )
    templates = tuple(
        EpisodeJobSpec(replace(candidate, job_id=str(i), image_id=str(i)),
                       i % 4, f"stratum{i}-group1")
        for i in range(6)
    )
    profiles = (UEProfile("ue", "periodic", 10),)
    orders = set()
    for seed in range(30):
        kwargs = dict(episode_id="test", task_id="task", seed=seed,
                      ue_profiles=profiles)
        episode = build_episode(templates, **kwargs)
        assert episode == build_episode(templates, **kwargs)
        order = tuple(job.candidate.image_id for job in episode.jobs)
        assert set(order) == {str(i) for i in range(6)}
        orders.add(order)
    # The old implementation produced the same alphabetical stratum order
    # for every seed, despite random assignment and arrival timing.
    assert len(orders) > 20


def test_network_deadlines_do_not_encode_task_urgency():
    urgent = FeatureValue(3, True, "fixture", "v2")
    normal = FeatureValue(0, True, "fixture", "v2")
    value = FeatureValue(1, True, "fixture", "v2")
    templates = []
    for index, urgency in enumerate((urgent, normal)):
        features = SchedulingFeatureVectorV2(
            value, FeatureValue.unavailable("fixture", "v2"),
            value, urgency, value, (1.0, 0.0),
        )
        candidate = LatentJobSpec(
            str(index), str(index), "ue", "task", str(index),
            0, 100, 10, "simulated_units", features,
        )
        templates.append(EpisodeJobSpec(candidate, index * 3, f"group{index}"))
    kwargs = dict(episode_id="deadline", task_id="task", seed=42,
                  ue_profiles=(UEProfile("ue", "periodic", 10),))
    independent = build_episode(templates, **kwargs)
    assert all(70 <= item.candidate.deadline_ms - item.candidate.arrival_ms <= 170
               for item in independent.jobs)
    coupled = build_episode(templates, deadline_mode="task_coupled", **kwargs)
    offsets = {item.candidate.image_id:
               item.candidate.deadline_ms - item.candidate.arrival_ms
               for item in coupled.jobs}
    assert 40 <= offsets["0"] <= 80
    assert 300 <= offsets["1"] <= 500
    sensitivity = with_task_coupled_deadlines(independent)
    assert sensitivity == with_task_coupled_deadlines(independent)
    assert [(item.candidate.image_id, item.candidate.arrival_ms)
            for item in sensitivity.jobs] == [
                (item.candidate.image_id, item.candidate.arrival_ms)
                for item in independent.jobs]
