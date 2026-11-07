from Scheduler.trial_evidence_policy import TrialCandidate, select_trial_candidate


def candidate(token, score, signature, *, arrival=0.0, deadline=100.0):
    return TrialCandidate(token, arrival, deadline, score, signature, 1.0)


def choose(ready, completed=(), *, now=1.0, conditional=True):
    return select_trial_candidate(ready, completed, now, max_wait_ms=20.0,
                                  near_bits=4, receiver_conditioning=conditional)


def test_conditioning_changes_only_the_near_repeat_case():
    repeat = candidate("repeat", 0.95, 0b1)
    fresh = candidate("fresh", 0.55, 0b1111111)
    ready = [repeat, fresh]
    assert choose(ready) == repeat
    assert choose(ready, [0], conditional=False) == repeat
    assert choose(ready, [0]) == fresh


def test_exact_repeat_is_shared_control():
    repeat = candidate("repeat", 0.95, 0)
    fresh = candidate("fresh", 0.55, 0b1111111)
    for conditional in (False, True):
        assert choose([repeat, fresh], [0], conditional=conditional) == fresh


def test_age_guard_prevents_unbounded_preference_for_newer_jobs_in_fixture():
    repeat = candidate("repeat", 0.95, 0)
    fresh = candidate("fresh", 0.55, 0b1111111, arrival=19.0)
    assert choose([repeat, fresh], [0], now=20.0) == repeat


def test_completion_window_and_empty_queue():
    late = candidate("late", 0.95, 0, deadline=1.5)
    timely = candidate("timely", 0.1, 0b1111111)
    assert choose([late, timely]) == timely
    assert choose([]) is None

def test_hard_gate_is_a_distinct_comparator():
    repeat = candidate("near", 0.95, 0b1)
    fresh = candidate("fresh", 0.10, 0b1111111)
    ready = [repeat, fresh]
    assert choose(ready, [0]) == repeat
    assert select_trial_candidate(ready, [0], 1.0, max_wait_ms=20.0, near_bits=4,
                                  receiver_conditioning=False, hard_near_gate=True) == fresh
