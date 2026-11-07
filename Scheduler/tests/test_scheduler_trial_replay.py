from Evaluation.scheduler_trial_replay import MeasuredJob, replay
from Scheduler.trial_evidence_policy import TrialCandidate


def job(token, arrival, service, fact=None, report=False, failed=False, signature=0, score=1.0):
    candidate = TrialCandidate(token, arrival, 10.0, score, signature, 1.0)
    return MeasuredJob(candidate, fact, report, service, failed)


def test_failing_or_unfinished_jobs_never_credit_evidence():
    offers = [job("failed", 0, 1, "a", True, True),
              job("unfinished", 2, 10, "b", True),
              job("still_waiting", 3, 1, "c", True)]
    result = replay(offers, 4, "fifo", 5)
    assert result["order"] == ["failed", "unfinished"]
    assert result["timely_facts"] == 0
    assert result["failed"] == result["unfinished"] == result["unstarted"] == 1


def test_false_report_can_change_the_next_conditional_decision():
    offers = [job("false", 0, 2, report=True),
              job("near", 1, 1, "a", True, signature=1),
              job("far", 1, 1, "b", True, signature=255, score=0.5)]
    static = replay(offers, 9, "static", 5)
    conditional = replay(offers, 9, "conditional", 5)
    assert static["order"] == ["false", "near", "far"]
    assert conditional["order"] == ["false", "far", "near"]
    assert conditional["false_reports"] == 1


def test_repeat_positive_counts_once():
    offers = [job("first", 0, 1, "road", True), job("repeat", 1, 1, "road", True)]
    result = replay(offers, 9, "static", 5)
    assert result["offered_facts"] == result["timely_facts"] == 1
