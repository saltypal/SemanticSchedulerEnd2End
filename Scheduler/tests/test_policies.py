import pytest
from Evaluation.contracts import Job, QueueSnapshot, ResourceBudget, Decision
from Evaluation.simulation import simulate
from Scheduler.policies import create_policy, POLICIES

def job(key, ue='ue1', arrival=0, units=10, priority=2, deadline=None):
    return Job(key,'image-'+key,ue,arrival,units,priority,deadline)

@pytest.mark.parametrize('name', POLICIES)
def test_empty_and_zero_budget(name):
    policy=create_policy(name)
    assert policy.decide(QueueSnapshot(0,()),ResourceBudget(10)).job_ids == ()
    assert policy.decide(QueueSnapshot(0,(job('a'),)),ResourceBudget(0)).job_ids == ()

def test_tie_break_arrival_and_id_not_input_order():
    queue=QueueSnapshot(10,(job('b'),job('a'),job('future',arrival=11)))
    for name in POLICIES:
        assert create_policy(name).decide(queue,ResourceBudget(10)).job_ids == ('a',)

def test_class_then_deadline_not_global_edf():
    queue=QueueSnapshot(0,(job('a',priority=1,deadline=10),job('b',priority=0,deadline=50),job('c',priority=0,deadline=20)))
    assert create_policy('fixed_priority').decide(queue,ResourceBudget(10)).job_ids == ('b',)
    assert create_policy('priority_edf').decide(queue,ResourceBudget(10)).job_ids == ('c',)

def test_round_robin_rotates_users_and_resets():
    policy=create_policy('round_robin')
    queue=QueueSnapshot(0,(job('a'),job('b'),job('c',ue='ue2')))
    assert policy.decide(queue,ResourceBudget(20,max_jobs=2)).job_ids == ('a','c')
    policy.reset()
    assert policy.decide(queue,ResourceBudget(10)).job_ids == ('a',)

def test_resources_skip_oversize_and_enforce_batch_cap():
    queue=QueueSnapshot(0,(job('a',units=100),job('b',units=6),job('c',units=5),job('d',units=4)))
    decision=create_policy('fifo').decide(queue,ResourceBudget(10,max_jobs=3))
    assert decision.job_ids == ('b','d')
    assert decision.used_units == 10

def test_arrivals_deadlines_and_horizon():
    jobs=[job('a',arrival=10,deadline=20),job('b',deadline=0),job('c',arrival=100),job('d',units=100,deadline=30)]
    result=simulate(jobs,create_policy('fifo'),ResourceBudget(10),slot_ms=10,horizon_ms=30)
    rows={row['job_id']:row for row in result['jobs']}
    assert rows['a']['completion_ms']==20 and rows['a']['deadline_met'] is True
    assert rows['b']['status']=='expired'
    assert rows['c']['status']=='not_arrived'
    assert rows['d']['status']=='expired'
    assert result['decisions'][0]['job_ids']==()

def test_replay_is_deterministic_except_measured_runtime():
    from Scheduler.fixtures import demonstration_jobs
    def trace():
        result=simulate(demonstration_jobs(),create_policy('round_robin'),ResourceBudget(100),horizon_ms=200)
        return result['jobs'],[(row['now_ms'],row['job_ids'],row['used_units']) for row in result['decisions']]
    assert trace()==trace()

def test_reject_invalid_contracts_and_policy():
    with pytest.raises(ValueError): Job('','','',0,0)
    with pytest.raises(ValueError): Job('a','i','u',float('nan'),1)
    with pytest.raises(ValueError): QueueSnapshot(0,(job('a'),job('a')))
    class InvalidPolicy:
        name='invalid'
        def reset(self): pass
        def decide(self,snapshot,budget): return Decision('invalid',0,('a','a'),20,budget.unit,'bad')
    with pytest.raises(ValueError,match='duplicate'):
        simulate([job('a')],InvalidPolicy(),ResourceBudget(20,max_jobs=2))
