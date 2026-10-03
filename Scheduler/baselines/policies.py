"""Deterministic nonpreemptive policies. Queue mutation belongs to INFRA."""
from INFRA.scheduling.contracts import Decision, QueueSnapshot, ResourceBudget

POLICIES = (
    'fifo',
    'round_robin',
    'fixed_priority',
    'priority_edf',
    'edf',
    'sjf',
    'aging',
)

class BaselinePolicy:
    def __init__(self, name='fifo', *, aging_threshold_ms=100):
        if name not in POLICIES:
            raise ValueError(f'Unknown policy {name!r}; expected {POLICIES}')
        self.name = name
        self.last_ue = None
        if aging_threshold_ms < 0:
            raise ValueError('aging_threshold_ms must be nonnegative')
        self.aging_threshold_ms = aging_threshold_ms

    def reset(self):
        self.last_ue = None

    def _decide(self, snapshot: QueueSnapshot, budget: ResourceBudget) -> Decision:
        remaining = list(snapshot.eligible())
        selected = []
        used = 0
        while remaining and len(selected) < budget.max_jobs:
            # Skip jobs that cannot fit this slot; do not split a job implicitly.
            fitting = [job for job in remaining if job.resource_units <= budget.units - used]
            if not fitting:
                break
            if self.name == 'round_robin':
                users = sorted({job.ue_id for job in fitting})
                following = [ue for ue in users if self.last_ue is None or ue > self.last_ue]
                user = (following or users)[0]
                job = min((job for job in fitting if job.ue_id == user), key=self._fifo_key)
                self.last_ue = user
            else:
                job = min(fitting, key=self._key)
            remaining.remove(job)
            selected.append(job.job_id)
            used += job.resource_units
        return Decision(self.name, snapshot.now_ms, tuple(selected), used, budget.unit,
                        'selected eligible fitting jobs' if selected else 'no eligible job fits budget')

    @staticmethod
    def _fifo_key(job):
        return (job.arrival_ms, job.job_id)

    def _key(self, job):
        if self.name == 'fixed_priority':
            return (job.priority_class, *self._fifo_key(job))
        if self.name == 'priority_edf':
            deadline = float('inf') if job.deadline_ms is None else job.deadline_ms
            return (job.priority_class, deadline, *self._fifo_key(job))
        if self.name == 'edf':
            deadline = float('inf') if job.deadline_ms is None else job.deadline_ms
            return (deadline, *self._fifo_key(job))
        if self.name == 'sjf':
            return (job.resource_units, *self._fifo_key(job))
        if self.name == 'aging':
            # Starvation prevention: once a job has waited long enough, it
            # receives the strongest effective priority.  FIFO breaks ties.
            waited = max(0, self._now_ms - job.arrival_ms)
            aged = 0 if waited >= self.aging_threshold_ms else 1
            return (aged, job.priority_class, *self._fifo_key(job))
        return self._fifo_key(job)

    def decide(self, snapshot: QueueSnapshot, budget: ResourceBudget) -> Decision:
        self._now_ms = snapshot.now_ms
        return self._decide(snapshot, budget)


def create_policy(name, **options):
    """Create a legacy INFRA policy.

    Semantic scheduling policies use :func:`Scheduler.policy_registry.create_semantic_policy`.
    Keeping this factory preserves the original ``QueueSnapshot`` contract.
    """
    return BaselinePolicy(name, **options)
