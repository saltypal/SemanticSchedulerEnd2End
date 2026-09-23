"""Deterministic nonpreemptive policies. Queue mutation belongs to INFRA."""
from INFRA.contracts import Decision, QueueSnapshot, ResourceBudget

POLICIES = ('fifo', 'round_robin', 'fixed_priority', 'priority_edf')

class BaselinePolicy:
    def __init__(self, name='fifo'):
        if name not in POLICIES:
            raise ValueError(f'Unknown policy {name!r}; expected {POLICIES}')
        self.name = name
        self.last_ue = None

    def reset(self):
        self.last_ue = None

    def decide(self, snapshot: QueueSnapshot, budget: ResourceBudget) -> Decision:
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
        return self._fifo_key(job)

def create_policy(name):
    return BaselinePolicy(name)
