"""Scheduling policies; importing this package never imports neural models."""
from Scheduler.baselines.policies import BaselinePolicy, POLICIES, create_policy
SemanticScheduler = BaselinePolicy
__all__ = ['BaselinePolicy', 'SemanticScheduler', 'POLICIES', 'create_policy']
