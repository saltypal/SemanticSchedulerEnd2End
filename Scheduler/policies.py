"""Compatibility alias; canonical implementation: Scheduler.baselines.policies."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('Scheduler.baselines.policies')
