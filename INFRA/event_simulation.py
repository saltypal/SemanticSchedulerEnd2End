"""Compatibility alias; canonical implementation: INFRA.scheduling.event_simulation."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('INFRA.scheduling.event_simulation')
