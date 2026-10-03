"""Compatibility alias; canonical implementation: INFRA.scheduling.requests."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('INFRA.scheduling.requests')
