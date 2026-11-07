"""Compatibility alias; canonical implementation: Scheduler.semantic.registry."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('Scheduler.semantic.registry')
