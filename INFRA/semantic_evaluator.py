"""Compatibility alias; canonical implementation: INFRA.evaluation.semantic."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('INFRA.evaluation.semantic')
