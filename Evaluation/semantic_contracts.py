"""Compatibility alias; canonical implementation: Evaluation.scheduling.semantic_contracts."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('Evaluation.scheduling.semantic_contracts')
