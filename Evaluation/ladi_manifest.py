"""Compatibility alias; canonical implementation: Evaluation.data.ladi_manifest."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('Evaluation.data.ladi_manifest')
