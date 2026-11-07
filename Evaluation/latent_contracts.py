"""Compatibility alias; canonical implementation: Evaluation.latent.contracts."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('Evaluation.latent.contracts')
