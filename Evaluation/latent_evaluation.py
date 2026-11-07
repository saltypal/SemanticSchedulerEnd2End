"""Compatibility alias; canonical implementation: Evaluation.latent.evaluation."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('Evaluation.latent.evaluation')
