"""Compatibility alias; canonical implementation: INFRA.data.ladi_manifest."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('INFRA.data.ladi_manifest')
