"""Resolve historical Python names without recreating old top-level folders.

New code uses Codec, Scheduler, parameters and Evaluation. Preserved notebooks
may explicitly call install() in their setup cell before their old imports.
"""
import importlib
import importlib.abc
import importlib.util
import sys

_ALIASES = (
    ('INFRA.learning', 'Scheduler.training'),
    ('EncDec.Evaluation', 'Evaluation.codec_metrics'),
    ('INFRA', 'Evaluation'),
    ('EncDec', 'Codec'),
    ('BetterScheduler', 'parameters'),
    ('SemParam', 'Archived.owners_before_simplification_20261027.SemParam'),
)

class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target):
        self.target = target
        self.canonical_spec = None

    def create_module(self, spec):
        module = importlib.import_module(self.target)
        self.canonical_spec = module.__spec__
        return module

    def exec_module(self, module):
        # Keep serialized class identities and relative-import ownership intact.
        module.__spec__ = self.canonical_spec
        module.__loader__ = self.canonical_spec.loader

class _HistoricalNames(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        for old, new in _ALIASES:
            if fullname == old or fullname.startswith(old + '.'):
                canonical = new + fullname[len(old):]
                spec = importlib.util.find_spec(canonical)
                if spec is None:
                    return None
                return importlib.util.spec_from_loader(fullname, _AliasLoader(canonical),
                    is_package=spec.submodule_search_locations is not None)
        return None

def install():
    if not any(isinstance(finder, _HistoricalNames) for finder in sys.meta_path):
        sys.meta_path.insert(0, _HistoricalNames())
