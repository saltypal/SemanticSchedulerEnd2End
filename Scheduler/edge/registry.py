"""Policy registry: one place that merges every module's ``POLICIES`` dict.

Names come from config/edge/policies.yaml; this module only maps a name to a factory and
checks that every configured name exists (so a typo fails loudly, not silently at run time).
"""
from importlib import import_module

from Evaluation.edge.edge_config import load_config

_MODULES = ("Scheduler.edge.classic", "Scheduler.edge.theory", "Scheduler.edge.edgeserving",
            "Scheduler.edge.sv_dpp", "Scheduler.edge.mpc", "Scheduler.edge.extended",
            "Scheduler.edge.batching", "Scheduler.edge.semcover", "Scheduler.edge.bcedge")


def all_policies():
    """Merge POLICIES from every policy module that exists. Duplicate names are an error."""
    merged = {}
    for module_name in _MODULES:
        try:
            module = import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name != module_name:        # a real missing dependency, not an absent module
                raise
            continue
        for name, factory in module.POLICIES.items():
            if name in merged:
                raise ValueError("duplicate policy name: " + name)
            merged[name] = factory
    return merged


def build(name, **kwargs):
    policies = all_policies()
    if name not in policies:
        raise KeyError("unknown policy %r; known: %s" % (name, sorted(policies)))
    return policies[name](**kwargs)


def configured_names():
    cfg = load_config("policies")
    return list(cfg["main_table"]) + list(cfg["equivalence_appendix"]) + list(cfg["references"])


def missing_configured():
    """Names listed in config/edge/policies.yaml that no module provides yet."""
    have = set(all_policies())
    return [n for n in configured_names() if n not in have]
