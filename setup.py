"""Discover canonical root packages and temporary src compatibility packages."""
from setuptools import find_packages, setup

canonical = find_packages(where=".", include=["EncDec*", "SemParam*", "Scheduler*", "INFRA*"])
legacy = find_packages(
    where="src",
    include=["EncDecPipeline*", "TransformerPipeline*", "CodebookParam*", "Channels*", "Baselines*", "Evaluation*", "utils*"],
)
setup(packages=sorted(set(canonical + legacy)))
