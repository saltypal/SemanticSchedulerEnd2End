"""Package the four active owners and explicit historical namespaces."""
from setuptools import find_packages, setup

setup(packages=find_packages(where=".", include=["Codec*", "Scheduler*", "parameters*", "Evaluation*", "Archived", "Archived.owners_before_simplification_20261027*", "scripts*", "semcom_benchmark*"]))
