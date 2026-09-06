"""Simple explicit timing for notebook cells and pipeline phases."""

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator


@contextmanager
def timed() -> Iterator[dict[str, float]]:
    result: dict[str, float] = {}
    started = perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = perf_counter() - started
