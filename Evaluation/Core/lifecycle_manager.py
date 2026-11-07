"""Small lifecycle helper for components with optional close methods."""

from contextlib import ExitStack


class LifecycleManager:
    def __init__(self) -> None:
        self._stack = ExitStack()

    def add(self, component: object) -> object:
        close = getattr(component, "close", None)
        if callable(close):
            self._stack.callback(close)
        return component

    def close(self) -> None:
        self._stack.close()
