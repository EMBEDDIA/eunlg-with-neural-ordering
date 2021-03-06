from typing import Any


class ComponentNameCollisionError(Exception):
    pass


class UnknownComponentException(Exception):
    pass


class Registry(object):
    def __init__(self, outdir=None) -> None:
        self._registry = {}

        # added this for easier testing of neural scoring - some parameters given in outdir name
        self.outdir = outdir if outdir else None

    def register(self, name: str, service: Any) -> None:
        if name in self._registry:
            raise ComponentNameCollisionError("A component of name '{}' already exists".format(name))
        else:
            self._registry[name] = service

    def get(self, name: str) -> Any:
        if name not in self._registry:
            raise UnknownComponentException("No component named '{}'".format(name))
        else:
            return self._registry[name]
