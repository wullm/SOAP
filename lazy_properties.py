#!/bin/env python


def lazy_property(fn):
    """
    Decorator for class methods to turn them into lazily evaluated properties.

    Decorating a function with @lazy_property is practically the same as
    decorating it with @property, except that the function is only evaluated
    once, and only when it is actually used.
    """

    attr_name = "_lazy_" + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property
