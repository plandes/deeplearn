"""Base classes for overriding optimizers.

"""
__author__ = 'Paul Landes'

from abc import ABC, abstractmethod


class ModelResourceFactory(ABC):
    """An abstract factory that creates either an optimizer or criteria by the
    :class:`.ModelExecutor`.  This is more of a marker class for other sub
    classes to be configured and created by an
    :class:`~zensols.config.ImportConfigFactory`.

    """
    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Create the resource."""
        pass
