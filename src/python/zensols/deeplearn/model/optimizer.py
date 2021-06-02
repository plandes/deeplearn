"""Base classes for overriding optimizers.

"""
__author__ = 'Paul Landes'

from abc import ABC, abstractmethod


class ModelResourceFactory(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass
