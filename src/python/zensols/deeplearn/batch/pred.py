from __future__ import annotations
"""Contains classs for creating predictions from a trained model.

"""
__author__ = 'Paul Landes'

from typing import Tuple, List, Iterable, Any, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass
from . import DataPoint


@dataclass
class PredictionMapper(ABC):
    """Used by a top level client to create features used to create instances of
    :class:`.DataPoint` and map label classes from nominal IDs to their string
    representations.

    """

    def create_data_point(self, cls: Type[DataPoint], stash: BatchStash,
                          feature: Any) -> DataPoint:
        """Create a data point.  This base implementation creates it with the passed
        parameters.

        :param cls: the data point class of which to make an instance

        :param stash: to be set as the batch stash on the data point and the
                      caller

        :param feature: to be set as the third argument and generate from
                        :meth:`create_features`

        """
        return cls(None, stash, feature)

    @abstractmethod
    def create_features(self, data: Any) -> Tuple[Any]:
        """Create an instance of a feature from ``data``.

        :param data: data used to create data points

        :return: the data used in the initializer of the respective (in list)
                 :class:`.DataPoint`

        """
        pass

    @abstractmethod
    def get_classes(self, nominals: Tuple[Iterable[int]]) -> List[List[str]]:
        """Return the label string values for indexes ``nominals``.

        :param nominals: the integers that map to the respective string class;
                         each tuple is a batch, and each item in the iterable
                         is a data point

        :return: a list for every tuple in ``nominals``

        """
        pass
