from __future__ import annotations
"""Contains classs for creating predictions from a trained model.

"""
__author__ = 'Paul Landes'

from typing import Tuple, List, Iterable, Any, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from zensols.deeplearn.batch import DataPoint, Batch, BatchStash


@dataclass
class PredictionMapper(ABC):
    """Used by a top level client to create features used to create instances of
    :class:`.DataPoint` and map label classes from nominal IDs to their string
    representations.

    .. document private functions
    .. automethod:: _create_data_point
    .. automethod:: _create_features

    """
    datas: Tuple[Any] = field()
    """The input data to create ad-hoc predictions."""

    batch_stash: BatchStash = field()
    """"The batch stash used to own batches and data points created by this
    instance.

    """

    def _create_prediction_batch(self, data: Any) -> Batch:
        dpcls: Type[DataPoint] = self.batch_stash.data_point_type
        bcls: Type[Batch] = self.batch_stash.batch_type
        features: Tuple[Any] = self._create_features(data)
        dps: Tuple[DataPoint] = tuple(
            map(lambda f: self._create_data_point(dpcls, f), features))
        return bcls(self.batch_stash, None, None, dps)

    def create_batches(self) -> List[Batch]:
        """Create a prediction batch that is detached from any stash resources, except
        this instance that created it.  This creates a tuple of features, each
        of which is used to create a :class:`.DataPoint`.

        """
        bcls: Type[Batch] = self.batch_stash.batch_type
        batches = []
        for data in self.datas:
            batch: Batch = self._create_prediction_batch(data)
            state = batch.__getstate__()
            dec_batch = bcls.__new__(bcls)
            dec_batch.__setstate__(state)
            dec_batch.batch_stash = self.batch_stash
            dec_batch.data_points = batch.data_points
            batches.append(dec_batch)
        return batches

    def _create_data_point(self, cls: Type[DataPoint],
                           feature: Any) -> DataPoint:
        """Create a data point.  This base implementation creates it with the passed
        parameters.

        :param cls: the data point class of which to make an instance

        :param stash: to be set as the batch stash on the data point and the
                      caller

        :param feature: to be set as the third argument and generate from
                        :meth:`_create_features`

        """
        return cls(None, self.batch_stash, feature)

    @abstractmethod
    def _create_features(self, data: Any) -> Tuple[Any]:
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
