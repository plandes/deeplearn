"""Contains classs for creating predictions from a trained model.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import Tuple, List, Any, Type
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from zensols.persist import PersistableContainer, persisted
from zensols.deeplearn.batch import DataPoint, Batch, BatchStash
from zensols.deeplearn.result import ResultsContainer


@dataclass
class PredictionMapper(PersistableContainer, metaclass=ABCMeta):
    """Used by a top level client to create features used to create instances of
    :class:`.DataPoint` and map label classes from nominal IDs to their string
    representations.

    .. document private functions
    .. automethod:: _create_data_point
    .. automethod:: _create_features

    """
    datas: Tuple[Any, ...] = field()
    """The input data to create ad-hoc predictions."""

    batch_stash: BatchStash = field()
    """"The batch stash used to own batches and data points created by this
    instance.

    """
    def __post_init__(self):
        super().__init__()

    @abstractmethod
    def _create_features(self, data: Any) -> Tuple[Any, ...]:
        """Create an instance of a feature from ``data``.

        :param data: data used to create data points

        :return: the data used in the initializer of the respective (in list)
                 :class:`.DataPoint`

        """
        pass

    @abstractmethod
    def map_results(self, result: ResultsContainer) -> Any:
        """Map ad-hoc prediction results from the :class:`.ModelExecutor` to an
        instance that makes sense to the client.

        :param result: contains the predictions produced by the model as
                       :obj:`~zensols.deeplear.result.ResultsContainer.predictions_raw`

        :return: a first class instance suitable for easy client consumption

        """
        pass

    def _create_prediction_batch(self, data: Any) -> Batch:
        dpcls: Type[DataPoint] = self.batch_stash.data_point_type
        features: Tuple[Any, ...] = self._create_features(data)
        dps: Tuple[DataPoint, ...] = tuple(
            map(lambda f: self._create_data_point(dpcls, f), features))
        return self.batch_stash.create_batch(dps)

    @property
    @persisted('_batches')
    def batches(self) -> List[Batch]:
        """Create a prediction batch that is detached from any stash resources,
        except this instance that created it.  This creates a tuple of features,
        each of which is used to create a :class:`.DataPoint`.

        """
        return self._create_batches()

    def _create_batches(self) -> List[Batch]:
        bcls: Type[Batch] = self.batch_stash.batch_type
        batches = []
        for data in self.datas:
            batch: Batch = self._create_prediction_batch(data)
            state = batch.__getstate__()
            dec_batch = object.__new__(bcls)
            dec_batch.__setstate__(state)
            dec_batch.batch_stash = self.batch_stash
            dec_batch.data_points = batch.data_points
            batches.append(dec_batch)
        return batches

    def _create_data_point(self, cls: Type[DataPoint],
                           feature: Any) -> DataPoint:
        """Create a data point.  This base implementation creates it with the
        passed parameters.

        :param cls: the data point class of which to make an instance

        :param stash: to be set as the batch stash on the data point and the
                      caller

        :param feature: to be set as the third argument and generate from
                        :meth:`_create_features`

        """
        return cls(None, self.batch_stash, feature)
