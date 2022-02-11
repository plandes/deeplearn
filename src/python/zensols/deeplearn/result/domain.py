from __future__ import annotations
"""Contains contain classes for results generated from training and testing a
model.

"""
__author__ = 'Paul Landes'

from typing import List, Dict, Set, Iterable, Any, Type, Tuple, Callable
from dataclasses import dataclass, field, InitVar
from enum import Enum
from abc import ABCMeta, abstractmethod
import logging
import sys
import copy as cp
from collections import OrderedDict
from itertools import chain
from datetime import datetime
from io import TextIOBase
import math
import sklearn.metrics as mt
import numpy as np
from torch import Tensor
from zensols.config import Configurable, Dictable
from zensols.deeplearn import (
    DeepLearnError, DatasetSplitType, ModelSettings, NetworkSettings
)
from zensols.deeplearn.batch import Batch

logger = logging.getLogger(__name__)


class ModelResultError(DeepLearnError):
    """"Thrown when results can not be compiled or computed."""
    pass


class NoResultError(ModelResultError):
    """Convenience used for helping debug the network.

    """
    def __init__(self, cls: Type):
        super().__init__(f'{cls}: no results available')


class ModelType(Enum):
    """The type of model give by the type of its output.

    """
    PREDICTION = 0
    CLASSIFICTION = 1
    RANKING = 2


@dataclass
class Metrics(Dictable):
    """A container class that provides results for data stored in a
    :class:`.ResultsContainer`.

    """
    labels: np.ndarray = field(repr=False)
    predictions: np.ndarray = field(repr=False)

    @property
    def contains_results(self) -> bool:
        """Return ``True`` if this container has results.

        """
        return len(self) > 0

    def _protect(self, fn: Callable):
        if self.contains_results:
            return fn()
        else:
            return math.nan

    def __len__(self) -> int:
        shape = self.predictions.shape
        assert len(shape) == 1
        return shape[0]


@dataclass
class PredictionMetrics(Metrics):
    """Real valued prediction results for :obj:`.ModelType.PREDICTION` result.

    """
    @property
    def root_mean_squared_error(self) -> float:
        """Return the root mean squared error metric.

        """
        return self._protect(lambda: math.sqrt(
            mt.mean_squared_error(self.labels, self.predictions)))

    @property
    def mean_absolute_error(self) -> float:
        """Return the mean absolute error metric.

        """
        return self._protect(
            lambda: mt.mean_absolute_error(self.labels, self.predictions))

    @property
    def r2_score(self) -> float:
        """Return the R^2 score metric.

        """
        return self._protect(
            lambda: mt.r2_score(self.labels, self.predictions))

    @property
    def correlation(self) -> float:
        """Return the correlation metric.

        """
        return self._protect(
            lambda: np.corrcoef(self.labels, self.predictions)[0][1])

    def _get_dictable_attributes(self) -> Iterable[Tuple[str, str]]:
        return (('rmse', 'root_mean_squared_error'),
                ('mae', 'mean_absolute_error'),
                ('r2', 'r2_score'),
                ('correlation', 'correlation'))

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(f'RMSE: {self.root_mean_squared_error:.3f}',
                         depth, writer)
        self._write_line(f'MAE: {self.mean_absolute_error:.3f}', depth, writer)
        self._write_line(f'R^2: {self.r2_score:.3f}', depth, writer)
        self._write_line(f"correlation: {self.correlation:.3f}", depth, writer)

    def __str__(self):
        return (f'rmse: {self.root_mean_squared_error:.3f}, ' +
                f'mae: {self.mean_absolute_error:.3f}, ' +
                f'r2: {self.r2_score:.3f}, ' +
                f'corr: {self.correlation:.3f}')


@dataclass
class ScoreMetrics(Metrics):
    average: str = field()
    """The type of average to apply to metrics produced by this class."""

    @property
    def f1(self) -> float:
        """Return the F1 metric as either the micro or macro based on the
        :obj:`average` attribute.

        """
        return self._protect(lambda: mt.f1_score(
            self.labels, self.predictions, average=self.average))

    @property
    def precision(self) -> float:
        """Return the precision metric as either the micro or macro based on the
        :obj:`average` attribute.

        """
        return self._protect(
            lambda: mt.precision_score(
                self.labels, self.predictions, average=self.average,
                # clean up warning for tests: sklearn complains with
                # UndefinedMetricWarning even though the data looks good
                zero_division=0))

    @property
    def recall(self) -> float:
        """Return the recall metric as either the micro or macro based on the
        :obj:`average` attribute.

        """
        return self._protect(lambda: mt.recall_score(
            self.labels, self.predictions, average=self.average))

    @property
    def long_f1_name(self) -> str:
        return f'{self.average}-F1'

    @property
    def short_f1_name(self) -> str:
        name = 'm' if self.average == 'micro' else 'M'
        return f'{name}F1'

    def _get_dictable_attributes(self) -> Iterable[Tuple[str, str]]:
        return self._split_str_to_attributes('f1 precision recall')

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(f'{self.average}: ' +
                         f'F1: {self.f1:.3f}, ' +
                         f'precision: {self.precision:.3f}, ' +
                         f'recall: {self.recall:.3f}', depth, writer)

    def __str__(self):
        return f'{self.short_f1_name}: {self.f1:.3f}'


@dataclass
class ClassificationMetrics(Metrics):
    """Real valued prediction results for :obj:`.ModelType.CLASSIFICATION`
    result.

    """
    n_outcomes: int = field()
    """The number of outcomes given for this metrics set."""

    def _predictions_empty(self):
        if self.__len__() == 0:
            return np.NaN

    @property
    def accuracy(self) -> float:
        """Return the accuracy metric (num correct / total).

        """
        return self._protect(
            lambda: mt.accuracy_score(self.labels, self.predictions))

    @property
    def n_correct(self) -> int:
        """The number or correct predictions for the classification.

        """
        is_eq = np.equal(self.labels, self.predictions)
        return self._protect(lambda: np.count_nonzero(is_eq == True))

    def create_metrics(self, average: str) -> ScoreMetrics:
        """Create a score metrics with the given average.

        """
        return ScoreMetrics(self.labels, self.predictions, average)

    @property
    def micro(self) -> ScoreMetrics:
        """Compute micro F1, precision and recall.

        """
        return self.create_metrics('micro')

    @property
    def macro(self) -> ScoreMetrics:
        """Compute macro F1, precision and recall.

        """
        return self.create_metrics('macro')

    @property
    def weighted(self) -> ScoreMetrics:
        """Compute weighted F1, precision and recall.

        """
        return self.create_metrics('weighted')

    def _get_dictable_attributes(self) -> Iterable[Tuple[str, str]]:
        return self._split_str_to_attributes(
            'accuracy n_correct micro macro')

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        if self.n_outcomes == 0:
            self._write_line('no results', depth, writer)
        else:
            self._write_line(f'accuracy: {self.accuracy:.3f} ' +
                             f'({self.n_correct}/{self.n_outcomes})',
                             depth, writer)
            self.micro.write(depth, writer)
            self.macro.write(depth, writer)
            self.weighted.write(depth, writer)

    def __str__(self):
        return str(self.micro)


@dataclass
class ResultsContainer(Dictable, metaclass=ABCMeta):
    """The base class for all metrics containers.  It helps in calculating loss,
    finding labels, predictions and other utility helpers.

    Every container has a start and stop time, which demarcates the duration
    the for which the populated metrics were being calculated.

    """
    FLOAT_TYPES = [np.float32, np.float64, float]
    """Used to determin the :obj:`model_type`."""

    def __post_init__(self):
        super().__init__()
        self.start_time = None
        self.end_time = None

    @property
    def is_started(self) -> bool:
        """The time at which processing started for the metrics populated in this
        container.

        :see: meth:`start`

        """
        return self.start_time is not None

    @property
    def is_ended(self) -> bool:
        """The time at which processing ended for the metrics populated in this
        container.

        :see: meth:`end`

        """
        return self.end_time is not None

    def start(self) -> datetime:
        """Record the time at which processing started for the metrics populated in
        this container.

        :see: obj:`is_started`

        """
        if self.start_time is not None:
            raise ModelResultError(
                f'Container has already tarted: {self}')
        if self.contains_results:
            raise ModelResultError(f'Container {self} already has results')
        self.start_time = datetime.now()
        return self.start_time

    def end(self) -> datetime:
        """Record the time at which processing started for the metrics populated in
        this container.

        :see: obj:`is_ended`

        """
        if self.start_time is None:
            raise ModelResultError(f'Container has not yet started: {self}')
        self._assert_finished(False)
        self.end_time = datetime.now()
        return self.end_time

    def _assert_finished(self, should_be: bool):
        """Make sure we've either finished or not based on ``should_be``."""
        if should_be:
            if not self.is_ended:
                raise ModelResultError(f'Container is not finished: {self}')
        else:
            if self.is_ended:
                raise ModelResultError(
                    f'Container has finished: {self}')

    def clone(self) -> ResultsContainer:
        """Return a clone of the current container.  Sub containers (lists) are deep
        copied in sub classes, but everything is shallow copied.

        This is needed to create a temporary container to persist whose
        :meth:`end` gets called by the
        :class:`~zensols.deeplearn.model.ModelExecutor`.

        """
        return cp.copy(self)

    @property
    def contains_results(self):
        """``True`` if this container has results.

        """
        return len(self) > 0

    @property
    def min_loss(self) -> float:
        """The lowest loss recorded in this container.

        """
        self._assert_finished(True)
        return min(self.losses)

    @property
    def max_loss(self) -> float:
        """The highest loss recorded in this container.

        """
        self._assert_finished(True)
        return max(self.losses)

    @property
    def ave_loss(self) -> float:
        """The average loss of this result set.

        """
        self._assert_finished(True)
        losses = self.losses
        d = len(losses)
        return (sum(losses) / d) if d > 0 else 0

    @property
    def n_outcomes(self) -> int:
        """The number of outcomes.

        """
        return self.predictions.shape[0]

    @property
    def n_iterations(self) -> int:
        """The number of iterations, which is different from the :obj:`n_outcomes`
        since a single (say training) iteration can produce multiple outcomes
        (for example sequence classification).

        """
        return self._get_iterations()

    @property
    def model_type(self) -> ModelType:
        """The type of the model based on what whether the outcome data is a
        float or integer.

        """
        arr = self.predictions
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'outcomes type: {arr.dtype}')
        if arr.dtype in self.FLOAT_TYPES:
            return ModelType.PREDICTION
        else:
            return ModelType.CLASSIFICTION

    @abstractmethod
    def _get_labels(self) -> np.ndarray:
        pass

    @abstractmethod
    def _get_predictions(self) -> np.ndarray:
        pass

    @abstractmethod
    def _get_iterations(self) -> int:
        pass

    @property
    def labels(self) -> np.ndarray:
        """The labels or ``None`` if none were provided (i.e. during
        test/evaluation).

        """
        self._assert_finished(True)
        return self._get_labels()

    @property
    def predictions(self) -> np.ndarray:
        """The predictions from the model.  This also flattens the predictions in to a
        1D array for the purpose of computing metrics.

        :return: the flattened predictions

        """
        self._assert_finished(True)
        return self._get_predictions()

    @property
    def prediction_metrics(self) -> PredictionMetrics:
        """Return prediction based metrics.

        """
        return PredictionMetrics(self.labels, self.predictions)

    @property
    def classification_metrics(self) -> ClassificationMetrics:
        """Return classification based metrics.

        """
        return ClassificationMetrics(
            self.labels, self.predictions, self.n_outcomes)

    @property
    def metrics(self) -> Metrics:
        """Return the metrics based on the :obj:`model_type`.

        """
        mtype = self.model_type
        if mtype == ModelType.CLASSIFICTION:
            metrics = self.classification_metrics
        elif mtype == ModelType.PREDICTION:
            metrics = self.prediction_metrics
        else:
            raise ModelResultError(f'Unknown or unsupported tupe: {mtype}')
        return metrics

    def _get_dictable_attributes(self) -> Iterable[Tuple[str, str]]:
        return self._split_str_to_attributes('n_outcomes metrics')

    def __str__(self):
        return (f'{self.__class__.__name__}: ' +
                f'start: {self.start_time}, end: {self.end_time}')

    def __repr__(self):
        return self.__str__()


@dataclass
class EpochResult(ResultsContainer):
    """Contains results recorded from an epoch of a neural network model.  This is
    during a training/validation or test cycle.

    Note that there is a terminology difference between what the model and the
    result set call outcomes.  For the model, outcomes are the mapped/refined
    results, which are usually the argmax of the softmax of the logits.  For
    results, these are the predictions of the given data to be compared against
    the gold labels.

    """
    _RES_ARR_NAMES = 'label pred'.split()

    index: int = field()
    """The Nth epoch of the run (across training, validation, test)."""

    split_type: DatasetSplitType = field()
    """The name of the split type (i.e. ``train`` vs ``test``)."""

    batch_losses: List[float] = field(default_factory=list)
    """The losses generated from each iteration of the epoch."""

    batch_ids: List[int] = field(default_factory=list)
    """The ID of the batch from each iteration of the epoch."""

    n_data_points: List[int] = field(default_factory=list)
    """The number of data points for each batch for the epoch."""

    def __post_init__(self):
        super().__post_init__()
        self._predictions = []
        self._labels = []
        self._outputs = []

    def update(self, batch: Batch, loss: Tensor, labels: Tensor, preds: Tensor,
               outputs: Tensor):
        """Add another set of statistics, predictions and gold labels to
        :obj:`prediction_updates`.

        :param batch: the batch on which the stats/data where trained, tested
                      or validated; used to update the loss as a multiplier on
                      its size

        :param loss: the loss returned by the loss function

        :param labels: the gold labels, or ``None`` if this is a prediction run

        :param preds: the predictions, or ``None`` for scored models (see
                      :obj:`prediction_updates`)

        """
        self._assert_finished(False)
        shape = preds.shape if labels is None else labels.shape
        assert shape is not None
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'{self.index}:{self.split_type}: ' +
                         f'update batch: {batch.id}, ' +
                         f'label_shape: {shape}')
        # object function loss; 'mean' is the default 'reduction' parameter for
        # loss functions; we can either muliply it back out or use 'sum' in the
        # criterion initialize
        if loss is None:
            self.batch_losses.append(-1)
        else:
            self.batch_losses.append(loss.item() * float(batch.size()))
        # batches are always the first dimension
        self.n_data_points.append(shape[0])
        # add predictions that exist
        if preds is not None:
            self._predictions.append(preds.numpy())
            # see end() comments: without predictions, labels are useless
            if labels is not None:
                self._labels.append(labels.numpy())
        if outputs is not None:
            self._outputs.append(outputs.numpy())
        self.batch_ids.append(batch.id)

    def end(self):
        super().end()
        labs = preds = None
        # if there are no predictions (the case from the training phase), don't
        # include any data since labels by themselves are useless for all use
        # cases (metrics, scoring, certainty assessment, and any analysis etc)
        if len(self._predictions) > 0:
            if len(self._labels) > 0:
                labs = tuple(map(lambda arr: arr.flatten(), self._labels))
                labs = np.concatenate(labs, axis=0)
            preds = tuple(map(lambda arr: arr.flatten(), self._predictions))
            preds = np.concatenate(preds, axis=0)
        if labs is None:
            labs = np.array([], dtype=np.int64)
        if preds is None:
            preds = np.array([], dtype=np.int64)
        self._all_labels = labs
        self._all_predictions = preds

    def clone(self) -> ResultsContainer:
        cl = cp.copy(self)
        for attr in 'batch_losses batch_ids n_data_points'.split():
            setattr(cl, attr, list(getattr(self, attr)))
        return cl

    @property
    def batch_predictions(self) -> List[np.ndarray]:
        """The batch predictions given in the shape as output from the model.

        """
        return self._predictions

    @property
    def batch_labels(self) -> List[np.ndarray]:
        """The batch labels given in the shape as output from the model.

        """
        return self._labels

    @property
    def batch_outputs(self) -> List[np.ndarray]:
        return self._outputs

    def _get_labels(self) -> np.ndarray:
        return self._all_labels

    def _get_predictions(self) -> np.ndarray:
        return self._all_predictions

    def _get_iterations(self) -> int:
        return int(self.batch_losses)

    @property
    def losses(self) -> List[float]:
        """Return the loss for each epoch of the run.  If used on a ``EpocResult`` it
        is the Nth iteration.

        """
        return self.batch_losses

    def _get_dictable_attributes(self) -> Iterable[Tuple[str, str]]:
        return chain.from_iterable(
            (super()._get_dictable_attributes(),
             self._split_str_to_attributes('index')))

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        bids = ','.join(self.batch_ids)
        dps = ','.join(map(str, self.n_data_points))
        self._write_line(f'index: {self.index}', depth, writer)
        self._write_line(f'batch IDs: {bids}', depth, writer, True)
        self._write_line(f'data point count per batch: {dps}',
                         depth, writer, True)

    def __len__(self):
        return len(self.batch_ids)

    def __str__(self):
        s = super().__str__()
        return f'{s}, type: {self.split_type}'


@dataclass
class DatasetResult(ResultsContainer):
    """Contains results for a dataset, such as training, validating and test.

    """
    def __post_init__(self):
        super().__post_init__()
        self._results: List[EpochResult] = []

    def append(self, epoch_result: EpochResult):
        self._assert_finished(False)
        self._results.append(epoch_result)

    @property
    def results(self) -> List[EpochResult]:
        return self._results

    @property
    def contains_results(self):
        return any(map(lambda r: r.contains_results, self.results))

    def end(self):
        super().end()
        if self.contains_results:
            self.start_time = self.results[0].start_time
            self.end_time = self.results[-1].end_time

    def clone(self) -> ResultsContainer:
        cl = cp.copy(self)
        cl._results = []
        for er in self.results:
            cl._results.append(er.clone())
        return cl

    @property
    def losses(self) -> List[float]:
        """Return the loss for each epoch of the run.  If used on a ``EpocResult`` it
        is the Nth iteration.

        """
        return tuple(map(lambda r: r.ave_loss, self.results))

    def _cat_arrs(self, attr: str) -> np.ndarray:
        arrs = tuple(map(lambda r: getattr(r, attr), self.results))
        return np.concatenate(arrs, axis=0)

    def _get_labels(self) -> np.ndarray:
        arrs = tuple(map(lambda r: r.labels, self.results))
        return np.concatenate(arrs, axis=0)

    def _get_predictions(self) -> np.ndarray:
        arrs = tuple(map(lambda r: r.predictions, self.results))
        return np.concatenate(arrs, axis=0)

    def _get_iterations(self) -> int:
        return sum(map(lambda er: len(er.losses), self._results))

    @property
    def convergence(self) -> int:
        """Return the Nth epoch index this result set convergened.  If used on a
        ``EpocResult`` it is the Nth iteration.

        """
        losses = self.losses
        lowest = min(losses)
        return losses.index(lowest)

    @property
    def converged_epoch(self) -> EpochResult:
        """Return the last epoch that arrived at the lowest loss.

        """
        idx = self.convergence
        return self.results[idx]

    def _format_time(self, attr: str):
        if hasattr(self, attr):
            val: datetime = getattr(self, attr)
            if val is not None:
                return val.strftime("%m/%d/%Y %H:%M:%S:%f")

    def _get_dictable_attributes(self) -> Iterable[Tuple[str, str]]:
        return chain.from_iterable(
            (super()._get_dictable_attributes(),
             self._split_str_to_attributes(
                 ('start_time end_time ave_loss min_loss converged_epoch ' +
                  'statistics'))))

    @property
    def statistics(self) -> Dict[str, Any]:
        """Return the statistics of the data set result.

        :return:

            a dictionary with the following:

              * ``n_epochs``: the number of epoch results

              * ``n_epoch_converged``: the 0 based index for which epoch
                converged (lowest validation loss before it went back up)

              * ``n_batches``: the number of batches on which were trained,
                               tested or validated

              * ``ave_data_points``: the average number of data pointes on
                                     which were trained, tested or validated
                                     per batch

              * ``n_total_data_points``: the number of data pointes on which
                                         were trained, tested or validated

        """
        epochs = self.results
        n_data_points = 0
        n_batches = 0
        if len(epochs) > 0:
            epoch: EpochResult = epochs[0]
            n_data_points = epoch.n_data_points
            n_batches = len(epoch.batch_ids)
            for epoch in epochs:
                assert n_data_points == epoch.n_data_points
            n_total_points = sum(n_data_points)
            ave_data_points = n_total_points / len(n_data_points)
        return {'n_epochs': len(epochs),
                'n_epoch_converged': self.converged_epoch.index + 1,
                'n_batches': n_batches,
                'ave_data_points': ave_data_points,
                'n_total_data_points': n_total_points}

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              include_details: bool = False, converged_epoch: bool = True,
              include_metrics: bool = True, include_all_metrics: bool = False):
        """Write the results data.

        :param depth: the number of indentation levels

        :param writer: the data sink

        :param include_settings: whether or not to include model and network
                                 settings in the output

        :param include_config: whether or not to include the configuration in
                               the output

        """
        er: EpochResult = self.converged_epoch
        res = er if converged_epoch else self
        self._write_line(
            f'min/ave/max loss: {res.min_loss:.5f}/{res.ave_loss:.5f}/' +
            f'{er.max_loss:.5f}',
            depth, writer)
        if include_all_metrics:
            self._write_line('classification:', depth, writer)
            res.classification_metrics.write(depth + 1, writer)
            self._write_line('prediction:', depth, writer)
            res.prediction_metrics.write(depth + 1, writer)
        elif include_metrics:
            res.metrics.write(depth, writer)
        if include_details:
            self._write_line('epoch details:', depth, writer)
            self.results[0].write(depth + 1, writer)


@dataclass
class ModelResult(Dictable):
    """A container class used to capture the training, validation and test results.
    The data captured is used to report and plot curves.

    """
    RUNS = 1

    config: Configurable = field()
    """Useful for retrieving hyperparameter settings later after unpersisting
    from disk.

    """

    name: str = field()
    """The name of this result set."""

    model_settings: InitVar[Dict[str, Any]] = field()
    """The setttings used to configure the model."""

    net_settings: InitVar[Dict[str, Any]] = field()
    """The network settings used by the model for this result set."""

    decoded_attributes: Set[str] = field()
    """The attributes that were coded and used in this model."""

    dataset_result: Dict[DatasetSplitType, DatasetResult] = \
        field(default_factory=dict)
    """The dataset (i.e. ``validation``, ``test``) level results."""

    def __post_init__(self, model_settings: ModelSettings,
                      net_settings: NetworkSettings):
        self.RUNS += 1
        self.index = self.RUNS
        splits = tuple(DatasetSplitType)
        self.dataset_result = {k: DatasetResult() for k in splits}
        self.model_settings = model_settings.asdict('class_name')
        self.net_settings = net_settings.asdict('class_name')
        self.net_settings['module_class_name'] = \
            net_settings.get_module_class_name()

    @classmethod
    def reset_runs(self):
        """Reset the run counter.

        """
        self.RUNS = 1

    @classmethod
    def get_num_runs(self):
        return self.RUNS

    def clone(self) -> ModelResult:
        cl = cp.copy(self)
        cl.dataset_result = {}
        for k, v in self.dataset_result.items():
            cl.dataset_result[k] = v.clone()
        return cl

    def get_intermediate(self) -> ModelResult:
        cl = self.clone()
        for ds in cl.dataset_result.values():
            if not ds.is_started:
                ds.start()
            if not ds.is_ended:
                ds.end()
        return cl

    @property
    def train(self) -> DatasetResult:
        """Return the training run results.

        """
        return self.dataset_result[DatasetSplitType.train]

    @property
    def validation(self) -> DatasetResult:
        """Return the validation run results.

        """
        return self.dataset_result[DatasetSplitType.validation]

    @property
    def test(self) -> DatasetResult:
        """Return the testing run results.

        """
        return self.dataset_result[DatasetSplitType.test]

    def reset(self, name: DatasetSplitType):
        """Clear all results for data set ``name``.

        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'restting dataset result \'{name}\'')
        self.dataset_result[name] = DatasetResult()

    @property
    def contains_results(self) -> bool:
        return len(self.test) > 0 or len(self.validation) > 0

    @property
    def non_empty_dataset_result(self) -> Dict[str, DatasetResult]:
        dct = OrderedDict()
        for split_name in 'train validation test'.split():
            ds = getattr(self, split_name)
            if ds.contains_results:
                dct[split_name] = ds
        return dct

    @property
    def last_test_name(self) -> str:
        """Return the anem of the dataset that exists in the container, and thus, the
        last to be populated.  In order, this is test and then validation.

        """
        if self.test.contains_results:
            return DatasetSplitType.test
        if self.validation.contains_results:
            return DatasetSplitType.validation
        raise NoResultError(self.__class__)

    @property
    def last_test(self) -> DatasetResult:
        """Return either the test or validation results depending on what is available.

        """
        return self.dataset_result[self.last_test_name]

    def write_result_statistics(self, split_type: DatasetSplitType,
                                depth: int = 0, writer=sys.stdout):
        ds: DatasetResult = self.dataset_result[split_type]
        stats = ds.statistics
        ave_dps = stats['ave_data_points']
        n_dps = stats['n_total_data_points']
        self._write_line(f"batches: {stats['n_batches']}",
                         depth, writer)
        self._write_line(f"ave data points per batch/total: {ave_dps:.1f}/" +
                         f'{n_dps}', depth, writer)
        if split_type == DatasetSplitType.validation:
            self._write_line('converged/epochs: ' +
                             f"{stats['n_epoch_converged']}/" +
                             f"{stats['n_epochs']}", depth, writer)

    def _get_dictable_attributes(self) -> Iterable[Tuple[str, str]]:
        return chain.from_iterable(
            (self._split_str_to_attributes(
                'name index model_settings net_settings'),
             (('dataset_result', 'non_empty_dataset_result'),)))

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              include_settings: bool = False, include_converged: bool = False,
              include_config: bool = False, include_all_metrics: bool = False):
        """Generate a human readable format of the results.

        """
        lr = self.model_settings["learning_rate"]
        self._write_line(f'Name: {self.name}', depth, writer)
        self._write_line(f'Run index: {self.index}', depth, writer)
        self._write_line(f'Learning rate: {lr}', depth, writer)
        ds_res: DatasetResult
        for split_type, ds_res in self.dataset_result.items():
            self._write_line(f'{split_type.name}:', depth + 1, writer)
            if ds_res.contains_results:
                start_time = ds_res._format_time('start_time')
                end_time = ds_res._format_time('end_time')
                if start_time is not None:
                    self._write_line(f'started: {start_time}',
                                     depth + 2, writer)
                    self._write_line(f'ended: {end_time}',
                                     depth + 2, writer)
                self.write_result_statistics(split_type, depth + 2, writer)
                multi_epic = len(self.dataset_result[split_type].results) > 1
                if include_converged and multi_epic:
                    self._write_line('average over epoch:', depth + 2, writer)
                    ds_res.write(depth + 3, writer, include_details=True,
                                 converged_epoch=False)
                    self._write_line('converged epoch:', depth + 2, writer)
                    ds_res.write(depth + 3, writer, include_details=False,
                                 converged_epoch=True)
                else:
                    all_metrics = (include_all_metrics and
                                   split_type == DatasetSplitType.test)
                    # don't write useless training metrics since training
                    # doesn't produce predictions
                    metrics = (split_type != DatasetSplitType.train)
                    ds_res.write(
                        depth + 2, writer,
                        include_metrics=metrics,
                        include_all_metrics=all_metrics)
            else:
                self._write_line('no results', depth + 2, writer)
        if include_settings:
            if self.decoded_attributes is None:
                dattribs = None
            else:
                dattribs = sorted(self.decoded_attributes)
            self._write_line('settings:', depth, writer)
            self._write_line(f'attributes: {dattribs}', depth + 1, writer)
            self._write_line('model:', depth + 1, writer)
            self._write_dict(self.model_settings, depth + 2, writer)
            self._write_line('network:', depth + 1, writer)
            self._write_dict(self.net_settings, depth + 2, writer)
        if include_config:
            self._write_line('configuration:', depth, writer)
            self.config.write(depth + 1, writer)

    def __str__(self):
        model_name = self.net_settings['module_class_name']
        return f'{model_name} ({self.index})'

    def __repr__(self):
        return self.__str__()
