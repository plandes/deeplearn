"""Contains contain classes for results generated from training and testing a
model.

"""
__author__ = 'Paul Landes'

from typing import List, Dict, Set, Iterable, Any, Type, Tuple
from dataclasses import dataclass, field, InitVar
from enum import Enum
from abc import ABCMeta, abstractmethod
import logging
import sys
from collections import OrderedDict
from itertools import chain
from datetime import datetime
from io import TextIOBase
import math
import sklearn.metrics as mt
import numpy as np
import torch
from zensols.config import Configurable, Dictable
from zensols.deeplearn import ModelSettings, NetworkSettings
from zensols.deeplearn.batch import Batch

logger = logging.getLogger(__name__)


class NoResultsException(Exception):
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


@dataclass
class PredictionMetrics(Metrics):
    """Real valued prediction results for :obj:`.ModelType.PREDICTION` result.

    """
    @property
    def root_mean_squared_error(self) -> float:
        """Return the root mean squared error metric.

        """
        mse = mt.mean_squared_error(self.labels, self.predictions)
        return math.sqrt(mse)

    @property
    def mean_absolute_error(self) -> float:
        """Return the mean absolute error metric.

        """
        return mt.mean_absolute_error(self.labels, self.predictions)

    @property
    def r2_score(self) -> float:
        """Return the R^2 score metric.

        """
        return mt.r2_score(self.labels, self.predictions)

    @property
    def correlation(self) -> float:
        """Return the correlation metric.

        """
        return np.corrcoef(self.labels, self.predictions)[0][1]

    def _get_dictable_attributes(self) -> Iterable[Tuple[str, str]]:
        return (('rmse', 'root_mean_squared_error'),
                ('mae', 'mean_absolute_error'),
                ('r2', 'r2_score'),
                ('correlation', 'correlation'))

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(f'RMSE: {self.root_mean_squared_error:.3f}', depth, writer)
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
    average: str

    @property
    def f1(self) -> float:
        """Return the F1 metric as either the micro or macro based on the
        :obj:`average` attribute.

        """
        return mt.f1_score(
            self.labels, self.predictions, average=self.average)

    @property
    def precision(self) -> float:
        """Return the precision metric as either the micro or macro based on the
        :obj:`average` attribute.

        """
        return mt.precision_score(
            self.labels, self.predictions, average=self.average)

    @property
    def recall(self) -> float:
        """Return the recall metric as either the micro or macro based on the
        :obj:`average` attribute.

        """
        return mt.recall_score(
            self.labels, self.predictions, average=self.average)

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

    :param n_outcomes: the number of outcomes given for this metrics set

    """
    n_outcomes: int

    @property
    def accuracy(self) -> float:
        """Return the accuracy metric (num correct / total).

        """
        return mt.accuracy_score(self.labels, self.predictions)

    @property
    def n_correct(self) -> int:
        is_eq = np.equal(self.labels, self.predictions)
        return np.count_nonzero(is_eq == True)

    @property
    def micro(self) -> ScoreMetrics:
        """Compute F1, precision and recall.

        """
        return ScoreMetrics(self.labels, self.predictions, 'micro')

    @property
    def macro(self) -> Dict[str, float]:
        """Compute F1, precision and recall.

        """
        return ScoreMetrics(self.labels, self.predictions, 'macro')

    def _get_dictable_attributes(self) -> Iterable[Tuple[str, str]]:
        return self._split_str_to_attributes(
            'accuracy n_correct micro macro')

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(f'accuracy: {self.accuracy:.3f} ' +
                         f'({self.n_correct}/{self.n_outcomes})',
                         depth, writer)
        self.micro.write(depth, writer)
        self.macro.write(depth, writer)

    def __str__(self):
        return str(self.micro)


@dataclass
class ResultsContainer(Dictable, metaclass=ABCMeta):
    """The base class for all metrics containers.  It helps in calculating loss,
    finding labels, predictions and other utility helpers.

    """
    PREDICTIONS_INDEX = 0
    LABELS_INDEX = 1
    FLOAT_TYPES = [np.float32, np.float64, np.float]

    def __post_init__(self):
        super().__init__()

    def _clear(self):
        for attr in '_labels _preds'.split():
            if hasattr(self, attr):
                delattr(self, attr)

    @property
    def contains_results(self):
        """Return ``True`` if this container has results.

        """
        return len(self) > 0

    def _assert_results(self):
        "Raises an exception if there are no results."
        if not self.contains_results:
            raise NoResultsException(self.__class__)

    @property
    def min_loss(self) -> float:
        """Return the lowest loss recorded in this container.

        """
        self._assert_results()
        return min(self.losses)

    @abstractmethod
    def get_outcomes(self) -> np.ndarray:
        """Return the outcomes as an array with the first row the provided labels and
        the second row the predictions.  If no labels are given during the
        prediction (i.e. evaluation) there will only be one row, which are the
        predictions.

        """
        pass

    @property
    def n_outcomes(self) -> int:
        """Return the number of outcomes.

        """
        return self.get_outcomes().shape[1]

    @property
    def model_type(self) -> ModelType:
        """Return the type of the model based on what whether the outcome data is a
        float or integer.

        """
        oc = self.get_outcomes()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'outcomes type: {oc.dtype}')
        if oc.dtype in self.FLOAT_TYPES:
            return ModelType.PREDICTION
        else:
            return ModelType.CLASSIFICTION

    @property
    def labels(self) -> np.ndarray:
        """Return the labels or ``None`` if none were provided (i.e. during
        test/evaluation).

        """
        if not hasattr(self, '_labels'):
            self._assert_results()
            arr = self.get_outcomes()[self.LABELS_INDEX]
            # flatten for multiclass-multioutput
            if arr.shape[-1] > 1:
                self._labels = arr.flatten()
            else:
                self._labels = np.array([])
        return self._labels

    @property
    def predictions(self) -> np.ndarray:
        """Return the predictions from the model.

        """
        if not hasattr(self, '_preds'):
            self._assert_results()
            arr = self.get_outcomes()[self.PREDICTIONS_INDEX]
            # flatten for multiclass-multioutput
            if arr.shape[-1] > 1:
                self._preds = arr.flatten()
            else:
                self._preds = np.array([])
        return self._preds

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
            raise ValueError(f'unknown or unsupported tupe: {mtype}')
        return metrics

    def _get_dictable_attributes(self) -> Iterable[Tuple[str, str]]:
        return self._split_str_to_attributes('n_outcomes metrics')


@dataclass
class EpochResult(ResultsContainer):
    """Contains results recorded from an epoch of a neural network model.  This is
    during a training/validation or test cycle.

    :param index: the Nth epoch of the run (across training, validation, test)

    :param split_type: the name of the split type (i.e. ``train`` vs ``test``)

    :param batch_losses: the losses generated from each iteration of the epoch

    :param batch_ids: the ID of the batch from each iteration of the epoch

    :param prediction_updates: the predictions generated by the model from each
                               iteration of the epoch

    :param n_data_points: the number of data points for each batch for the
                          epoch

    """
    index: int
    split_type: str
    batch_losses: List[float] = field(default_factory=list)
    prediction_updates: List[torch.Tensor] = field(default_factory=list)
    batch_ids: List[int] = field(default_factory=list)
    n_data_points: List[int] = field(default_factory=list)

    def update(self, batch: Batch, loss: torch.Tensor, labels: torch.Tensor,
               preds: torch.Tensor, label_shape: List[tuple]):
        logger.debug(f'{self.index}:{self.split_type}: ' +
                     f'update batch: {batch.id}, label_shape: {label_shape}')
        # object function loss; 'mean' is the default 'reduction' parameter for
        # loss functions; we can either muliply it back out or use 'sum' in the
        # criterion initialize
        self.batch_losses.append(loss.item() * float(batch.size()))
        # batches are always the first dimension
        self.n_data_points.append(label_shape[0])
        # stack and append for metrics computation later
        # print('-' * 30)
        # print('p', type(preds), preds)
        # print('l', type(labels), labels)
        if preds is not None:
            res = torch.stack((preds, labels), 0)
            self.prediction_updates.append(res.clone().detach().cpu())
        self.batch_ids.append(batch.id)
        self._clear()

    def get_outcomes(self) -> np.ndarray:
        self._assert_results()
        if len(self.prediction_updates) > 0:
            return np.concatenate(self.prediction_updates, 1)
        else:
            return torch.tensor([[], []], dtype=torch.int64)

    @property
    def ave_loss(self) -> float:
        """Return the average loss of this result set.

        """
        self._assert_results()
        return sum(self.batch_losses) / len(self.batch_losses)

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

    def __getitem__(self, i: int) -> np.ndarray:
        return self.prediction_updates[i]

    def __len__(self):
        return len(self.batch_ids)

    def __str__(self):
        return f'{self.index}: ave loss: {self.ave_loss:.3f}, len: {len(self)}'

    def __repr__(self):
        return self.__str__()


@dataclass
class DatasetResult(ResultsContainer):
    """Contains results from training/validating or test cycle.

    :param results: the results generated from the iterations of the epoch

    :param start_time: the time when the dataset started traing, testing etc.

    :param start_time: the time when the dataset finished traing, testing etc.


    """
    results: List[EpochResult] = field(default_factory=list)
    start_time: datetime = field(default=None)
    end_time: datetime = field(default=None)

    def append(self, epoch_result: EpochResult):
        self.results.append(epoch_result)
        self._clear()

    @property
    def contains_results(self):
        return any(map(lambda r: r.contains_results, self.results))

    def start(self):
        if self.contains_results:
            raise ValueError(f'container {self} already contains results')
        self.start_time = datetime.now()

    def end(self):
        self.end_time = datetime.now()

    @property
    def losses(self) -> List[float]:
        """Return the loss for each epoch of the run.  If used on a ``EpocResult`` it
        is the Nth iteration.

        """
        return tuple(map(lambda r: r.ave_loss, self.results))

    @property
    def ave_loss(self) -> float:
        loss_sum = sum(self.losses)
        batch_sum = sum(map(lambda r: len(r), self.results))
        return 0 if batch_sum == 0 else loss_sum / batch_sum

    def get_outcomes(self):
        if len(self.results) == 0:
            return np.ndarray((2, 0))
        else:
            prs = tuple(map(lambda r: r.get_outcomes(), self.results))
            return np.concatenate(prs, axis=1)

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

              * ``n_data_points``: the number of data pointes on which were
                                   trained, tested or validated

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
            n_data_points = sum(n_data_points) / len(n_data_points)
        return {'n_epochs': len(epochs),
                'n_epoch_converged': self.converged_epoch.index + 1,
                'n_batches': n_batches,
                'n_data_points': n_data_points}

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              include_details: bool = False, converged_epoch: bool = True,
              include_all_metrics: bool = False):
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
        self._write_line(f'ave/min loss: {res.ave_loss:.5f}/{er.min_loss:.5f}',
                         depth, writer)
        if include_all_metrics:
            self._write_line('classification:', depth, writer)
            res.classification_metrics.write(depth + 1, writer)
            self._write_line('prediction:', depth, writer)
            res.prediction_metrics.write(depth + 1, writer)
        else:
            res.metrics.write(depth, writer)
        if include_details:
            self._write_line('epoch details:', depth, writer)
            self.results[0].write(depth + 1, writer)

    def __getitem__(self, i: int) -> EpochResult:
        return self.results[i]


@dataclass
class ModelResult(Dictable):
    """A container class used to capture the training, validation and test results.
    The data captured is used to report and plot curves.

    :param config: useful for retrieving hyperparameter settings later after
                   unpersisting from disk

    :param model_settings: the setttings used to configure the model

    """
    TRAIN_DS_NAME = 'train'
    VALIDATION_DS_NAME = 'validation'
    TEST_DS_NAME = 'test'
    RUNS = 1

    config: Configurable
    name: str
    model_settings: InitVar[Dict[str, Any]]
    net_settings: InitVar[Dict[str, Any]]
    decoded_attributes: Set[str]
    dataset_result: Dict[str, DatasetResult] = field(default_factory=dict)

    def __post_init__(self, model_settings: ModelSettings,
                      net_settings: NetworkSettings):
        self.RUNS += 1
        self.index = self.RUNS
        splits = 'train validation test'.split()
        self.dataset_result = {k: DatasetResult() for k in splits}
        self.model_settings = model_settings.asdict('class_name')
        self.net_settings = net_settings.asdict('class_name')
        self.net_settings['module_class_name'] = net_settings.get_module_class_name()

    @classmethod
    def reset_runs(self):
        """Reset the run counter.

        """
        self.RUNS = 1

    @classmethod
    def get_num_runs(self):
        return self.RUNS

    def __getitem__(self, name: str) -> DatasetResult:
        return self.dataset_result[name]

    @property
    def train(self) -> DatasetResult:
        """Return the training run results.

        """
        return self.dataset_result[self.TRAIN_DS_NAME]

    @property
    def validation(self) -> DatasetResult:
        """Return the validation run results.

        """
        return self.dataset_result[self.VALIDATION_DS_NAME]

    @property
    def test(self) -> DatasetResult:
        """Return the testing run results.

        """
        return self.dataset_result[self.TEST_DS_NAME]

    def reset(self, name: str):
        """Clear all results for data set ``name``.

        """
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
        if self.test.contains_results:
            return self.TEST_DS_NAME
        if self.validation.contains_results:
            return self.VALIDATION_DS_NAME
        raise NoResultsException(self.__class__)

    @property
    def last_test(self) -> DatasetResult:
        """Return either the test or validation results depending on what is available.

        """
        return self[self.last_test_name]

    def write_result_statistics(self, result_name: str, depth: int = 0,
                                writer=sys.stdout):
        ds: DatasetResult = self.dataset_result[result_name]
        stats = ds.statistics
        ave_dps = stats['n_data_points']
        self._write_line(f"batches: {stats['n_batches']}",
                         depth, writer)
        self._write_line(f"ave data points per batch: {ave_dps:.1f}",
                         depth, writer)
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
        self._write_line(f'Name: {self.name}', depth, writer)
        self._write_line(f'Run index: {self.index}', depth, writer)
        self._write_line(f'Learning rate: {self.model_settings["learning_rate"]}',
                         depth, writer)
        ds_res: DatasetResult
        for name, ds_res in self.dataset_result.items():
            self._write_line(f'{name}:', depth + 1, writer)
            if ds_res.contains_results:
                start_time = ds_res._format_time('start_time')
                end_time = ds_res._format_time('end_time')
                if start_time is not None:
                    self._write_line(f'started: {start_time}',
                                     depth + 2, writer)
                    self._write_line(f'ended: {end_time}',
                                     depth + 2, writer)
                self.write_result_statistics(name, depth + 2, writer)
                multi_epic = len(self.dataset_result[name].results) > 1
                if include_converged and multi_epic:
                    self._write_line('average over epoch:', depth + 2, writer)
                    ds_res.write(depth + 3, writer, include_details=True,
                                 converged_epoch=False)
                    self._write_line('converged epoch:', depth + 2, writer)
                    ds_res.write(depth + 3, writer, include_details=False,
                                 converged_epoch=True)
                else:
                    all_metrics = (include_all_metrics and name == 'test')
                    ds_res.write(
                        depth + 2, writer, include_all_metrics=all_metrics)
            else:
                self._write_line(f'no results', depth + 2, writer)
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
