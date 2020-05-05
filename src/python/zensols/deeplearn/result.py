"""Contains contain classes for results generated from training and testing a
model.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
import logging
import sys
import pickle
from pathlib import Path
from itertools import chain
from typing import Any, List, Dict
import sklearn.metrics as mt
import numpy as np
import torch
import pandas as pd
import math
import matplotlib.pyplot as plt
from zensols.config import Configurable
from zensols.persist import persisted, PersistableContainer
from zensols.deeplearn import (
    Batch,
    NetworkSettings,
    NetworkModelSettings,
)

logger = logging.getLogger(__name__)


class NoResultsException(Exception):
    """Convenience used for helping debug the network.

    """
    def __init__(self):
        super().__init__('no results available')


class ResultsContainer(PersistableContainer, metaclass=ABCMeta):
    """Container class for results while training, testing and validating a model.

    """
    PREDICTIONS_INDEX = 0
    LABELS_INDEX = 1

    @abstractmethod
    def get_ids(self) -> List[Any]:
        "See property ``ids``."
        pass

    @abstractmethod
    def get_outcomes(self) -> np.ndarray:
        "See property ``outcomes``"
        pass

    @abstractmethod
    def get_loss(self) -> float:
        "See property ``loss``."
        pass

    @abstractmethod
    def get_losses(self) -> List[float]:
        "See property ``losses``."
        pass

    @property
    def contains_results(self) -> bool:
        """Return ``True`` if this container has results.

        """
        return len(self) > 0

    def _assert_results(self):
        "Raises an exception if there are no results."
        if not self.contains_results:
            raise NoResultsException()

    def __len__(self):
        return len(self.get_ids())

    def _data_updated(self):
        "Void out all cached data"
        self._get_persistable_metadata().clear()

    @property
    @persisted('_ids', transient=True)
    def ids(self) -> List[Any]:
        """Return the IDs of the data points found in the batch for this result set.

        """
        self._assert_results()
        return self.get_ids()

    @property
    @persisted('_outcomes', transient=True)
    def outcomes(self) -> np.ndarray:
        """Return the outcomes as an array with the first row the provided labels and
        the second row the predictions.  If no labels are given during the
        prediction (i.e. evaluation) there will only be one row, which are the
        predictions.

        """
        self._assert_results()
        return self.get_outcomes()

    @property
    def labels(self) -> np.ndarray:
        """Return the labels or ``None`` if none were provided (i.e. during
        test/evaluation).

        """
        self._assert_results()
        return self.outcomes[self.LABELS_INDEX]

    @property
    def predictions(self) -> np.ndarray:
        """Return the predictions from the model.

        """
        self._assert_results()
        return self.outcomes[self.PREDICTIONS_INDEX]

    @property
    @persisted('_loss', transient=True)
    def loss(self) -> float:
        """Return the average loss of this result set.

        """
        self._assert_results()
        return self.get_loss()

    @property
    @persisted('_losses', transient=True)
    def losses(self) -> List[float]:
        """Return the loss for each epoch of the run.  If used on a ``EpocResult`` it
        is the Nth iteration.

        """
        return self.get_losses()

    @property
    @persisted('_convergence', transient=True)
    def convergence(self):
        """Return the Nth epoch this result set convergened.  If used on a
        ``EpocResult`` it is the Nth iteration.

        """
        losses = self.losses
        lowest = min(losses)
        return losses.index(lowest)

    @staticmethod
    def compute_metrics(average: str, y_true: np.ndarray,
                        y_pred: np.ndarray) -> Dict[str, float]:
        scores = tuple(map(lambda f: f(y_true,  y_pred, average=average),
                           (mt.f1_score, mt.precision_score, mt.recall_score)))
        return {'f1': scores[0],
                'precision': scores[1],
                'recall': scores[2]}

    def _compute_metrics(self, average: str) -> Dict[str, float]:
        """Compute F1, precision and recall.

        :param average: the type of metric to produce (either ``micro`` or
                        ``macro``).

        """
        return self.compute_metrics(average, self.labels, self.predictions)

    @property
    @persisted('_micro_metrics', transient=True)
    def micro_metrics(self) -> Dict[str, float]:
        """Compute F1, precision and recall.

        """
        self._assert_results()
        return self._compute_metrics('micro')

    @property
    @persisted('_macro_metrics', transient=True)
    def macro_metrics(self) -> Dict[str, float]:
        """Compute F1, precision and recall.

        """
        self._assert_results()
        return self._compute_metrics('macro')

    @property
    @persisted('_dataframe', transient=True)
    def dataframe(self) -> pd.DataFrame:
        """Return the results as a pandas dataframe.

        """
        self._assert_results()
        return pd.DataFrame({'id': self.ids,
                             'label': self.labels,
                             'prediction': self.predictions})

    ## TODO: add or replace a min loss to the class and report that instead
    ## since average loss seems less useful
    def write(self, writer=sys.stdout, indent=0):
        """Generate a human readable representation of the results.

        :param writer: the text sink
        :param indent: the indentation space
        """
        sp = ' ' * (indent * 2)
        micro = self.micro_metrics
        macro = self.macro_metrics
        writer.write(f'{sp}loss: {self.loss}\n')
        writer.write(f'{sp}num outcomes: {self.outcomes.shape[1]}\n')
        writer.write(f'{sp}epoch convergence: {self.convergence}\n')
        writer.write(f"{sp}micro: F1: {micro['f1']:.3f}, " +
                     f"precision: {micro['precision']:.2f}, " +
                     f"recall: {micro['recall']:.2f}\n")
        writer.write(f"{sp}macro: F1: {macro['f1']:.3f}, " +
                     f"precision: {macro['precision']:.2f}, " +
                     f"recall: {macro['recall']:.2f}\n")


@dataclass
class EpochResult(ResultsContainer):
    """Contains results recorded from an epoch of a neural network model.  This is
    during a training/validation or test cycle.

    :param loss_updates: the losses generated from each iteration of the epoch
    :param id_updates: the IDs of the data points from each iteration of the
                       epoch
    :param prediction_updates: the predictions generated by the model from each
                               iteration of the epoch

    """
    index: int
    split_type: str
    loss_updates: List[float] = field(default_factory=list)
    id_updates: List[int] = field(default_factory=list)
    prediction_updates: List[np.ndarray] = field(default_factory=list)
    n_data_points: List[int] = field(default_factory=list)

    def update(self, batch: Batch, loss: torch.Tensor, labels: torch.Tensor,
               preds: torch.Tensor, label_shape: List[tuple]):
        logger.debug(f'{self.index}:{self.split_type}: ' +
                     f'update batch: {batch.id}, label_shape: {label_shape}')
        # object function loss; 'mean' is the default 'reduction' parameter for
        # loss functions; we can either muliply it back out or use 'sum' in the
        # criterion initialize
        self.loss_updates.append(loss.item() * len(batch))
        # batches are always the first dimension
        self.n_data_points.append(label_shape[0])
        # get the indexes of the max value across labels and outcomes
        # labels = labels.argmax(1)
        # preds = output.argmax(1)
        # stack and append for metrics computation later
        res = torch.stack((preds, labels), 0)
        self.prediction_updates.append(res.cpu().numpy())
        # keep IDs for tracking
        self.id_updates.append(batch.data_point_ids)
        self._data_updated()

    def get_ids(self):
        return tuple(chain.from_iterable(self.id_updates))

    def get_loss(self):
        return sum(self.loss_updates) / len(self)

    def get_losses(self):
        return self.loss_updates

    def get_outcomes(self):
        return np.concatenate(self.prediction_updates, 1)

    def __getitem__(self, i: int) -> np.ndarray:
        return self.prediction_updates[i]

    def __str__(self):
        return f'{self.index}: loss: {self.loss:.3f}, len: {len(self)}'

    def __repr__(self):
        return self.__str__()


@dataclass
class DatasetResult(ResultsContainer):
    """Contains results from training/validating or test cycle.

    :param results: the results generated from the iterations of the epoch
    """
    results: List[EpochResult] = field(default_factory=list)

    def append(self, epoch_result: EpochResult):
        self.results.append(epoch_result)
        self._data_updated()

    def get_ids(self):
        ids = chain.from_iterable(map(lambda r: r.get_ids(), self.results))
        return tuple(ids)

    def get_loss(self):
        loss_sum = sum(map(lambda r: r.loss, self.results))
        batch_sum = sum(map(lambda r: len(r), self.results))
        return 0 if batch_sum == 0 else loss_sum / batch_sum

    def get_losses(self):
        return tuple(map(lambda r: r.loss, self.results))

    def get_outcomes(self):
        prs = tuple(map(lambda r: r.outcomes, self.results))
        return np.concatenate(prs, axis=1)

    def __getitem__(self, i: int) -> EpochResult:
        return self.results[i]


@dataclass
class NetworkModelResult(ResultsContainer):
    """A container class used to capture the training, validation and test results.
    The data captured is used to report and plot curves.

    :param cl_settings: the setttings used to configure the model
    :param net_settings: the settings used to configure the network
    :param config: optional configuration, which might be useful for
                   retrieving hyperparameter settings later after
                   unpersisting from disk

    """
    config: Configurable
    cl_settings: NetworkModelSettings
    net_settings: NetworkSettings

    def __post_init__(self):
        super().__post_init__()
        global _runs
        if '_runs' not in globals():
            _runs = 0
        _runs += 1
        self.index = _runs
        splits = 'train validation test'.split()
        self.epochs = {k: DatasetResult() for k in splits}

    @staticmethod
    def reset_runs():
        """Reset the run counter.

        """
        global _runs
        _runs = 0

    def __getitem__(self, name: str) -> DatasetResult:
        return self.epochs[name]

    @property
    def train(self) -> DatasetResult:
        """Return the training run results.

        """
        self._data_updated()
        return self.epochs['train']

    @property
    def validation(self) -> DatasetResult:
        """Return the validation run results.

        """
        self._data_updated()
        return self.epochs['validation']

    @property
    def test(self) -> DatasetResult:
        """Return the testing run results.

        """
        self._data_updated()
        return self.epochs['test']

    @property
    def contains_results(self) -> bool:
        return len(self.test) > 0 or len(self.validation) > 0

    @property
    def last_test_epoch_name(self) -> str:
        if len(self.test) > 0:
            return 'test'
        if len(self.validation) > 0:
            return 'validation'
        raise NoResultsException()

    @property
    def last_test_epoch(self) -> DatasetResult:
        """Return either the test or validation results depending on what is available.

        """
        return self[self.last_test_epoch_name]

    def get_ids(self):
        return self.last_test_epoch.get_ids()

    def get_outcomes(self):
        return self.last_test_epoch.get_outcomes()

    def get_loss(self):
        return self.last_test_epoch.get_loss()

    def get_losses(self) -> List[float]:
        return self.last_test_epoch.get_losses()

    @property
    def results_path(self) -> Path:
        """The model file used to persist for this configuration.

        """
        path_fmt = self.cl_settings.results_path_format
        return Path(path_fmt.format(**{'r': self}))

    def save(self) -> Path:
        """Save the results recorded during the training and testing of the model.

        """
        path = self.results_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f'saved results: {path}')
        return path

    @staticmethod
    def load_from(path: Path) -> Any:
        """Load the results from a path.

        """
        with open(path, 'rb') as f:
            return pickle.load(f)

    def load(self) -> Any:
        """Load the model using the path given in the configuration.

        """
        return self.load_from(self.results_path)

    def get_stats(self, result_name: str):
        self._data_updated()
        epocs = self.epochs[result_name].results
        fn = 0
        if len(epocs) > 0:
            fn = epocs[0].n_data_points
            for epoc in epocs:
                assert fn == epoc.n_data_points
        return {'n_epocs': len(epocs),
                'n_data_points': fn}

    def write_stats(self, result_name: str, writer=sys.stdout, indent=0):
        sp = ' ' * (indent * 2)
        stats = self.get_stats(result_name)
        epocs = stats['n_epocs']
        fn = sum(stats['n_data_points'])
        writer.write(f'{sp}num epocs: {epocs}\n')
        writer.write(f'{sp}num data points per epoc: {fn}\n')

    def write(self, writer=sys.stdout, indent=0):
        """Generate a human readable format of the results.

        """
        sp = ' ' * (indent * 2)
        writer.write(f'{sp}{self}\n')
        sp = ' ' * ((indent + 1) * 2)
        for name, es in self.epochs.items():
            writer.write(f'{sp}{name}:\n')
            if es.contains_results:
                es.write(writer, indent + 2)
            else:
                spe = ' ' * ((indent + 2) * 2)
                writer.write(f'{spe}no results\n')

    def __str__(self):
        return (f'{self.cl_settings.model_type} ({self.index}): ' +
                f'learning_rate: {self.cl_settings.learning_rate}')

    def __repr__(self):
        return self.__str__()


class ResultGrapher(object):
    """Graphs the an instance of ``NetworkModelResult``.  This creates subfigures,
    one for each of the results given as input to ``plot``.

    :see: plot

    """
    def __init__(self, name: str, figsize: List[int] = (15, 10),
                 split_types: List[str] = None, title: str = None):
        """Initialize the grapher.

        :param name: the name that goes in the title of the graph
        :param figsize: the size of the top level figure (not the panes)
        :param split_types: the splits to graph (list of size 2); defaults to
                            ['train', 'validation']
        :param title: the title format used to create each sub pane graph.

        """
        self.name = name
        self.figsize = figsize
        if split_types is None:
            self.split_types = 'train validation'.split()
        else:
            self.split_types = self.split_types
        if title is None:
            self.title = ('Figure {r.index} ' +
                          '(lr={r.cl_settings.learning_rate:.5f}, ' +
                          'F1={r.micro_metrics[f1]:.3f})')
        else:
            self.title = title

    def _render_title(self, cont: ResultsContainer) -> str:
        return self.title.format(**{'r': cont})

    def plot(self, containers: List[NetworkModelResult]):
        ncols = min(2, len(containers))
        nrows = math.ceil(len(containers) / ncols)
        logger.debug(f'plot grid: {nrows} X {ncols}')
        fig, axs = plt.subplots(
            ncols=ncols, nrows=nrows, sharex=True, figsize=self.figsize)
        logger.debug(f'ax type: {type(axs)}')
        if not isinstance(axs, np.ndarray):
            logger.debug(f'adding dim')
            axs = np.array([[axs]])
        if axs.shape == (ncols,):
            axs = np.expand_dims(axs, axis=0)
        logger.debug(f'ax shape: {axs.shape}')
        fig.suptitle(f'Training and Validation Learning Rates: {self.name}')
        handles = []
        row = 0
        col = 0
        for i, cont in enumerate(containers):
            logger.debug(f'plotting {cont}')
            es = tuple(map(lambda n: (n.capitalize(), cont.epochs[n]),
                           self.split_types))
            x = range(len(es[0][1].losses))
            ax = axs[row][col]
            ax.plot(x, es[0][1].losses, color='r', label=es[0][0])
            ax.plot(x, es[1][1].losses, color='b', label=es[1][0])
            ax.set_title(self._render_title(cont))
            handles.append(ax)
            ax.set(xlabel='Epochs', ylabel='Loss')
            col += 1
            if col == ncols:
                col = 0
                row += 1
        plt.legend(tuple(map(lambda e: e[0], es)))
        #plt.show()
