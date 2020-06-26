"""Client entry point to the model.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field
from typing import Tuple, Any
import sys
import logging
import pandas as pd
from io import TextIOWrapper
from pathlib import Path
from zensols.util import time
from zensols.config import (
    Configurable,
    Writable,
    ImportConfigFactory,
)
from zensols.persist import (
    persisted,
    PersistableContainer,
    PersistedWork,
    Deallocatable,
)
from zensols.dataset import DatasetSplitStash
from zensols.deeplearn import NetworkSettings, ModelSettings
from zensols.deeplearn.vectorize import (
    SparseTensorFeatureContext,
    FeatureVectorizerManagerSet,
)
from zensols.deeplearn.batch import BatchStash, BatchMetadata
from zensols.deeplearn.result import (
    ModelResult,
    ModelResultGrapher,
    ModelResultManager,
)
from . import ModelManager, ModelExecutor

logger = logging.getLogger(__name__)


@dataclass
class ModelFacade(PersistableContainer, Writable):
    """Provides easy to use client entry points to the model executor, which
    trains, validates, tests, saves and loads the model.

    More common attributes, such as the learning rate and number of epochs, are
    properties that dispatch to :py:attrib:~`executor`--for the others, go
    directly to the property.

    :param factory: the factory used to create the executor

    :param progress_bar: create text/ASCII based progress bar if ``True``

    :param progress_bar_cols: the number of console columns to use for the
                              text/ASCII based progress bar

    :param executor_name: the configuration entry name for the executor, which
                          defaults to ``executor``

    :param cache_level: determines how much and when to deallcate (see
                        :class:`.ModelFacadeCacheLevel`)

    :see zensols.deeplearn.domain.ModelSettings:

    """
    config: Configurable
    progress_bar: bool = field(default=True)
    progress_bar_cols: int = field(default=79)
    executor_name: str = field(default='executor')
    cache_batches: bool = field(default=True)
    save_train_result: bool = field(default=False)
    save_test_result: bool = field(default=True)
    save_plot_result: bool = field(default=True)
    writer: TextIOWrapper = field(default=sys.stdout)

    def __post_init__(self):
        super().__init__()
        self._config_factory = PersistedWork(
            '_config_factory', self, cache_global=True)
        self._executor = PersistedWork('_executor', self, cache_global=True)
        self.debuged = False
        self.last_result = None

    def _create_executor(self) -> ModelExecutor:
        """Create a new instance of an executor.  Used by :py:attrib:~`executor`.

        """
        logger.info('creating new executor')
        executor = self.config_factory(
            self.executor_name,
            progress_bar=self.progress_bar,
            progress_bar_cols=self.progress_bar_cols)
        executor.model_settings.cache_batches = self.cache_batches
        return executor

    @property
    @persisted('_config_factory')
    def config_factory(self):
        """The configuration factory used to create facades.

        """
        return ImportConfigFactory(self.config)

    @property
    @persisted('_executor')
    def executor(self) -> ModelExecutor:
        """A cached instance of the executor tied to the instance of this class.

        """
        return self._create_executor()

    @property
    def net_settings(self) -> NetworkSettings:
        return self.executor.net_settings

    @property
    def model_settings(self) -> ModelSettings:
        return self.executor.model_settings

    @property
    def batch_stash(self) -> BatchStash:
        """The stash used to encode and decode batches by the executor.

        """
        dss: DatasetSplitStash = self.executor.dataset_stash
        return dss.delegate

    @property
    def vectorizer_manager_set(self) -> FeatureVectorizerManagerSet:
        """Return the vectorizer manager set used for the facade.  This is taken from
        the executor's batch stash.

        """
        return self.executor.batch_stash.vectorizer_manager_set

    @property
    def batch_metadata(self) -> BatchMetadata:
        """Return the batch metadata used on the executor.  This will only work if
        there is an attribute set called ``batch_metadata_factory`` set on
        :py:attrib:~`executor.net_settings` (i.e. ``EmbeddingNetworkSettings``
        in the ``zensols.deepnlp`` package).

        :see: :class:`zensols.deepnlp.model.module.EmbeddingNetworkSettings`

        """
        return self.executor.net_settings.batch_metadata_factory()

    @property
    def label_attribute_name(self):
        """Get the label attribute name.

        """
        return self.batch_metadata.mapping.label_attribute_name

    @property
    def dropout(self) -> float:
        """The dropout for the entire network.

        """
        return self.executor.dropout

    @dropout.setter
    def dropout(self, dropout: float):
        """The dropout for the entire network.

        """
        self.executor.net_settings.dropout = dropout

    @property
    def epochs(self) -> int:
        """The number of epochs for training and validation.

        """
        return self.executor.epochs

    @epochs.setter
    def epochs(self, n_epochs: int):
        """The number of epochs for training and validation.

        """
        self.executor.model_settings.epochs = n_epochs

    @property
    def learning_rate(self) -> float:
        """The learning rate to set on the optimizer.

        """
        return self.executor.learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate: float):
        """The learning rate to set on the optimizer.

        """
        self.executor.model_settings.learning_rate = learning_rate

    def clear(self):
        """Clear out any cached executor.

        """
        executor = self.executor
        config_factory = self.config_factory
        executor.deallocate()
        config_factory.deallocate()
        self._executor.clear()
        self._config_factory.clear()

    def reload(self):
        """Clears all state and reloads the configuration.

        """
        self.clear()
        self.config.reload()

    @classmethod
    def load_from_path(cls, path: Path, *args, **kwargs):
        """Construct a new facade from the data saved in a persisted model file.  This
        uses the :py:meth:`.ModelManager.load_from_path` to reconstruct the
        returned facade, which means some attributes are taken from default if
        not taken from ``*args`` or ``**kwargs``.

        Arguments:
           Passed through to the initializer of invoking class ``cls``.

        :see: :py:meth:`.ModelManager.load_from_path`

        """
        logger.info(f'loading from facade from {path}')
        mm = ModelManager.load_from_path(path)
        if 'executor_name' not in kwargs:
            kwargs['executor_name'] = mm.model_executor_name
        executor = mm.load_executor()
        mm.config_factory.deallocate()
        facade = cls(executor.config, *args, **kwargs)
        executor.model_settings.cache_batches = facade.cache_batches
        facade._config_factory.set(executor.config_factory)
        facade._executor.set(executor)
        return facade

    def debug(self, debug_value: Any = True):
        """Debug the model by setting the configuration to debug mode and invoking a
        single forward pass.  Logging must be configured properly to get the
        output, which is typically just invoking
        :py:meth:`logging.basicConfig`.

        """
        self.reload()
        executor = self.executor
        self._configure_debug_logging()
        executor.debug = debug_value
        executor.progress_bar = False
        executor.model_settings.batch_limit = 1
        self.debuged = True
        executor.train()

    def save_last_result(self):
        executor = self.executor
        if executor.result_manager is not None:
            executor.result_manager.dump(self.last_result)
            if self.save_plot_result:
                self.plot_last_result(save=True)

    def train(self, description: str = None) -> ModelResult:
        """Train and test or just debug the model depending on the configuration.

        :param description: a description used in the results, which is useful
                            when making incremental hyperparameter changes to
                            the model

        """
        executor = self.executor
        executor.reset()
        if self.writer is not None:
            executor.write(writer=self.writer)
        logger.info('training...')
        with time('trained'):
            res = executor.train(description)
        self.last_result = res
        if self.save_train_result:
            self.save_last_result()
        return res

    def test(self, description: str = None) -> ModelResult:
        """Load the model from disk and test it.

        """
        if self.debuged:
            raise ValueError('testing is not allowed in debug mode')
        executor = self.executor
        executor.load()
        logger.info('testing...')
        with time('trained'):
            res = executor.test(description)
        self.last_result = res
        if self.save_test_result:
            self.save_last_result()
        if self.writer is not None:
            res.write(writer=self.writer)
        return res

    def get_grapher(self, figsize: Tuple[int, int] = (15, 5),
                    title: str = None) -> ModelResultGrapher:
        """Return an instance of a model grapher.  This class can plot results of
        ``res`` using ``matplotlib``.

        :see: :class:`.ModelResultGrapher`

        """
        result_manager: ModelResultManager = self.executor.result_manager
        if title is None:
            title = self.executor.model_name
        result_manager = self.executor.result_manager
        path_dir = result_manager.get_next_text_path().parent
        path = path_dir / f'graph-{result_manager.get_last_key(False)}.png'
        return ModelResultGrapher(title, figsize, save_path=path)

    def plot_last_result(self, save: bool = False, show: bool = False):
        if self.last_result is None:
            raise ValueError('no result to plot; invoke train() and or test()')
        grapher = self.get_grapher()
        grapher.plot([self.last_result])
        if save:
            grapher.save()
        if show:
            grapher.show()

    def write_results(self, depth: int = 0, writer: TextIOWrapper = sys.stdout,
                      verbose: bool = False):
        """Load the last set of results from the file system and print them out.

        """
        logging.getLogger('zensols.deeplearn.result').setLevel(logging.INFO)
        logger.info('load previous results')
        rm = self.executor.result_manager
        if rm is None:
            rm = ValueError('no result manager available')
        res = rm.load()
        if res is None:
            raise ValueError('no results found')
        res.write(depth, writer, include_settings=verbose,
                  include_converged=verbose, include_config=verbose)

    def get_predictions(self, *args, **kwargs) -> pd.DataFrame:
        """Return the predictions made during the test phase of the model execution.
        The arguments are passed to :meth:`ModelExecutor.get_predictions`.

        :see: :meth:`.ModelExecutor.get_predictions`

        """
        executor = self.executor
        executor.load()
        return executor.get_predictions(*args, **kwargs)

    def write_predictions(self, lines: int = 10):
        """Print the predictions made during the test phase of the model execution.

        :param lines: the number of lines of the predictions data frame to be
                      printed

        :param writer: the data sink

        """
        preds = self.get_predictions()
        print(preds.head(lines), file=self.writer)

    def write(self, depth: int = 0, writer: TextIOWrapper = None,
              include_metadata: bool = True, include_config: bool = False):
        writer = self.writer if writer is None else writer
        writer = sys.stdout if writer is None else writer
        bmeta = None
        try:
            bmeta = self.batch_metadata
        except AttributeError:
            pass
        self._write_line(f'{self.executor.name}:', depth, writer)
        self.executor.write(depth + 1, writer)
        if bmeta is not None and include_metadata:
            self._write_line('metadata:', depth, writer)
            bmeta.write(depth + 1, writer)
        if include_config:
            self._write_line('config:', depth, writer)
            self.config.write(depth + 1, writer)

    def _deallocate_config_instance(self, inst: Any):
        if isinstance(self.config_factory, ImportConfigFactory):
            inst = self.config_factory.clear_instance(inst)
        dealloc = isinstance(inst, Deallocatable)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'deallocate {inst}: {type(inst)}: {dealloc}')
        if dealloc:
            inst.deallocate()

    def _configure_debug_logging(self):
        """When debuging the model, configure the logging system for output.  The
        correct loggers need to be set to debug mode to print the model
        debugging information such as matrix shapes.

        """
        for name in ['zensols.deeplearn.vectorize.vectorizers',
                     'zensols.deeplearn.model.executor',
                     __name__]:
            logging.getLogger(name).setLevel(logging.DEBUG)

    @staticmethod
    def get_encode_sparse_matrices() -> bool:
        """Return whether or not sparse matricies are encoded.

        :see: :meth:`set_sparse`

        """
        return SparseTensorFeatureContext.USE_SPARSE

    @staticmethod
    def set_encode_sparse_matrices(use_sparse: bool = False):
        """If called before batches are created, encode all tensors the would be
        encoded as dense rather than sparse when ``use_sparse`` is ``False``.
        Oherwise, tensors will be encoded as sparse where it makes sense on a
        per vectorizer basis.

        """
        SparseTensorFeatureContext.USE_SPARSE = use_sparse
