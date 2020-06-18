"""Client entry point to the model.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field, InitVar
from enum import IntEnum
from typing import Tuple
import sys
import logging
import pandas as pd
from io import TextIOWrapper
from pathlib import Path
from zensols.config import Configurable, ConfigFactory, Writable
from zensols.persist import persisted, Deallocatable, PersistedWork
from zensols.util import time
from zensols.deeplearn.vectorize import (
    SparseTensorFeatureContext,
    FeatureVectorizerManagerSet,
)
from zensols.deeplearn.batch import BatchMetadata
from zensols.deeplearn.result import ModelResult, ModelResultGrapher
from . import ModelManager, ModelExecutor

logger = logging.getLogger(__name__)


class ModelFacadeCacheLevel(IntEnum):
    """Indicates generally how much to cache in a :class:`.ModelFacade` instance.
    Specifically it determines what is deallocated and when.  Note that the
    executor is always cached per :class:`.ModelFacade` instance regardless.

    Levels:
      * NONE: cache nothing and deallocate the ``cache_factory``
      * LOW: cache nothing, but do not deallocate the ``cache_factory``
      * EXECUTOR: globally cache the executor, but not batches
      * BATCHES: cache everything, including the executor globally, batches

    :see: :py:attib:~`.ModelFacade.cache_level`

    """
    NONE = 0
    LOW = 1
    EXECUTOR = 2
    BATCHES = 3


@dataclass
class ModelFacade(Deallocatable, Writable):
    """Provides easy to use client entry points to the model executor, which
    trains, validates, tests, saves and loads the model.

    :param factory: the factory used to create the executor

    :param progress_bar: create text/ASCII based progress bar if ``True``

    :param progress_bar_cols: the number of console columns to use for the
                              text/ASCII based progress bar

    :param executor_name: the configuration entry name for the executor, which
                          defaults to ``executor``

    :param cache_level: determines how much and when to deallcate (see
                        :class:`.ModelFacadeCacheLevel`)

    :param load_type: how to load the model, which is one of
                      * ``none``: reuse whatever model was just trained
                      * ``model``: only load the model state
                      * ``executor``: reload the entire executor via the
                        :class:`.ModelManager`

    :see zensols.deeplearn.domain.ModelSettings:

    """
    config_factory: ConfigFactory
    progress_bar: bool = field(default=True)
    progress_bar_cols: int = field(default=79)
    executor_name: str = field(default='executor')
    load_type: str = field(default='model')
    writer: TextIOWrapper = field(default=sys.stdout)
    cache_level: ModelFacadeCacheLevel = field(
        default=ModelFacadeCacheLevel.LOW)

    def __post_init__(self):
        cache_executor = self.cache_level >= ModelFacadeCacheLevel.EXECUTOR
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'cache executor: {cache_executor}')
        self._executor = PersistedWork(
            '_executor', self, cache_global=cache_executor)
        executor = self.executor
        if cache_executor:
            self.config_factory = executor.config_factory
        self.debuged = False

    @property
    @persisted('_executor')
    def executor(self) -> ModelExecutor:
        """Return a cached instance of the executor tied to the instance of this class.

        """
        return self._create_executor()

    @property
    def config(self) -> Configurable:
        """Return the configuration used to created resources for the facade.

        """
        return self.config_factory.config

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

    # @property
    # def cache_level(self) -> ModelFacadeCacheLevel:
    #     return self._cache_level

    # @cache_level.setter
    # def cache_level(self, cache_level: ModelFacadeCacheLevel):
    #     self._cache_level = cache_level
    #     self._set_executor_cache_level(self.executor, cache_level)

    def set_dropout(self, dropout: float):
        """Set the dropout for the entire network.

        """
        self.executor.net_settings.dropout = dropout

    def set_epochs(self, n_epochs: int):
        """Set the number of epochs for training and validation.

        """
        self.executor.model_settings.epochs = n_epochs

    def set_learning_rate(self, learning_rate: float):
        """The learning rate to set on the optimizer.

        """
        self.executor.model_settings.learning_rate = learning_rate

    def _create_executor(self) -> ModelExecutor:
        """Create a new instance of an executor.  Used by :py:attrib:~`executor`.

        """
        executor = self.config_factory(
            self.executor_name,
            progress_bar=self.progress_bar,
            progress_bar_cols=self.progress_bar_cols)
        cache_batches = self.cache_level >= ModelFacadeCacheLevel.BATCHES
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'setting batch caching: {cache_batches}')
        executor.model_settings.cache_batches = cache_batches
        #self._set_executor_cache_level(executor, self.cache_level)
        return executor

    # def _set_executor_cache_level(self, executor, cache_level):
    #     cache_batches = cache_level >= ModelFacadeCacheLevel.BATCHES
    #     cache_executor = cache_level >= ModelFacadeCacheLevel.EXECUTOR
    #     if logger.isEnabledFor(logging.DEBUG):
    #         logger.debug(f'setting batch caching: {cache_batches}')
    #     executor.model_settings.cache_batches = cache_batches
    #     self._executor.cache_global = cache_executor

    def clear_batches(self):
        """Clear and deallocate all batches in the executor.

        """
        logger.info('deallocating batches')
        self.executor.clear_batches()

    def deallocate(self, cache_level: ModelFacadeCacheLevel = None):
        cache_level = self.cache_level if cache_level is None else cache_level
        super().deallocate()
        if cache_level < ModelFacadeCacheLevel.EXECUTOR:
            logger.info('clearing executor')
            self.clear_executor()
        if cache_level == ModelFacadeCacheLevel.NONE:
            logger.info('deallocating config_factory')
            self.config_factory.deallocate()

    def clear_executor(self):
        """Clear out any cached executor.

        """
        executor = self.executor
        executor.deallocate()
        self._executor.clear()

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
        facade = cls(mm.config_factory, *args, **kwargs)
        facade._executor.set(mm.load_executor())
        return facade

    def debug(self):
        """Debug the model by setting the configuration to debug mode and invoking a
        single forward pass.  Logging must be configured properly to get the
        output, which is typically just invoking
        :py:meth:`logging.basicConfig`.

        """
        executor = self.executor
        executor.reset()
        self._configure_debug_logging()
        executor.progress_bar = False
        executor.net_settings.debug = True
        executor.model_settings.batch_limit = 1
        self.debuged = True
        executor.train()

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
            return executor.train(description)

    def test(self, description: str = None) -> ModelResult:
        """Load the model from disk and test it.

        """
        executor = self.executor
        if self.debuged:
            raise ValueError('testing is not allowed in debug mode')
        if self.load_type == 'executor':
            path = executor.model_settings.path
            logger.info(f'testing from path: {path}')
            mm = ModelManager(path, self.config_factory)
            executor = mm.load_executor()
        elif self.load_type == 'model':
            executor.load()
        elif self.load_type == 'none':
            pass
        else:
            raise ValueError(f'unknown load_type: {self.load_type}')
        logger.info('testing...')
        res = executor.test(description)
        if self.writer is not None:
            res.write(writer=self.writer, verbose=False)
        return res

    def plot(self, res: ModelResult, figsize: Tuple[int, int] = (15, 5),
             title: str = None):
        """Plot results of ``res`` using ``matplotlib``.

        :see: :class:`.ModelResultGrapher`

        """
        if title is None:
            title = self.executor.model_name
        grapher = ModelResultGrapher(title, figsize)
        grapher.plot(res)

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
        res.write(depth, writer, verbose)

    def get_predictions(self, *args, **kwargs) -> pd.DataFrame:
        """Return the predictions made during the test phase of the model execution.
        The arguments are passed to :meth:`ModelExecutor.get_predictions`.

        :see: :meth:`.ModelExecutor.get_predictions`

        """
        executor = self.executor
        executor.load()
        return executor.get_predictions(*args, **kwargs)

    def write_predictions(self, lines: int = 10, writer: TextIOWrapper = None):
        """Print the predictions made during the test phase of the model execution.

        :param lines: the number of lines of the predictions data frame to be
                      printed

        :param writer: the data sink

        """
        writer = self.writer if writer is None else writer
        preds = self.get_predictions()
        print(preds.head(lines), file=writer)

    def write(self, depth: int = 0, writer: TextIOWrapper = None,
              include_config: bool = False):
        writer = self.writer if writer is None else writer
        writer = sys.stdout if writer is None else writer
        bmeta = None
        try:
            bmeta = self.batch_metadata
        except AttributeError:
            pass
        self._write_line(f'{self.executor.name}:', depth, writer)
        self.executor.write(depth + 1, writer)
        if bmeta is not None:
            self._write_line('metadata:', depth, writer)
            bmeta.write(depth + 1, writer)
        if include_config:
            self._write_line('config:', depth, writer)
            self.config.write(depth + 1, writer)

    def _configure_debug_logging(self):
        """When debuging the model, configure the logging system for output.  The
        correct loggers need to be set to debug mode to print the model
        debugging information such as matrix shapes.

        """
        lg = logging.getLogger('zensols.deepnlp.vectorize.vectorizers')
        lg.setLevel(logging.INFO)
        lg = logging.getLogger(__name__ + '.module')
        lg.setLevel(logging.DEBUG)

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
