"""Client entry point to the model.

"""
__author__ = 'Paul Landes'

from typing import Any, Callable, List, Union
from dataclasses import dataclass, field
import sys
import logging
import pandas as pd
from io import TextIOBase
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
    Stash,
)
from zensols.dataset import DatasetSplitStash
from zensols.deeplearn import NetworkSettings, ModelSettings
from zensols.deeplearn.vectorize import (
    SparseTensorFeatureContext,
    FeatureVectorizerManagerSet,
)
from zensols.deeplearn.batch import BatchStash, BatchMetadata, DataPoint
from zensols.deeplearn.result import (
    EpochResult,
    ModelResult,
    ModelResultManager,
    PredictionsDataFrameFactory,
)
from . import (
    ModelManager,
    ModelExecutor,
    FacadeClassExplorer,
    MetadataNetworkSettings,
    ResultAnalyzer,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelFacade(PersistableContainer, Writable):
    """This class provides easy to use client entry points to the model executor,
    which trains, validates, tests, saves and loads the model.

    More common attributes, such as the learning rate and number of epochs, are
    properties that dispatch to :py:obj:`executor`.  For the others, go
    directly to the property.

    :param factory: the factory used to create the executor

    :param progress_bar: create text/ASCII based progress bar if ``True``

    :param progress_bar_cols: the number of console columns to use for the
                              text/ASCII based progress bar

    :param executor_name: the configuration entry name for the executor, which
                          defaults to ``executor``

    :param writer: the writer to this in methods like :meth:`train`, and
                   :meth:`test` for writing performance metrics results and
                   predictions or ``None`` to not output them

    :see: :class:`zensols.deeplearn.domain.ModelSettings`

    """
    SINGLETONS = {}

    config: Configurable
    progress_bar: bool = field(default=True)
    progress_bar_cols: int = field(default=79)
    executor_name: str = field(default='executor')
    writer: TextIOBase = field(default=sys.stdout)

    def __post_init__(self):
        super().__init__()
        self._config_factory = PersistedWork('_config_factory', self)
        self._executor = PersistedWork('_executor', self)
        self.debuged = False

    @classmethod
    def get_singleton(cls, *args, **kwargs) -> Any:
        key = str(cls)
        inst = cls.SINGLETONS.get(key)
        if inst is None:
            inst = cls(*args, **kwargs)
            cls.SINGLETONS[key] = inst
        return inst

    def _create_executor(self) -> ModelExecutor:
        """Create a new instance of an executor.  Used by :obj:`executor`.

        """
        logger.info('creating new executor')
        executor = self.config_factory(
            self.executor_name,
            progress_bar=self.progress_bar,
            progress_bar_cols=self.progress_bar_cols)
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
        """Return the executor's network settings.

        """
        return self.executor.net_settings

    @property
    def model_settings(self) -> ModelSettings:
        """Return the executor's model settings.

        """
        return self.executor.model_settings

    @property
    def result_manager(self) -> ModelResultManager:
        """Return the executor's result manager.

        """
        rm: ModelResultManager = self.executor.result_manager
        if rm is None:
            rm = ValueError('no result manager available')
        return rm

    @property
    def feature_stash(self) -> Stash:
        """The stash used to generate the feature, which is not to be confused
        with the batch source stash ``batch_stash``.

        """
        return self.executor.feature_stash

    @property
    def batch_stash(self) -> BatchStash:
        """The stash used to encode and decode batches by the executor.

        """
        return self.executor.batch_stash

    @property
    def dataset_stash(self) -> DatasetSplitStash:
        """The stash used to encode and decode batches split by dataset.

        """
        return self.executor.dataset_stash

    @property
    def vectorizer_manager_set(self) -> FeatureVectorizerManagerSet:
        """Return the vectorizer manager set used for the facade.  This is taken from
        the executor's batch stash.

        """
        return self.batch_stash.vectorizer_manager_set

    @property
    def batch_metadata(self) -> BatchMetadata:
        """Return the batch metadata used on the executor.  This will only work if
        :obj:`net_settings` extends from :class:`.MetadataNetworkSettings`.

        :see: :class:`zensols.deepnlp.model.module.EmbeddingNetworkSettings`

        """
        ns = self.net_settings
        if isinstance(ns, MetadataNetworkSettings):
            return ns.batch_metadata_factory()

    @property
    def label_attribute_name(self):
        """Get the label attribute name.

        """
        bmeta = self.batch_metadata
        if bmeta is not None:
            return bmeta.mapping.label_attribute_name

    @property
    def dropout(self) -> float:
        """The dropout for the entire network.

        """
        return self.net_settings.dropout

    @dropout.setter
    def dropout(self, dropout: float):
        """The dropout for the entire network.

        """
        self.net_settings.dropout = dropout

    @property
    def epochs(self) -> int:
        """The number of epochs for training and validation.

        """
        return self.model_settings.epochs

    @epochs.setter
    def epochs(self, n_epochs: int):
        """The number of epochs for training and validation.

        """
        self.model_settings.epochs = n_epochs

    @property
    def learning_rate(self) -> float:
        """The learning rate to set on the optimizer.

        """
        return self.model_settings.learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate: float):
        """The learning rate to set on the optimizer.

        """
        self.executor.model_settings.learning_rate = learning_rate

    @property
    def cache_batches(self) -> bool:
        """The cache_batches for the entire network.

        """
        return self.model_settings.cache_batches

    @cache_batches.setter
    def cache_batches(self, cache_batches: bool):
        """The cache_batches for the entire network.

        """
        # if the caching strategy changed, be safe and deallocate and purge to
        # lazy recreate everything
        if self.model_settings.cache_batches != cache_batches:
            self.clear()
        self.model_settings.cache_batches = cache_batches

    def clear(self):
        """Clear out any cached executor.

        """
        if logger.isEnabledFor(logging.INFO):
            logger.info('clearing')
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

    def deallocate(self):
        super().deallocate()
        self.SINGLETONS.pop(str(self.__class__), None)

    @classmethod
    def load_from_path(cls, path: Path, *args, **kwargs) -> Any:
        """Construct a new facade from the data saved in a persisted model file.  This
        uses the :py:meth:`.ModelManager.load_from_path` to reconstruct the
        returned facade, which means some attributes are taken from default if
        not taken from ``*args`` or ``**kwargs``.

        Arguments:
           Passed through to the initializer of invoking class ``cls``.

        :return: a new instance of a :class:`.ModelFacade`
        :see: :py:meth:`.ModelManager.load_from_path`

        """
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'loading from facade from {path}')
        mm = ModelManager.load_from_path(path)
        if 'executor_name' not in kwargs:
            kwargs['executor_name'] = mm.model_executor_name
        executor = mm.load_executor()
        mm.config_factory.deallocate()
        facade = cls(executor.config, *args, **kwargs)
        facade._config_factory.set(executor.config_factory)
        facade._executor.set(executor)
        return facade

    def debug(self, debug_value: Union[bool, int] = True):
        """Debug the model by setting the configuration to debug mode and invoking a
        single forward pass.  Logging must be configured properly to get the
        output, which is typically just invoking
        :py:meth:`logging.basicConfig`.

        """
        executor = self.executor
        self._configure_debug_logging()
        executor.debug = debug_value
        executor.progress_bar = False
        executor.model_settings.batch_limit = 1
        self.debuged = True
        executor.train()

    def persist_result(self):
        """Save the last recorded result during an :py:meth:`.Executor.train` or
        :py:meth:`.Executor.test` invocation to disk.  Optionally also save a
        plotted graphics file to disk as well when :obj:`persist_plot_result`
        is set to ``True``.

        Note that in Jupyter notebooks, this method has the side effect of
        plotting the results in the cell when ``persist_plot_result`` is
        ``True``.

        :param persist_plot_result: if ``True``, plot and save the graph as a
                                    PNG file to the results directory

        """
        executor = self.executor
        if executor.result_manager is not None:
            self.result_manager.dump(executor.model_result)

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
        if self.writer is not None:
            res.write(writer=self.writer)
        return res

    def train_production(self, description: str = None) -> ModelResult:
        """Train on the training and test data sets, then test

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
            res = executor.train_production(description)
        return res

    @property
    def last_result(self) -> ModelResult:
        """The last recorded result during an :meth:`.ModelExecutor.train` or
        :meth:`.ModelExecutor.test` invocation is used.

        """
        res = self.executor.model_result
        if res is None:
            rm: ModelResultManager = self.result_manager
            res = rm.load()
            if res is None:
                raise ValueError('no results found')
        return res

    def write_result(self, depth: int = 0, writer: TextIOBase = sys.stdout,
                     include_settings: bool = False,
                     include_converged: bool = False,
                     include_config: bool = False):
        """Load the last set of results from the file system and print them out.  The
        result to print is taken from :obj:`last_result`

        :param depth: the number of indentation levels

        :param writer: the data sink

        :param include_settings: whether or not to include model and network
                                 settings in the output

        :param include_config: whether or not to include the configuration in
                               the output

        """
        if logger.isEnabledFor(logging.INFO):
            logger.info('load previous results')
        res = self.last_result
        res.write(depth, writer, include_settings=include_settings,
                  include_converged=include_converged,
                  include_config=include_config)

    def plot_result(self, result: ModelResult = None, save: bool = False,
                    show: bool = False) -> ModelResult:
        """Plot results and optionally save and show them.  If this is called in a
        Jupyter notebook, the plot will be rendered in a cell.

        :param result: the result to plot, or if ``None``, use
                       :py:meth:`last_result`

        :param save: if ``True``, save the plot to the results directory with
                     the same naming as the last data results

        :param show: if ``True``, invoke ``matplotlib``'s ``show`` function to
                     visualize in a non-Jupyter environment

        :return: the result used to graph, which comes from the executor when
                 none is given to the invocation

        """
        result = self.last_result if result is None else result
        grapher = self.executor.result_manager.get_grapher()
        grapher.plot([result])
        if save:
            grapher.save()
        if show:
            grapher.show()
        return result

    def get_predictions(self, column_names: List[str] = None,
                        transform: Callable[[DataPoint], tuple] = None,
                        batch_limit: int = sys.maxsize,
                        name: str = None) -> pd.DataFrame:
        """Generate a Pandas dataframe containing all predictinos from the test data
        set.

        :param column_names: the list of string column names for each data item
                             the list returned from ``data_point_transform`` to
                             be added to the results for each label/prediction

        :param transform:

            a function that returns a tuple, each with an element respective of
            ``column_names`` to be added to the results for each
            label/prediction; if ``None`` (the default), ``str`` used (see the
            `Iris Jupyter Notebook
            <https://github.com/plandes/deeplearn/blob/master/notebook/iris.ipynb>`_
            example)

        :param batch_limit: the max number of batche of results to output

        :param name: the key of the previously saved results to fetch the
                     results, or ``None`` (the default) to get the last result
                     set saved

        """
        if name is None:
            res = self.last_result
        else:
            rm: ModelResultManager = self.result_manager
            res = rm.load(name)
        if not res.test.contains_results:
            raise ValueError('no test results found')
        res: EpochResult = res.test.results[0]
        df_fac = PredictionsDataFrameFactory(
            res, self.batch_stash, column_names, transform, batch_limit)
        return df_fac.dataframe

    def write_predictions(self, lines: int = 10):
        """Print the predictions made during the test phase of the model execution.

        :param lines: the number of lines of the predictions data frame to be
                      printed

        :param writer: the data sink

        """
        preds = self.get_predictions()
        print(preds.head(lines), file=self.writer)

    def get_result_analyzer(self, key: str = None,
                            cache_previous_results: bool = False) \
            -> ResultAnalyzer:
        """Return a results analyzer for comparing in flight training progress.

        """
        rm: ModelResultManager = self.result_manager
        if key is None:
            key = rm.get_last_key()
        return ResultAnalyzer(self.executor, key, cache_previous_results)

    def _create_facade_explorer(self) -> FacadeClassExplorer:
        """Return a facade explorer used to print the facade's object graph.

        """
        return FacadeClassExplorer()

    def write(self, depth: int = 0, writer: TextIOBase = None,
              include_executor: bool = True, include_metadata: bool = True,
              include_settings: bool = True, include_model: bool = True,
              include_config: bool = False, include_object_graph: bool = False):
        writer = self.writer if writer is None else writer
        writer = sys.stdout if writer is None else writer
        bmeta = None
        try:
            bmeta = self.batch_metadata
        except AttributeError:
            pass
        if include_executor:
            self._write_line(f'{self.executor.name}:', depth, writer)
            self.executor.write(depth + 1, writer,
                                include_settings=include_settings,
                                include_model=include_model)
        if include_metadata and bmeta is not None:
            self._write_line('metadata:', depth, writer)
            bmeta.write(depth + 1, writer)
        if include_object_graph:
            self._write_line('graph:', depth, writer)
            ce = self._create_facade_explorer()
            ce.write(self, depth=depth + 1, writer=writer)
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
        for name in ['zensols.deeplearn.model.executor', __name__]:
            logging.getLogger(name).setLevel(logging.DEBUG)

    def _configure_cli_logging(self, info_loggers: List[str],
                               debug_loggers: List[str]):
        info_loggers.extend([
            # load messages
            'zensols.deeplearn.batch.stash',
            # multi-process (i.e. batch creation)
            'zensols.multi.stash',
            # validation/training loss messages
            'zensols.deeplearn.model.executor.status',
            'zensols.deeplearn.model.lifecycle',
            # load/save messages
            __name__])
        if not self.progress_bar:
            info_loggers.extend([
                # save results messages
                'zensols.deeplearn.result',
                # validation/training loss messages
                'zensols.deeplearn.model.executor.progress',
                # model save/load
                'zensols.deeplearn.model.manager'])

    @staticmethod
    def configure_default_cli_logging(log_level: int = logging.WARNING):
        """Configure the logging system with the defaults.

        """
        fmt = '%(asctime)s[%(levelname)s]%(name)s: %(message)s'
        logging.basicConfig(format=fmt, level=log_level)

    def configure_cli_logging(self, log_level: int = None):
        """"Configure command line (or Python REPL) debugging.  Each facade can turn on
        name spaces that make sense as useful information output for long
        running training/testing iterations.

        This calls :py:meth:`_configure_cli_logging` to collect the names of
        loggers at various levels.

        """
        info = []
        debug = []
        if log_level is not None:
            self.configure_default_cli_logging(log_level)
        self._configure_cli_logging(info, debug)
        for name in info:
            logging.getLogger(name).setLevel(logging.INFO)
        for name in debug:
            logging.getLogger(name).setLevel(logging.DEBUG)

    def configure_jupyter(self, log_level: int = logging.WARNING,
                          progress_bar_cols: int = 120):
        """Configures logging and other configuration related to a Jupyter notebook.
        This is just like :py:meth:`configure_cli_logging`, but adjusts logging
        for what is conducive for reporting in Jupyter cells.

        ;param log_level: the default logging level for the logging system

        :param progress_bar_cols: the number of columns to use for the progress
                                  bar

        """
        self.configure_cli_logging(log_level)
        for name in [
                # turn off loading messages
                'zensols.deeplearn.batch.stash']:
            logging.getLogger(name).setLevel(logging.WARNING)
        # number of columns for the progress bar
        self.executor.progress_bar_cols = progress_bar_cols
        # turn off console output (non-logging)
        self.writer = None

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
