"""Client entry point to the model.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Any, Callable, List
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
from zensols.deeplearn import NetworkSettings, ModelSettings
from zensols.deeplearn.vectorize import (
    SparseTensorFeatureContext,
    FeatureVectorizerManagerSet,
)
from zensols.deeplearn.batch import BatchStash, BatchMetadata, DataPoint
from zensols.deeplearn.result import (
    EpochResult,
    ModelResult,
    ModelResultGrapher,
    ModelResultManager,
    PredictionsDataFrameFactory,
)
from . import (
    ModelManager,
    ModelExecutor,
    FacadeClassExplorer,
    MetadataNetworkSettings,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelFacade(PersistableContainer, Writable):
    """This class provides easy to use client entry points to the model executor,
    which trains, validates, tests, saves and loads the model.

    More common attributes, such as the learning rate and number of epochs, are
    properties that dispatch to :py:attrib:~`executor`--for the others, go
    directly to the property.

    :param factory: the factory used to create the executor

    :param progress_bar: create text/ASCII based progress bar if ``True``

    :param progress_bar_cols: the number of console columns to use for the
                              text/ASCII based progress bar

    :param executor_name: the configuration entry name for the executor, which
                          defaults to ``executor``

    :param cache_level: determines how much and when to deallocate (see
                        :class:`.ModelFacadeCacheLevel`)

    :see zensols.deeplearn.domain.ModelSettings:

    """
    SINGLETONS = {}

    config: Configurable
    progress_bar: bool = field(default=True)
    progress_bar_cols: int = field(default=79)
    executor_name: str = field(default='executor')
    cache_batches: bool = field(default=True)
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
    def feature_stash(self) -> Stash:
        """The stash used to generate the feature, which is not to be confused
        with the batch source stash``batch_stash``.

        """
        return self.executor.feature_stash

    @property
    def batch_stash(self) -> BatchStash:
        """The stash used to encode and decode batches by the executor.

        """
        return self.executor.batch_stash

    @property
    def vectorizer_manager_set(self) -> FeatureVectorizerManagerSet:
        """Return the vectorizer manager set used for the facade.  This is taken from
        the executor's batch stash.

        """
        return self.batch_stash.vectorizer_manager_set

    @property
    def batch_metadata(self) -> BatchMetadata:
        """Return the batch metadata used on the executor.  This will only work if
        :py:attrib:~`net_settings` extends from
        :class:`.MetadataNetworkSettings`.

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
        executor = self.executor
        self._configure_debug_logging()
        executor.debug = debug_value
        executor.progress_bar = False
        executor.model_settings.batch_limit = 1
        self.debuged = True
        executor.train()

    def persist_results(self, persist_plot_result: bool = True):
        """Save the last recorded result during an :py:meth:`.Executor.train` or
        :py:meth:`.Executor.test` invocation to disk.  Optionally also save a
        plotted graphics file to disk as well when
        :py:attrib:~`persist_plot_result` is set to ``True``.

        Note that in Jupyter notebooks, this method has the side effect of
        plotting the results in the cell when ``persist_plot_result`` is
        ``True``.

        :param persist_plot_result: if ``True``, plot and save the graph as a
                                    PNG file to the results directory

        """
        executor = self.executor
        if executor.result_manager is not None:
            rm: ModelResultManager = self.executor.result_manager
            rm.dump(executor.model_result)
            if persist_plot_result:
                self.plot_result(save=True)

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

    def get_grapher(self, figsize: Tuple[int, int] = (15, 5),
                    title: str = None) -> ModelResultGrapher:
        """Return an instance of a model grapher.  This class can plot results of
        ``res`` using ``matplotlib``.

        :see: :class:`.ModelResultGrapher`

        """
        result_manager: ModelResultManager = self.executor.result_manager
        if title is None:
            title = self.executor.model_name
        path_dir = result_manager.get_next_text_path().parent
        path = path_dir / f'graph-{result_manager.get_last_key(False)}.png'
        return ModelResultGrapher(title, figsize, save_path=path)

    @property
    def last_result(self) -> ModelResult:
        """The last recorded result during an :py:meth:`.Executor.train` or
        :py:meth:`.Executor.test` invocation is used.

        """
        res = self.executor.model_result
        if res is None:
            rm: ModelResultManager = self.executor.result_manager
            if rm is None:
                rm = ValueError('no result manager available')
            res = rm.load()
            if res is None:
                raise ValueError('no results found')
        return res

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
        result = self.last_result
        if result is None:
            raise ValueError('no result to plot; invoke train() and or test()')
        grapher = self.get_grapher()
        grapher.plot([result])
        if save:
            grapher.save()
        if show:
            grapher.show()
        return result

    def write_result(self, depth: int = 0, writer: TextIOBase = sys.stdout,
                     verbose: bool = False):
        """Load the last set of results from the file system and print them out.  The
        result to print is taken from :py:meth:`last_result`

        """
        logging.getLogger('zensols.deeplearn.result').setLevel(logging.INFO)
        logger.info('load previous results')
        res = self.last_result
        res.write(depth, writer, include_settings=verbose,
                  include_converged=verbose, include_config=verbose)

    def get_predictions(self, column_names: List[str] = None,
                        transform: Callable[[DataPoint], tuple] = None,
                        name: str = None) -> pd.DataFrame:
        """Generate a Pandas dataframe containing all predictinos from the test data
        set.

        :param column_names: the list of string column names for each data item
                             the list returned from ``data_point_transform`` to
                             be added to the results for each label/prediction

        :param transform: a function that returns a tuple, each with an element
                          respective of ``column_names`` to be added to the
                          results for each label/prediction; if ``None`` (the
                          default), ``str`` used

        :param name: the key of the previously saved results to fetch the
                     results, or ``None`` (the default) to get the last result
                     set saved

        """
        res = self.last_result
        if not res.test.contains_results:
            raise ValueError('no test results found')
        res: EpochResult = res.test.results[0]
        df_fac = PredictionsDataFrameFactory(
            res, self.batch_stash, column_names, transform)
        return df_fac.dataframe

    def write_predictions(self, lines: int = 10):
        """Print the predictions made during the test phase of the model execution.

        :param lines: the number of lines of the predictions data frame to be
                      printed

        :param writer: the data sink

        """
        preds = self.get_predictions()
        print(preds.head(lines), file=self.writer)

    def _create_facade_explorer(self):
        return FacadeClassExplorer()

    def write(self, depth: int = 0, writer: TextIOBase = None,
              include_executor: bool = True, include_metadata: bool = True,
              include_config: bool = False,
              include_object_graph: bool = False):
        writer = self.writer if writer is None else writer
        writer = sys.stdout if writer is None else writer
        bmeta = None
        try:
            bmeta = self.batch_metadata
        except AttributeError:
            pass
        if include_executor:
            self._write_line(f'{self.executor.name}:', depth, writer)
            self.executor.write(depth + 1, writer)
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

    def configure_cli_logging(self):
        """"Configure command line (or Python REPL) debugging.  Each facade can turn on
        name spaces that make sense as useful information output for long
        running training/testing iterations.

        """
        names = [
            # load messages
            'zensols.deeplearn.batch.stash',
            # save results messages
            'zensols.deeplearn.result',
            # load/save messages
            __name__]
        for name in names:
            logging.getLogger(name).setLevel(logging.INFO)

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
