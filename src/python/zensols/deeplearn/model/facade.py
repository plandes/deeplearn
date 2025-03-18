"""Client entry point to the model.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import (
    Any, Callable, List, Union, Iterable, Type, Dict, Optional, ClassVar
)
from dataclasses import dataclass, field, InitVar
import sys
import os
import logging
from io import TextIOBase
from pathlib import Path
from zensols.util import time
from zensols.config import (
    Configurable,
    ConfigFactory,
    Writable,
    ImportConfigFactory,
)
from zensols.persist import (
    persisted, PersistableContainer, PersistedWork,
    Deallocatable, Stash,
)
from zensols.dataset import DatasetSplitStash
from zensols.deeplearn import ModelError, NetworkSettings, ModelSettings
from zensols.deeplearn.vectorize import (
    SparseTensorFeatureContext, FeatureVectorizerManagerSet,
)
from zensols.deeplearn.batch import (
    Batch, DataPoint, BatchStash, BatchMetadata, BatchFeatureMapping
)
from zensols.deeplearn.model import BatchMetrics
from zensols.deeplearn.result import (
    EpochResult, ModelResult, ModelResultManager, ModelResultReporter,
    PredictionsDataFrameFactory,
)
from . import (
    ModelManager, ModelExecutor, PredictionMapper,
    FacadeClassExplorer, ResultAnalyzer
)

logger = logging.getLogger(__name__)


@dataclass
class ModelFacade(PersistableContainer, Writable):
    """This class provides easy to use client entry points to the model
    executor, which trains, validates, tests, saves and loads the model.

    More common attributes, such as the learning rate and number of epochs, are
    properties that dispatch to :py:obj:`executor`.  For the others, go
    directly to the property.

    :see: :class:`zensols.deeplearn.domain.ModelSettings`

    """
    _SINGLETONS: ClassVar[Dict[str, ModelFacade]] = {}

    config: Configurable = field()
    """The configuraiton used to create the facade, and used to create a new
    configuration factory to load models.

    """
    config_factory: InitVar[ConfigFactory] = field(default=None)
    """The configuration factory used to create this facade, or ``None`` if no
    factory was used.

    """
    progress_bar: bool = field(default=True)
    """Create text/ASCII based progress bar if ``True``."""

    progress_bar_cols: Union[str, int] = field(default='term')
    """The number of console columns to use for the text/ASCII based progress
    bar.  If the value is ``term``, then use the terminal width.

    """
    executor_name: str = field(default='executor')
    """The configuration entry name for the executor, which defaults to
    ``executor``.

    """
    writer: TextIOBase = field(default=sys.stdout)
    """The writer to this in methods like :meth:`train`, and :meth:`test` for
    writing performance metrics results and predictions or ``None`` to not
    output them.

    """
    predictions_dataframe_factory_class: Union[str, Type[PredictionsDataFrameFactory]] = \
        field(default=PredictionsDataFrameFactory)
    """The factory class used to create predictions, or the string of the
    configuration section.  If the latter, the factory is created using
    :obj:`config_factory`.

    :see: :meth:`get_predictions_factory`

    """
    model_result_reporter_class: Union[str, Type[ModelResultReporter]] = \
        field(default=ModelResultReporter)
    """The class used to report result summaries over all the runs."""

    result_name: str = field(default=None)
    """A descriptor used in the results, which is useful when making
    incremental hyperparameter changes to the model.

    """
    def __post_init__(self, config_factory: ConfigFactory):
        super().__init__()
        self._init_config_factory(config_factory)
        self._config_factory = PersistedWork('_config_factory', self)
        self._executor = PersistedWork('_executor', self)
        self.debuged = False
        if self.progress_bar_cols == 'term':
            try:
                term_width = os.get_terminal_size()[0]
                # make space for embedded validation loss messages
                self.progress_bar_cols = term_width - 5
            except OSError:
                logger.debug('unable to automatically determine ' +
                             'terminal width--skipping')
                self.progress_bar_cols = None

    @classmethod
    def get_singleton(cls, *args, **kwargs) -> ModelFacade:
        """Return the singleton application using ``args`` and ``kwargs`` as
        initializer arguments.

        """
        key: str = str(cls)
        inst: ModelFacade = cls._SINGLETONS.get(key)
        if inst is None:
            inst = cls(*args, **kwargs)
            cls._SINGLETONS[key] = inst
        return inst

    def _init_config_factory(self, config_factory: ConfigFactory):
        if isinstance(config_factory, ImportConfigFactory):
            params = config_factory.__dict__
            keeps = set('reload shared reload_pattern'.split())
            params = {k: params[k] for k in set(params.keys()) & keeps}
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'import config factory params: {params}')
            self._config_factory_params = params
        else:
            self._config_factory_params = {}

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
        return ImportConfigFactory(self.config, **self._config_factory_params)

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
            rm = ModelError('No result manager available')
        return rm

    @property
    def cross_fold_result_manager(self) -> ModelResultManager:
        """Return the executor's cross-fold validation result manager.

        """
        rm: ModelResultManager = self.executor.cross_fold_result_manager
        if rm is None:
            rm = ModelError('No result manager available')
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
    def cross_fold_batch_stash(self) -> BatchStash:
        """The stash used to encode and decode batches by the executor.

        """
        return self.executor.cross_fold_batch_stash

    @property
    def dataset_stash(self) -> DatasetSplitStash:
        """The stash used to encode and decode batches split by dataset.

        """
        return self.executor.dataset_stash

    @property
    def vectorizer_manager_set(self) -> FeatureVectorizerManagerSet:
        """Return the vectorizer manager set used for the facade.  This is taken
        from the executor's batch stash.

        """
        return self.batch_stash.vectorizer_manager_set

    @property
    def batch_metadata(self) -> BatchMetadata:
        """Return the batch metadata used on the executor.

        :see: :class:`zensols.deepnlp.model.module.EmbeddingNetworkSettings`

        """
        return self.batch_stash.batch_metadata

    @property
    def label_attribute_name(self):
        """Get the label attribute name.

        """
        bmeta = self.batch_metadata
        if bmeta is not None:
            return bmeta.mapping.label_attribute_name

    def _notify(self, event: str, context: Any = None):
        """Notify observers of events from this class.

        """
        self.model_settings.observer_manager.notify(event, self, context)

    def remove_metadata_mapping_field(self, attr: str) -> bool:
        """Remove a field by attribute if it exists across all metadata
        mappings.

        This is useful when a very expensive vectorizer slows down tasks, such
        as prediction, on a single run of a program.  For this use case,
        override :meth:`predict` to call this method before calling the super
        ``predict`` method.

        :param attr: the name of the field's attribute to remove

        :return: ``True`` if the field was removed, ``False`` otherwise

        """
        removed = False
        meta: BatchMetadata = self.batch_metadata
        mapping: BatchFeatureMapping
        for mapping in meta.mapping.manager_mappings:
            removed = removed or mapping.remove_field(attr)
        return removed

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

    def get_batch_metrics(self) -> BatchMetrics:
        """Return a class that performs metrics on the :obj:`batch_stash`."""
        return BatchMetrics(self.batch_stash)

    def get_cross_fold_batch_metrics(self) -> BatchMetrics:
        """Return a class that performs metrics on the :obj:`batch_stash`."""
        return BatchMetrics(self.cross_fold_batch_stash)

    def clear(self):
        """Clear out any cached executor."""
        if logger.isEnabledFor(logging.INFO):
            logger.info('clearing')
        executor = self.executor
        config_factory = self.config_factory
        executor.deallocate()
        config_factory.deallocate()
        self._executor.clear()
        self._config_factory.clear()

    def reload(self):
        """Clears all state and reloads the configuration."""
        self.clear()
        self.config.reload()

    def deallocate(self):
        super().deallocate()
        self._SINGLETONS.pop(str(self.__class__), None)

    @classmethod
    def load_from_path(cls, path: Path, *args, **kwargs) -> ModelFacade:
        """Construct a new facade from the data saved in a persisted model file.
        This uses the :py:meth:`.ModelManager.load_from_path` to reconstruct the
        returned facade, which means some attributes are taken from default if
        not taken from ``*args`` or ``**kwargs``.

        :param path: the path of the model file on the file system

        :param model_config_overwrites:
            a :class:`~zensols.config.configbase.Configurable` used to
            overrwrite configuration in the model package config

        :param args: passed through to the initializer of invoking class ``cls``

        :param kwargs: passed through to the initializer of invoking class
                       ``cls``

        :return: a new instance of a :class:`.ModelFacade`

        :see: :meth:`.ModelManager.load_from_path`

        """
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'loading from facade from {path}')
        # load the model from disk
        mm: ModelManager = ModelManager.load_from_path(path)
        # the configuration section name of the executor
        if 'executor_name' not in kwargs:
            kwargs['executor_name'] = mm.model_executor_name
        # merge in external configuration into the model config before the
        # configuration factory begins to instantiate
        src_conf: Optional[Configurable] = kwargs.pop(
            'model_config_overwrites', None)
        # instantiate the executor
        executor: ModelExecutor = mm.load_executor(src_conf)
        # set the path of the model
        executor.model_settings.path = path
        # cleanup any CUDA memory
        mm.config_factory.deallocate()
        # create a new facade and configure with the loaded model
        facade: ModelFacade = cls(executor.config, *args, **kwargs)
        facade._config_factory.set(executor.config_factory)
        facade._executor.set(executor)
        facade._model_config = mm.config_factory.config
        return facade

    def update_model_config_factory(self, path: Path = None,
                                    config_factory: ConfigFactory = None):
        """Update the model's configuration factory and configuration that was
        previously used to train the model.  The modification is made in the
        model's state file.

        **Important**: the caller is responsible for creating the
        :obj:`executor` model (i.e. either directly with
        :meth:`.ModelExecutor._get_or_create_model` or indirectly with
        :meth:`.ModelExector.write_model`).

        :param path: the path of the model directory, which defaults to the
                     model data directory (rather than the model results
                     directory)

        :param config_factory: the configuration factory that will be replaced,
                               which defaults to :obj:`config_factory`

        """
        if config_factory is None:
            self.executor._get_or_create_model()
            config_factory = self.config_factory
        mm: ModelManager
        if path is None:
            mm = self.executor.model_manager
        else:
            # load the model from disk
            mm = ModelManager.load_from_path(path)
        mm._update_config_factory(config_factory)

    @property
    def model_config(self) -> Configurable:
        """The configurable packaged with the model if this facade was created
        with :mth:`load_from_path`.

        """
        if hasattr(self, '_model_config'):
            return self._model_config

    def debug(self, debug_value: Union[bool, int] = True):
        """Debug the model by setting the configuration to debug mode and
        invoking a single forward pass.  Logging must be configured properly to
        get the output, which is typically just invoking
        :py:meth:`logging.basicConfig`.

        :param debug_value: ``True`` turns on executor debugging; if an
                            ``int``, the higher the value, the more the logging

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
        executor: ModelExecutor = self.executor
        rmng: ModelResultManager = self.result_manager
        if executor.result_manager is not None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'dumping model result: {executor.model_result}')
            rmng.dump(executor.model_result)

    def train(self) -> ModelResult:
        """Train and test or just debug the model depending on the
        configuration.

        """
        result_name: str = self.result_name
        executor: ModelExecutor = self.executor
        executor.reset()
        logger.info('training model...')
        self._notify('train_start', result_name)
        with time('trained'):
            res = executor.train(result_name)
        self._notify('train_end', result_name)
        return res

    def test(self) -> ModelResult:
        """Load the model from disk and test it."""
        if self.debuged:
            raise ModelError('Testing is not allowed in debug mode')
        result_name: str = self.result_name
        executor: ModelExecutor = self.executor
        executor.load()
        logger.info('testing model...')
        self._notify('test_start', result_name)
        with time('tested'):
            res = executor.test(result_name)
        if self.writer is not None:
            res.write(writer=self.writer)
        self._notify('test_end', result_name)
        return res

    def cross_validate(self, n_repeats: int = 1) -> ModelResult:
        """Perform a cross validation of using the additional resources
        configured in the :obj:`executor`.

        :param n_repeats: the number of train/test iterations per fold

        """
        result_name: str = self.result_name
        executor: ModelExecutor = self.executor
        executor.reset()
        logger.info('cross validating model...')
        self._notify('train_start', result_name)
        with time('cross validated'):
            res = executor.cross_validate(n_repeats)
        self._notify('train_end', result_name)
        return res

    def train_production(self) -> ModelResult:
        """Like :meth:`train` but uses the test dataset in addition to the
        training dataset to train the model.

        """
        result_name: str = self.result_name
        executor: ModelExecutor = self.executor
        executor.reset()
        if self.writer is not None:
            executor.write(writer=self.writer)
        logger.info('training production model...')
        self._notify('train_production_start', result_name)
        with time('trained'):
            res = executor.train_production(result_name)
        self._notify('train_production_end', result_name)
        return res

    def predict(self, datas: Iterable[Any]) -> Any:
        """Make ad-hoc predictions on batches without labels, and return the
        results.

        :param datas: the data predict on, each as a separate element as a data
                      point in a batch

        """
        executor: ModelExecutor = self.executor
        ms: ModelSettings = self.model_settings
        if ms.prediction_mapper_name is None:
            raise ModelError(
                f'The model settings ({ms.name}) is not configured to create ' +
                "prediction batches: no set 'prediction_mapper'")
        # the batch stash is used to create batches from 'datas'
        pm: PredictionMapper = self.config_factory.new_instance(
            ms.prediction_mapper_name, datas, self.batch_stash)
        self._notify('predict_start')
        try:
            # create ad-hoc batches from 'datas'
            batches: List[Batch] = pm.batches
            if not executor.model_exists:
                executor.load()
            logger.info('predicting...')
            with time('predicted'):
                # use the model to predict ad-hoc batches
                res: ModelResult = executor.predict(batches)
            eres: EpochResult = res.results[0]
            ret: Any = pm.map_results(eres)
        finally:
            self._notify('predict_end')
            pm.deallocate()
        return ret

    def stop_training(self):
        """Early stop training if the model is currently training.  This invokes
        the :meth:`.TrainManager.stop`, communicates to the training process to
        stop on the next check.

        :return: ``True`` if the application is configured to early stop and
                 the signal has not already been given

        """
        self._notify('stop_training')
        return self.executor.train_manager.stop()

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
                raise ModelError(f'No results foundin directory: {rm.path}')
        return res

    def write_result(self, depth: int = 0, writer: TextIOBase = sys.stdout,
                     include_settings: bool = False,
                     include_converged: bool = False,
                     include_config: bool = False):
        """Load the last set of results from the file system and print them out.
        The result to print is taken from :obj:`last_result`

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
        """Plot results and optionally save and show them.  If this is called in
        a Jupyter notebook, the plot will be rendered in a cell.

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
        grapher.plot_loss([result])
        if save:
            grapher.save()
        if show:
            grapher.show()
        return result

    def _create_predictions_factory(self, **kwargs) -> \
            PredictionsDataFrameFactory:
        cls = self.predictions_dataframe_factory_class
        if isinstance(cls, str):
            return self.config_factory.instance(cls, **kwargs)
        else:
            return cls(**kwargs)

    def get_predictions_factory(self, column_names: List[str] = None,
                                transform: Callable[[DataPoint], tuple] = None,
                                batch_limit: int = sys.maxsize,
                                name: str = None) -> \
            PredictionsDataFrameFactory:
        """Generate a predictions factory from the test data set.

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

        :param name: the name/ID (name of the file sans extension in the
                     results directory) of the previously archived saved
                     results to fetch or ``None`` to get the last result

        :param kwargs: additional keyword arguments used to create the factory

        """
        rm: ModelResultManager = self.result_manager
        res: ModelResult
        if name is None:
            res = self.last_result
            key: str = rm.get_last_key(False)
        else:
            res = rm.results_stash[name].model_result
            key: str = name
        if res is None:
            raise ModelError(f'No test results found: {name}')
        if not res.test.contains_results:
            raise ModelError('No test results found')
        path: Path = rm.key_to_path(key)
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'reading predictions from {path}')
        return self._create_predictions_factory(
            source=path,
            result=res,
            name=name,
            stash=self.batch_stash,
            column_names=column_names,
            data_point_transform=transform,
            batch_limit=batch_limit)

    def get_result_analyzer(self, key: str = None,
                            cache_previous_results: bool = False) \
            -> ResultAnalyzer:
        """Return a results analyzer for comparing in flight training progress.

        """
        rm: ModelResultManager = self.result_manager
        if key is None:
            key = rm.get_last_key()
        return ResultAnalyzer(self.executor, key, cache_previous_results)

    def _create_result_reporter(self, **kwargs) -> ModelResultReporter:
        cls = self.model_result_reporter_class
        if isinstance(cls, str):
            return self.config_factory.instance(cls, **kwargs)
        else:
            return cls(**kwargs)

    def get_result_reporter(self, cross_fold: bool = False,
                            include_validation: bool = True) -> \
            ModelResultReporter:
        """Return an instance of the summary report generator."""
        rm: ModelResultManager
        if cross_fold:
            rm = self.cross_fold_result_manager
        else:
            rm = self.result_manager
        return self._create_result_reporter(
            result_manager=rm,
            include_validation=include_validation)

    @property
    def class_explorer(self) -> FacadeClassExplorer:
        return self._create_facade_explorer()

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
        """When debuging the model, configure the logging system for output.
        The correct loggers need to be set to debug mode to print the model
        debugging information such as matrix shapes.

        """
        for name in ['zensols.deeplearn.model',
                     __name__]:
            logging.getLogger(name).setLevel(logging.DEBUG)

    def _configure_cli_logging(self, info_loggers: List[str],
                               debug_loggers: List[str]):
        info_loggers.extend([
            # multi-process (i.e. batch creation)
            'zensols.multi.stash',
            'zensols.deeplearn.batch.multi',
            # validation/training loss messages
            'zensols.deeplearn.model.executor.status',
            __name__])
        if not self.progress_bar:
            info_loggers.extend([
                # load messages
                'zensols.deeplearn.batch.stash',
                # save results messages
                'zensols.deeplearn.result',
                # validation/training loss messages
                'zensols.deeplearn.model.executor.progress',
                # model save/load
                'zensols.deeplearn.model.manager',
                # early stop messages
                'zensols.deeplearn.model.trainmng',
                # performance metrics formatting
                'zensols.deeplearn.model.format',
                # model packaging
                'zensols.deeplearn.model.pack',
                # model save messages
                'zensols.deeplearn.result.manager',
                # observer module API messages
                'zensols.deeplearn.observer.status',
                # CLI interface
                'zensols.deeplearn.cli'])

    @staticmethod
    def configure_default_cli_logging(log_level: int = logging.WARNING):
        """Configure the logging system with the defaults.

        """
        fmt = '%(asctime)s[%(levelname)s]%(name)s: %(message)s'
        logging.basicConfig(format=fmt, level=log_level)

    def configure_cli_logging(self, log_level: int = None):
        """"Configure command line (or Python REPL) debugging.  Each facade can
        turn on name spaces that make sense as useful information output for
        long running training/testing iterations.

        This calls "meth:`_configure_cli_logging` to collect the names of
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
        """Configures logging and other configuration related to a Jupyter
        notebook.  This is just like :py:meth:`configure_cli_logging`, but
        adjusts logging for what is conducive for reporting in Jupyter cells.

        ;param log_level: the default logging level for the logging system

        :param progress_bar_cols: the number of columns to use for the progress
                                  bar

        """
        self.configure_cli_logging(log_level)
        for name in [
                # turn off loading messages
                'zensols.deeplearn.batch.stash',
                # turn off model save messages
                'zensols.deeplearn.result.manager']:
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
