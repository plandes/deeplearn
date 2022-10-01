"""Command line entry point to the application using the application CLI.

"""
__author__ = 'plandes'

from typing import Dict, Any, List, Type, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import gc
import sys
import itertools as it
import copy as cp
from io import TextIOBase
from pathlib import Path
from zensols.persist import dealloc, Deallocatable, PersistedWork, persisted
from zensols.config import (
    Writable, Configurable, ImportConfigFactory, DictionaryConfig
)
from zensols.cli import (
    ApplicationError, Application, ApplicationFactory,
    ActionCliManager, Invokable, CliHarness,
)
from zensols.dataset import (
    SplitStashContainer, StratifiedStashSplitKeyContainer
)
from zensols.deeplearn import DeepLearnError, TorchConfig, ModelSettings
from zensols.deeplearn.model import ModelFacade, ModelError, ModelPacker
from zensols.deeplearn.result import (
    ModelResultManager, ModelResultReporter, PredictionsDataFrameFactory,
    ModelResultComparer
)

logger = logging.getLogger(__name__)


class InfoItem(Enum):
    """Indicates what information to dump in
    :meth:`.FacadeInfoApplication.print_information`.

    """
    meta = auto()
    param = auto()
    model = auto()
    config = auto()
    batch = auto()


class ClearType(Enum):
    """Indicates what type of data to delete (clear).

    """
    none = auto()
    batch = auto()
    source = auto()


@dataclass
class FacadeApplication(Deallocatable):
    """Base class for applications that use :class:`.ModelFacade`.

    """
    CLI_META = {'mnemonic_excludes': {'get_cached_facade', 'create_facade',
                                      'deallocate', 'clear_cached_facade'},
                'option_overrides': {'model_path': {'long_name': 'model',
                                                    'short_name': None}}}
    """Tell the command line app API to igonore subclass and client specific use
    case methods.

    """
    config: Configurable = field()
    """The config used to create facade instances."""

    facade_name: str = field(default='facade')
    """The client facade."""

    # simply copy this field and documentation to the implementation class to
    # add model path location (for those subclasses that don't have the
    # ``CLASS_INSPECTOR`` class level attribute set (see
    # :obj:`~zensols.util.introspect.inspect.ClassInspector.INSPECT_META`);
    # this can also be set as a parameter such as with
    # :methd:`.FacadeModelApplication.test`
    model_path: Path = field(default=None)
    """The path to the model or use the last trained model if not provided.

    """
    config_factory_args: Dict[str, Any] = field(default_factory=dict)
    """The arguments given to the :class:`~zensols.config.ImportConfigFactory`,
    which could be useful for reloading all classes while debugingg.

    """
    config_overwrites: Configurable = field(default=None)
    """A configurable that clobbers any configuration in :obj:`config` for those
    sections/options set.

    """
    cache_global_facade: bool = field(default=True)
    """Whether to globally cache the facade returned by
    :meth:`get_cached_facade`.

    """
    def __post_init__(self):
        self.dealloc_resources = []
        self._cached_facade = PersistedWork(
            '_cached_facade', self,
            cache_global=self.cache_global_facade)

    def _enable_cli_logging(self, facade: ModelFacade = None):
        if facade is None:
            with dealloc(self.create_facade()) as facade:
                self._enable_cli_logging(facade)
        else:
            facade.progress_bar = False
            facade.configure_cli_logging()

    def _get_model_path(self) -> Path:
        """Return the path to the model, which defaults to :obj:`model_path`."""
        return self.model_path

    def create_facade(self) -> ModelFacade:
        """Create a new instance of the facade."""
        # we must create a new (non-shared) instance of the facade since it
        # will get deallcated after complete.
        config = self.config
        model_path = self._get_model_path()
        if self.config_overwrites is not None:
            config = cp.deepcopy(config)
            config.merge(self.config_overwrites)
        if model_path is None:
            cf = ImportConfigFactory(config, **self.config_factory_args)
            facade: ModelFacade = cf.instance(self.facade_name)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'created facade: {facade}')
            self.dealloc_resources.extend((cf, facade))
        else:
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'loading model from {model_path}')
            with dealloc(ImportConfigFactory(
                    config, **self.config_factory_args)) as cf:
                cls: Type[ModelFacade] = cf.get_class(self.facade_name)
            facade: ModelFacade = cls.load_from_path(model_path)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'created facade: {type(facade)} ' +
                             f'from path: {model_path}')
            self.dealloc_resources.append(facade)
        return facade

    @persisted('_cached_facade')
    def get_cached_facade(self) -> ModelFacade:
        """Return a created facade that is cached in this application instance.

        """
        return self.create_facade()

    def clear_cached_facade(self):
        """Clear any cached facade this application instance.

        """
        if self._cached_facade.is_set():
            self._cached_facade().deallocate()
        self._cached_facade.clear()

    def deallocate(self):
        super().deallocate()
        self._try_deallocate(self.dealloc_resources, recursive=True)
        self._cached_facade.deallocate()


@dataclass
class FacadeInfoApplication(FacadeApplication):
    """Contains methods that provide information about the model via the facade.

    """
    CLI_META = ActionCliManager.combine_meta(
        FacadeApplication,
        {'mnemonic_overrides': {'print_information': 'info'},
         'option_overrides': {'info_item': {'long_name': 'item',
                                            'short_name': 'i'},
                              'debug_value': {'long_name': 'execlevel',
                                              'short_name': None}}})

    def print_information(self, info_item: InfoItem = None):
        """Output facade data set, vectorizer and other configuration information.

        :param info_item: what to print

        """
        # see :class:`.FacadeApplicationFactory'
        def write_batch():
            for batch in it.islice(facade.batch_stash.values(), 2):
                batch.write()

        if not hasattr(self, '_no_op'):
            with dealloc(self.create_facade()) as facade:
                print(f'{facade.model_settings.model_name}:')
                fn_map = \
                    {None: facade.write,
                     InfoItem.meta: facade.batch_metadata.write,
                     InfoItem.param: facade.executor.write_settings,
                     InfoItem.model: facade.executor.write_model,
                     InfoItem.config: facade.config.write,
                     InfoItem.batch: write_batch}
                fn = fn_map.get(info_item)
                if fn is None:
                    raise DeepLearnError(f'No such info item: {info_item}')
                fn()

    def debug(self, debug_value: int = None):
        """Debug the model.

        :param debug_value: the executor debugging level

        """
        debug_value = True if debug_value is None else debug_value
        with dealloc(self.create_facade()) as facade:
            facade.debug(debug_value)


@dataclass
class FacadeResultApplication(FacadeApplication):
    """Contains methods that dump previous results.

    """
    CLI_META = ActionCliManager.combine_meta(
        FacadeApplication,
        {'mnemonic_overrides': {'result_summary': 'summary',
                                'result_ids': 'resids',
                                'metrics': 'results',
                                'majority_label_metrics': 'majlab',
                                'compare_results': 'cmpres'},
         'option_overrides': {'include_validation': {'long_name': 'validation',
                                                     'short_name': None},
                              'out_file': {'long_name': 'outfile',
                                           'short_name': 'o'}}})

    def result_summary(self, out_file: Path = None,
                       include_validation: bool = False):
        """Create a summary of all archived results.

        :param out_file: the output path

        :param include_validation: whether to include validation results

        """
        if out_file is None:
            out_file = Path('result-summary.csv')
        with dealloc(self.create_facade()) as facade:
            rm: ModelResultManager = facade.result_manager
            self._enable_cli_logging(facade)
            reporter = ModelResultReporter(rm)
            reporter.include_validation = include_validation
            return reporter.dump(out_file)

    def metrics(self, sort: str = 'wF1', res_id: str = None,
                out_file: Path = None):
        """Write a spreadhseet of label performance metrics for a previously trained
        and tested model.

        :param sort_col: the column to sort results

        :param res_id: the result ID or use the last if not given

        :param out_file: the output path

        """
        if out_file is None:
            out_file = Path('metrics.csv')
        with dealloc(self.create_facade()) as facade:
            df = facade.get_predictions_factory(name=res_id).metrics_dataframe
            df = df.sort_values(sort, ascending=False).reset_index(drop=True)
            df.to_csv(out_file)
            self._enable_cli_logging(facade)
            logger.info(f'wrote: {out_file}')

    def result_ids(self):
        """Show all archived result IDs."""
        with dealloc(self.create_facade()) as facade:
            rm: ModelResultManager = facade.result_manager
            print('\n'.join(rm.results_stash.keys()))

    def result(self, res_id: str = None):
        """Show the last results.

        :param res_id: the result ID or use the last if not given

        """
        with dealloc(self.create_facade()) as facade:
            df_fac: PredictionsDataFrameFactory = \
                facade.get_predictions_factory(name=res_id)
            df_fac.result.write()

    def majority_label_metrics(self, res_id: str = None):
        """Show majority label metrics of the test dataset using a previous result set.

        :param res_id: the result ID or use the last if not given

        """
        with dealloc(self.create_facade()) as facade:
            pred_factory: PredictionsDataFrameFactory = \
                facade.get_predictions_factory(name=res_id)
            pred_factory.majority_label_metrics.write()

    def compare_results(self, res_id_a: str, res_id_b: str):
        """Compare two previous archived result sets.

        :param res_id_a: the first result ID to compare

        :param res_id_b: the second result ID to compare

        """
        with dealloc(self.create_facade()) as facade:
            rm: ModelResultComparer = facade.result_manager
            diff = ModelResultComparer(rm, res_id_a, res_id_b)
            diff.write()


@dataclass
class FacadePackageApplication(FacadeApplication):
    """Contains methods that package models.

    """
    CLI_META = ActionCliManager.combine_meta(
        FacadeApplication,
        {'mnemonic_overrides': {'pack_model': 'pack'},
         'option_overrides': {'output_model_dir': {'long_name': 'modeldir'}},
         'option_excludes': {'packer'}})

    packer: ModelPacker = field(default=None)
    """The model packer used to create the model distributions from this app."""

    def pack_model(self, res_id: str = None,
                   output_model_dir: Path = Path('.')):
        """Package a distribution model.

        :param res_id: the result ID or use the last if not given

        :param output_model_dir: the directory where the packaged model is
                                 written

        """
        if res_id is None:
            with dealloc(self.create_facade()) as facade:
                self._enable_cli_logging(facade)
                res_id: str = facade.result_manager.get_last_id()
        self._enable_cli_logging()
        self.packer.pack(res_id, output_model_dir)


@dataclass
class FacadeBatchApplication(FacadeApplication):
    """Test, train and validate models.

    """
    CLI_META = ActionCliManager.combine_meta(
        FacadeApplication,
        {'option_overrides':
         {'clear_type': {'long_name': 'ctype',
                         'short_name': None},
          'clear': {'short_name': None},
          'split': {'short_name': None},
          'limit': {'short_name': None}},
         'mnemonic_overrides':
         {'batch': {'option_includes': {'limit', 'clear_type', 'split'}}}})

    def _write_batch_splits(self, facade: ModelFacade):
        scont: SplitStashContainer = facade.batch_stash.split_stash_container
        if hasattr(scont, 'split_container') and \
           isinstance(scont.split_container, StratifiedStashSplitKeyContainer):
            stash: StratifiedStashSplitKeyContainer = scont.split_container
            stash.stratified_write = True
            stash.write()

    def batch(self, limit: int = None, clear_type: ClearType = ClearType.none,
              split: bool = False):
        """Create batches if not already, print statistics on the dataset.

        :param clear_type: what to delete to force recreate

        :param limit: the number of batches to create

        :param split: also write the stratified splits if available

        """
        with dealloc(self.create_facade()) as facade:
            self._enable_cli_logging(facade)
            if clear_type == ClearType.batch:
                logger.info('clearing batches')
                facade.batch_stash.clear()
            elif clear_type == ClearType.source:
                facade.batch_stash.clear_all()
                facade.batch_stash.clear()
            facade.dataset_stash.write()
            if split:
                self._write_batch_splits(facade)


@dataclass
class FacadeModelApplication(FacadeApplication):
    """Test, train and validate models.

    """
    CLI_META = ActionCliManager.combine_meta(
        FacadeApplication,
        {'option_overrides': {'use_progress_bar': {'long_name': 'progress',
                                                   'short_name': 'p'}},
         'mnemonic_overrides': {'train_production': 'trainprod',
                                'early_stop': {'option_includes': {},
                                               'name': 'stop'}}})

    use_progress_bar: bool = field(default=False)
    """Display the progress bar."""

    def create_facade(self) -> ModelFacade:
        """Create a new instance of the facade."""
        facade = super().create_facade()
        facade.progress_bar = self.use_progress_bar
        facade.configure_cli_logging()
        return facade

    def train(self):
        """Train the model and dump the results, including a graph of the
        train/validation loss.

        """
        with dealloc(self.create_facade()) as facade:
            facade.train()
            facade.persist_result()

    def test(self, model_path: Path = None):
        """Test an existing model the model and dump the results of the test.

        :param model_path: the path to the model or use the last trained model
                           if not provided

        """
        self.model_path = self._get_model_path()
        with dealloc(self.create_facade()) as facade:
            facade.test()

    def train_test(self):
        """Train, test the model, then dump the results with a graph.

        """
        with dealloc(self.create_facade()) as facade:
            facade.train()
            facade.test()
            facade.persist_result()

    def train_production(self):
        """Train, test the model on train and test datasets, then dump the results with
        a graph.

        """
        with dealloc(self.create_facade()) as facade:
            facade.train_production()
            facade.test()
            facade.persist_result()

    def early_stop(self):
        """Stops the execution of training the model.

        """
        with dealloc(self.create_facade()) as facade:
            facade.stop_training()


class FacadePredictApplication(FacadeApplication):
    """An applicaiton that provides prediction funtionality.

    """
    CLI_META = ActionCliManager.combine_meta(
        FacadeApplication, {'mnemonic_overrides':
                            {'predictions': {'name': 'preds'}}})

    def predictions(self, res_id: str = None, out_file: Path = None):
        """Write predictions to a CSV file.

        :param res_id: the result ID or use the last if not given

        :param out_file: the output path

        """
        with dealloc(self.create_facade()) as facade:
            if out_file is None:
                model_settings: ModelSettings = facade.executor.model_settings
                model_name = model_settings.normal_model_name
                out_file = Path(f'{model_name}.csv')
            try:
                df = facade.get_predictions(name=res_id)
            except ModelError as e:
                raise ApplicationError(
                    'Could not predict, probably need to train a model ' +
                    f'first: {e}') from e
            df.to_csv(out_file)
            self._enable_cli_logging(facade)
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'wrote predictions: {out_file}')


@dataclass
class FacadeApplicationFactory(ApplicationFactory):
    """This is a utility class that creates instances of
    :class:`.FacadeApplication`.  It's only needed if you need to create a
    facade without wanting invoke the command line attached to the
    applications.

    It does this by only invoking the first pass applications so all the
    correct initialization happens before returning factory artifacts.

    There mst be a :obj:`.FacadeApplication.facade_name` entry in the
    configuration tied to an instance of :class:`.FacadeApplication`.

    :see: :meth:`create_facade`

    """
    def create_facade(self, args: List[str] = None,
                      app_args: Dict[str, Any] = None) -> ModelFacade:
        """Create the facade tied to the application without invoking the command line.

        :param args: the (would be) command line arguments used to create the
                     application

        :param app_args: the arguments to set on the the facade application
                         after it is created and before it creates the facade

        """
        create_args = ['info']
        if args is not None:
            create_args.extend(args)
        app: Application = self.create(create_args)
        inv: Invokable = app.invoke_but_second_pass()[1]
        fac_app: FacadeApplication = inv.instance
        if app_args is not None:
            for k, v in app_args.items():
                setattr(fac_app, k, v)
        return fac_app.create_facade()


@dataclass
class FacadeApplicationManager(Writable):
    """A very high level client interface making it easy to configure and run
    models from an interactive environment such as a Python REPL or a Jupyter
    notebook (see :class:`.JupyterManager`)

    """
    cli_harness: CliHarness = field()
    """The CLI harness used to create the facade application."""

    cli_args_fn: List[str] = field(default=lambda: [])
    """Creates the arguments used to create the facade from the application
    factory.

    """
    reset_torch: bool = field(default=True)
    """Reset random state for consistency for each new created facade."""

    allocation_tracking: Union[bool, str] = field(default=False)
    """Whether or not to track resource/memory leaks.  If set to ``stack``, the
    stack traces of the unallocated objects will be printed.  If set to
    ``counts`` only the counts will be printed.  If set to ``True`` only the
    unallocated objects without the stack will be printed.

    """
    logger_name: str = field(default='notebook')
    """The name of the logger to use for logging in the notebook itself."""

    default_logging_level: str = field(default='WARNING')
    """If set, then initialize the logging system using this as the default logging
    level.  This is the upper case logging name such as ``WARNING``.

    """
    progress_bar_cols: int = field(default=120)
    """The number of columns to use for the progress bar."""

    config_overwrites: Dict[str, Dict[str, str]] = field(default_factory=dict)
    """Clobbers any configuration set by :meth:`config` for those sections/options
    set.

    """
    def __post_init__(self):
        if self.allocation_tracking:
            Deallocatable.ALLOCATION_TRACKING = True
        if self.logger_name is not None:
            self.logger = logging.getLogger(self.logger_name)
        else:
            self.logger = logger
        self._facade = None

    def _create_facade(self, args: List[str] = None,
                       app_args: Dict[str, Any] = None) -> ModelFacade:
        """Create the facade tied to the application without invoking the command line.

        :param args: the (would be) command line arguments used to create the
                     application

        :param app_args: the arguments to set on the the facade application
                         after it is created and before it creates the facade

        """
        create_args = ['info']
        if args is not None:
            create_args.extend(args)
        fac_app: FacadeApplication = self.cli_harness.get_instance(create_args)
        assert isinstance(fac_app, FacadeApplication)
        if app_args is not None:
            for k, v in app_args.items():
                setattr(fac_app, k, v)
        return fac_app.create_facade()

    def cleanup(self, include_cuda: bool = True, quiet: bool = False):
        """Report memory leaks, run the Python garbage collector and optionally empty
        the CUDA cache.

        :param include_cuda: if ``True`` clear the GPU cache

        :param quiet: do not report unallocated objects, regardless of the
                      setting of :obj:`allocation_tracking`

        """
        if self.allocation_tracking and not quiet:
            include_stack, only_counts = False, False
            if self.allocation_tracking == 'stack':
                include_stack, only_counts = True, False
            elif self.allocation_tracking == 'counts':
                include_stack, only_counts = False, True
            include_stack = (self.allocation_tracking == 'stack')
            Deallocatable._print_undeallocated(include_stack, only_counts)
        self.deallocate()
        Deallocatable._deallocate_all()
        gc.collect()
        if include_cuda:
            # free up memory in the GPU
            TorchConfig.empty_cache()

    def deallocate(self):
        """Deallocate all resources in the CLI factory if it exists."""
        if self._facade is not None:
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info('deallocating old factory')
            self._facade.deallocate()
            self._facade = None

    def config(self, section: str, **kwargs):
        """Add overwriting configuration used when creating the facade.

        :param section: the section to be overwritten (or added)

        :param kwargs: the key/value pairs used as the section data to
                       overwrite

        :see: :meth:`create_facade`

        """
        if section not in self.config_overwrites:
            self.config_overwrites[section] = {}
        self.config_overwrites[section].update(kwargs)

    def clear(self):
        """Clear all post create configuration set with :meth:`config`."""
        self.config_overwrites.clear()

    def create_facade(self, *args, **kwargs) -> ModelFacade:
        """Create and return a facade.  This deallocates and cleans up state from any
        previous facade creation as a side effect.

        :param args: given to the :obj:`cli_args_fn` function to create
                     arguments passed to the CLI

        """
        if len(self.config_overwrites) > 0:
            dconf = DictionaryConfig(self.config_overwrites)
            app_args = {'config_overwrites': dconf}
        else:
            app_args = None
        self.deallocate()
        # reclaim memory running GC and GPU cache clear
        self.cleanup()
        try:
            # reset random state for consistency of each new test
            if self.reset_torch:
                TorchConfig.init()
            # create a factory that instantiates Python objects
            cli_args_fn = self.cli_args_fn(*args, **kwargs)
            # create the facade used for this instance
            self._facade: ModelFacade = self._create_facade(
                cli_args_fn, app_args)
            return self._facade
        except Exception as e:
            try:
                # recover the best we can
                self.cleanup(quiet=True)
                self._facade = None
            except Exception:
                pass
            raise DeepLearnError(f'Could not create facade: {e}') from e

    @property
    def facade(self) -> ModelFacade:
        """The current facade for this notebook instance.

        :return: the existing facade, or that created by :meth:`create_facade`
                 if it doesn't already exist

        """
        if self._facade is None:
            self.create_facade()
        self._facade.writer = None
        return self._facade

    def run(self, display_results: bool = True):
        """Train, test and optionally show results.

        :param display_results: if ``True``, write and plot the results

        """
        try:
            facade = self.facade
            facade.train()
            facade.test()
            if display_results:
                facade.write_result()
                facade.plot_result()
        except Exception as e:
            try:
                facade = None
                # recover the best we can
                self.cleanup(quiet=True)
            except Exception:
                pass
            raise DeepLearnError('Could not run the model') from e

    def show_leaks(self, output: str = 'counts', fail: bool = True):
        """Show all resources/memory leaks in the current facade.  First, this
        deallocates the facade, then prints any lingering objects using
        :class:`~zensols.persist.Deallocatable`.

        **Important**: :obj:`allocation_tracking` must be set to ``True`` for
        this to work.

        :param output: one of ``stack``, ``counts``, or ``tensors``

        :param fail: if ``True``, raise an exception if there are any
                     unallocated references found

        """
        if self._facade is None:
            raise DeepLearnError('No facade created yet')
        if self.allocation_tracking:
            self._facade.deallocate()
            if output == 'counts':
                Deallocatable._print_undeallocated(only_counts=True, fail=fail)
            elif output == 'stack':
                Deallocatable._print_undeallocated(include_stack=True, fail=fail)
            elif output == 'tensors':
                TorchConfig.write_in_memory_tensors()
            else:
                raise DeepLearnError(f'Unknown output type: {output}')
            self._facade = None

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              include_model=False, include_metadata=False,
              include_settings=False, **kwargs):
        self.facade.write(
            depth, writer,
            include_model=include_model,
            include_metadata=include_metadata,
            include_settings=include_settings,
            **kwargs)


@dataclass
class JupyterManager(FacadeApplicationManager):
    """A facade application manager that provides additional convenience
    functionality.

    """
    reduce_logging: bool = field(default=False)
    """Whether to disable most information logging so the progress bar is more
    prevalent.

    """
    browser_width: int = field(default=95)
    """The width of the browser windows as a percentage."""

    def __post_init__(self):
        super().__post_init__()
        if self.browser_width is not None:
            self.set_browser_width(self.browser_width)

    @staticmethod
    def set_browser_width(width: int = 95):
        """Use the entire width of the browser to create more real estate.

        :param width: the width as a percent (``[0, 100]``) to use as the width
                      in the notebook

        """
        from IPython.core.display import display, HTML
        html = f'<style>.container {{ width:{width}% !important; }}</style>'
        display(HTML(html))

    def _init_jupyter(self):
        """Initialize the a Jupyter notebook by configuring the logging system and
        setting the progress bar.

        """
        if self.reduce_logging:
            logging.getLogger('zensols.deeplearn.model.executor.status').\
                setLevel(logging.WARNING)
        else:
            log_level = None
            if self.default_logging_level is not None:
                log_level = getattr(logging, self.default_logging_level)
            # set console based logging
            self.facade.configure_jupyter(
                log_level=log_level,
                progress_bar_cols=self.progress_bar_cols)

    def create_facade(self, *args, **kwargs) -> ModelFacade:
        facade = super().create_facade(*args, **kwargs)
        # initialize jupyter
        self._init_jupyter()
        return facade
