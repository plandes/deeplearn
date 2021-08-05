"""Command line entry point to the application using the Zensols application
CLI.

"""
__author__ = 'plandes'

from typing import Dict, Any, List, Type, Union
from dataclasses import dataclass, field, InitVar
from enum import Enum, auto
import logging
import gc
import itertools as it
import copy as cp
from pathlib import Path
from zensols.persist import dealloc, Deallocatable, PersistedWork, persisted
from zensols.config import Configurable, ImportConfigFactory, DictionaryConfig
from zensols.cli import Application, ApplicationFactory, Invokable
from zensols.deeplearn import DeepLearnError, TorchConfig
from zensols.deeplearn.model import ModelFacade
from zensols.deeplearn.batch import Batch
from zensols.deeplearn.result import (
    ModelResultManager, ModelResultReporter, PredictionsDataFrameFactory
)

logger = logging.getLogger(__name__)


class InfoItem(Enum):
    """Indicates what information to dump in
    :meth:`.FacadeInfoApplication.print_information`.

    """
    default = auto()
    executor = auto()
    metadata = auto()
    settings = auto()
    model = auto()
    config = auto()
    batch = auto()


@dataclass
class FacadeApplication(Deallocatable):
    """Base class for applications that use :class:`.ModelFacade`.

    """
    CLI_META = {'mnemonic_excludes': {'get_cached_facade', 'create_facade',
                                      'deallocate', 'clear_cached_facade'}}
    """Tell the command line app API to igonore subclass and client specific use
    case methods.

    """

    config: Configurable = field()
    """The config used to create facade instances."""

    facade_name: str = field(default='facade')
    """The client facade."""

    facade_path: Path = field(default=None)
    """The path to the model, or if ``None``, use the last trained model, which is
    directory specified by :obj:`~zensols.deeplearn.ModelSettings.path`.

    """

    config_factory_args: Dict[str, Any] = field(default_factory=dict)
    """The arguments given to the :class:`~zensols.config.ImportConfigFactory`,
    which could be useful for reloading all classes while debugingg.

    """

    config_overwrites: Configurable = field(default=None)
    """A configurable that clobbers any configuration in :obj:`config` for those
    sections/options set.

    """

    def __post_init__(self):
        self.dealloc_resources = []
        self._cached_facade = PersistedWork('_cached_facade', self, True)

    def create_facade(self, path: Path = None) -> ModelFacade:
        """Create a new instance of the facade.

        """
        # we must create a new (non-shared) instance of the facade since it
        # will get deallcated after complete.
        config = self.config
        if self.config_overwrites is not None:
            config = cp.deepcopy(config)
            config.merge(self.config_overwrites)
        if self.facade_path is None:
            cf = ImportConfigFactory(config, **self.config_factory_args)
            facade: ModelFacade = cf.instance(self.facade_name)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'created facade: {facade}')
            self.dealloc_resources.extend((cf, facade))
        else:
            with dealloc(ImportConfigFactory(
                    config, **self.config_factory_args)) as cf:
                cls: Type[ModelFacade] = cf.get_class(self.facade_name)
            facade: ModelFacade = cls.load_from_path(self.facade_path)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'created facade: {facade} from path: {path}')
            self.dealloc_resources.append(facade)
        return facade

    @persisted('_cached_facade')
    def get_cached_facade(self, path: Path = None) -> ModelFacade:
        """Return a created facade that is cached in this application instance.

        """
        return self.create_facade(path)

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
    CLI_META = {'mnemonic_overrides': {'print_information': 'info',
                                       'result_summary': 'resum'},
                'option_overrides': {'info_item': {'long_name': 'item',
                                                   'short_name': 'i'},
                                     'debug_value': {'long_name': 'execlevel',
                                                     'short_name': None}}}

    def print_information(self, info_item: InfoItem = InfoItem.default):
        """Output facade data set, vectorizer and other configuration information.

        """
        # see :class:`.FacadeApplicationFactory'
        if not hasattr(self, '_no_op'):
            defs = 'executor metadata settings model config'.split()
            params = {f'include_{k}': False for k in defs}
            with dealloc(self.create_facade()) as facade:
                key = f'include_{info_item.name}'
                if key in params:
                    params[key] = True
                    facade.write(**params)
                elif info_item == InfoItem.default:
                    facade.write()
                elif info_item == InfoItem.batch:
                    for batch in it.islice(facade.batch_stash.values(), 1):
                        batch.write()
                else:
                    raise DeepLearnError(f'No such info item: {info_item}')

    def debug(self, debug_value: int = None):
        """Debug the model.

        :param debug_value: the executor debugging level

        """
        debug_value = True if debug_value is None else debug_value
        with dealloc(self.create_facade()) as facade:
            facade.debug(debug_value)

    def result_summary(self, out_file: Path = None,
                       result_dir: Path = None):
        """Create a result summary from a directory.

        :param out_file: the output path

        :param result_dir: the directory to find the results

        """
        with dealloc(self.create_facade()) as facade:
            rm: ModelResultManager = facade.result_manager
            facade.progress_bar = False
            facade.configure_cli_logging()
            if out_file is None:
                out_file = Path(f'{rm.prefix}.csv')
            if result_dir is not None:
                rm = cp.copy(rm)
                rm.path = result_dir
            reporter = ModelResultReporter(rm)
            reporter.dump(out_file)


@dataclass
class FacadeModelApplication(FacadeApplication):
    """Test, train and validate models.

    """
    CLI_META = {'option_overrides':
                {'use_progress_bar': {'long_name': 'progress',
                                      'short_name': 'p'}},
                'mnemonic_overrides':
                {'clear_batches': {'option_includes': set(),
                                   'name': 'rmbatch'},
                 'batch': {'option_includes': {'limit'}},
                 'train_production': 'trainprod',
                 'early_stop': {'option_includes': set(),
                                'name': 'stop'},
                 'predictions': {'option_excludes': 'use_progress_bar',
                                 'name': 'preds'}}}

    use_progress_bar: bool = field(default=False)
    """Display the progress bar."""

    def create_facade(self) -> ModelFacade:
        facade = super().create_facade()
        facade.progress_bar = self.use_progress_bar
        facade.configure_cli_logging()
        return facade

    def clear_batches(self):
        """Clear all batch data."""
        with dealloc(self.create_facade()) as facade:
            logger.info('clearing batches')
            facade.batch_stash.clear()

    def batch(self, limit: int = 1):
        """Create batches (if not created already) and print statistics on the dataset.

        :param limit: the number of batches to print out

        """
        with dealloc(self.create_facade()) as facade:
            facade.executor.dataset_stash.write()
            batch: Batch
            for batch in it.islice(facade.batch_stash.values(), limit):
                batch.write()

    def train(self):
        """Train the model and dump the results, including a graph of the
        train/validation loss.

        """
        with dealloc(self.create_facade()) as facade:
            facade.train()
            facade.persist_result()

    def test(self):
        """Test an existing model the model and dump the results of the test.

        """
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

    def predictions(self, res_id: str = None, out_file: Path = None):
        """Write predictions to a CSV file.

        :param res_id: the result ID

        :param out_file: the output path

        """
        with dealloc(self.create_facade()) as facade:
            df_fac: PredictionsDataFrameFactory = \
                facade.get_predictions_factory(name=res_id)
            if out_file is None:
                fname = ModelResultManager.to_file_name(df_fac.name)
                out_file = Path(f'{fname}.csv')
            logger.info(f'reading from {df_fac.source}')
            df_fac.dataframe.to_csv(out_file)
            logger.info(f'wrote: {out_file}')

    def result(self, res_id: str = None):
        """Show the last results.

        :param res_id: the result ID

        """
        with dealloc(self.create_facade()) as facade:
            df_fac: PredictionsDataFrameFactory = \
                facade.get_predictions_factory(name=res_id)
            print(f'File: {df_fac.source}')
            df_fac.result.write()


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
class FacadeApplicationManager(object):
    """A very high level client interface making it easy to configure and run
    models from an interactive environment such as a Python REPL or a Jupyter
    notebook (see :class:`.JupyterManager`)

    """
    cli_class: Type[FacadeApplicationFactory] = field(default=None)
    """The class the application factory used to create the facade."""

    factory_args: Dict[str, Any] = field(default_factory=dict)
    """The arguments given to the initializer of :obj`cli_class` on create."""

    cli_args_fn: List[str] = field(default_factory=lambda: [])
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

    default_logging_level: InitVar[str] = field(default='WARNING')
    """If set, then initialize the logging system using this as the default logging
    level.  This is the upper case logging name such as ``WARNING``.

    """

    progress_bar_cols: InitVar[int] = field(default=120)
    """The number of columns to use for the progress bar."""

    browser_width: InitVar[int] = field(default=95)
    """The width of the browser windows as a percentage."""

    config_overwrites: Dict[str, Dict[str, str]] = field(default_factory=dict)
    """Clobbers any configuration in :obj:`config` for those sections/options set.

    """
    def __post_init__(self, default_logging_level: str, progress_bar_cols: int,
                      browser_width: int):
        if self.allocation_tracking:
            Deallocatable.ALLOCATION_TRACKING = True
        if browser_width is not None:
            self.set_browser_width(browser_width)
        if self.logger_name is not None:
            self.logger = logging.getLogger(self.logger_name)
        else:
            self.logger = logger

    def _create_factory(self) -> FacadeApplicationFactory:
        """Create a command line application factory."""
        if self.cli_class is None:
            raise DeepLearnError(
                'Either create with a cli_class attribute or override ' +
                'the _create_factory method')
        return self.cli_class(**self.factory_args)

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
        if hasattr(self, 'cli_factory'):
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info('deallocating old factory')
            self.cli_factory.deallocate()
            del self.cli_factory

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
        """Clear all post create configuration.

        :see: :meth:`config`

        """
        self.config_overwrites.clear()

    def create_facade(self, *args) -> ModelFacade:
        """Create and return a facade with columns that fit a notebook.

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
            # create a command line application factory
            self.cli_factory: FacadeApplicationFactory = self._create_factory()
            # reset random state for consistency of each new test
            if self.reset_torch:
                TorchConfig.init()
            # create a factoty that instantiates Python objects
            cli_args_fn = self.cli_args_fn(*args)
            # create the facade used for this instance
            self._facade: ModelFacade = self.cli_factory.create_facade(
                cli_args_fn, app_args)
            return self._facade
        except Exception as e:
            try:
                # recover the best we can
                self.cleanup(quiet=True)
                self._facade = None
            except Exception:
                pass
            raise DeepLearnError('Could not create facade') from e

    @property
    def facade(self) -> ModelFacade:
        """The current facade for this notebook instance.

        :return: the existing facade, or that created by :meth:`create_facade`
                 if it doesn't already exist

        """
        if not hasattr(self, '_facade'):
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
        if not hasattr(self, 'cli_factory'):
            raise DeepLearnError('No CLI factory yet created')
        if self.allocation_tracking:
            self.cli_factory.deallocate()
            if output == 'counts':
                Deallocatable._print_undeallocated(only_counts=True, fail=fail)
            elif output == 'stack':
                Deallocatable._print_undeallocated(include_stack=True, fail=fail)
            elif output == 'tensors':
                TorchConfig.write_in_memory_tensors()
            else:
                raise DeepLearnError(f'Unknown output type: {output}')
            del self.cli_factory


@dataclass
class JupyterManager(FacadeApplicationManager):
    """A facade application manager that provides additional convenience
    functionality.

    """
    @staticmethod
    def set_browser_width(width: int):
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
        log_level = None
        if self.default_logging_level is not None:
            log_level = getattr(logging, self.default_logging_level)
        # set console based logging
        self._facade.configure_jupyter(
            log_level=log_level,
            progress_bar_cols=self.progress_bar_cols)

    def create_facade(self, *args) -> ModelFacade:
        facade = super().create_facade(*args)
        # initialize jupyter
        self._init_jupyter()
        return facade
