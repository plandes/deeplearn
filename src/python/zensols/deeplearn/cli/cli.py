"""Command line entry point to the application.

"""
__author__ = 'plandes'

from typing import Type, Dict, Tuple, List, Any
from dataclasses import dataclass, field, InitVar
import logging
import sys
from io import TextIOBase
from zensols.persist import dealloc
from zensols.config import Configurable, Writable, StringConfig
from zensols.cli import OneConfPerActionOptionsCliEnv
from zensols.deeplearn import TorchConfig
from zensols.deeplearn.model import ModelFacade

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentVariable(object):
    """Describes an script like environment variable to be extracted and formatted
    from a configuration.

    :param config_name: the name of the configuration option in the
                        configuration

    :param section: the configuration section where the variable is found

    :param name: the name of the option in the configuration

    """
    config_name: str = field(default=None)
    section: str = field(default='default')
    name: str = field(default=None)

    def __post_init__(self):
        if self.name is None:
            self.name = self.config_name


@dataclass
class EnvironmentFormatter(Writable):
    """Formats data from a :class:`.Configurable` in to a script/make etc type
    string format used as (usually build) environent.

    :param config: contains the configuration to format.

    :param env_vars: the variables to output

    """
    config: Configurable
    env_vars: Tuple[EnvironmentVariable] = \
        field(default_factory=lambda: [EnvironmentVariable()])

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        var: EnvironmentVariable
        for var in self.env_vars:
            if var.name is None:
                ops = self.config.get_options(opt_keys=var.section)
                for k, v in ops.items():
                    self._write_line(f'{k.upper()}={v}', depth, writer)
            else:
                cval = self.config.get_option_object(
                    var.config_name, section=var.section)
                line = f'{var.name.upper()}={str(cval)}'
                self._write_line(line, depth, writer)


@dataclass
class FacadeCli(object):
    """A glue class called by :class:`.FacadeCommandLine` to invoke operations
    (train, test, etc) on a :class:`.ModelFacade`.


    This class willl typically have the following methods overriden:
      * :py:meth:`_create_environment_formatter`
      * :py:meth:`_get_facade_class`

    """
    config: Configurable
    overrides: InitVar[str] = field(default=None)
    use_progress_bar: bool = field(default=False)

    def __post_init__(self, overrides: str):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'overrides: {overrides} ({type(overrides)}), ' +
                         f'config: {self.config}')
        # done as early as possible before any framework initialization
        TorchConfig.set_random_seed()
        if overrides is not None:
            sc = StringConfig(overrides)
            self.config.merge(sc)

    def _create_environment_formatter(self) -> EnvironmentFormatter:
        """Return a new environment formatter.

        """
        return None

    def _get_facade_class(self) -> Type[ModelFacade]:
        """Return the :class:`.ModelFacade` (or subclass) used to invoke operations
        called by the command line.

        """
        return ModelFacade

    def _create_facade(self) -> ModelFacade:
        """Create a new instance of the facade.

        """
        facade_cls = self._get_facade_class()
        facade = facade_cls(self.config)
        facade.progress_bar = self.use_progress_bar
        if not self.use_progress_bar:
            facade.configure_cli_logging()
        return facade

    def print_environment(self):
        """Print the environment of the facade in a key/variable like script format
        usually used by builds.

        """
        ef = self._create_environment_formatter()
        if ef is not None:
            ef.write()

    def print_information(self):
        """Output facade data set, vectorizer and other configuration information.

        """
        with dealloc(self._create_facade()) as facade:
            facade.write(include_settings=True)

    def batch(self):
        """Create batches (if not created already) and print statistics on the dataset.

        """
        with dealloc(self._create_facade()) as facade:
            facade.executor.dataset_stash.write()

    def debug(self):
        """Train and test the model.

        """
        with dealloc(self._create_facade()) as facade:
            facade.debug()

    def train(self):
        """Train the model and dump the results, including a graph of the
        train/validation loss.

        """
        with dealloc(self._create_facade()) as facade:
            facade.train()
            facade.persist_result()

    def test(self):
        """Test an existing model the model and dump the results of the test.

        """
        with dealloc(self._create_facade()) as facade:
            facade.test()

    def train_test(self):
        """Train and test the model.  Afterward, dump the results, including a graph of
        the train/validation loss.

        """
        with dealloc(self._create_facade()) as facade:
            facade.train()
            facade.test()
            facade.persist_result()

    def train_production(self):
        """Train and test the model on the training and test datasets.  Afterward, dump
        the results, including a graph of the train/validation loss.  This is
        used for a "production" model that is used for some purpose other than
        evaluation.

        """
        with dealloc(self._create_facade()) as facade:
            facade.train_production()
            facade.test()
            facade.persist_result()

    def early_stop(self):
        """Stops the execution of training the model.

        Currently this is done by creating a file the executor monitors.

        """
        with dealloc(self._create_facade()) as facade:
            facade.executor.lifecycle_manager.stop()


class FacadeCommandLine(OneConfPerActionOptionsCliEnv):
    """The command line entry point for facade interaction, such as training and
    testing the model.  Typically this class is overridden to just call the
    :py:meth:`__init__` method.

    """
    def __init__(self, cli_class: Type[FacadeCli], *args, **kwargs):
        cnf = self._get_arg_config(cli_class)
        super().__init__(cnf, *args, **kwargs, no_os_environ=True)
        if 'pkg_dist' in kwargs:
            self.pkg_dist = kwargs['pkg_dist']

    @property
    def overrides_op(self):
        return ['-o', '--overrides', False,
                {'dest': 'overrides',
                 'metavar': 'STRING',#'section.name=value[,section.name=value,...]',
                 'help': 'comma separated config overrides'}]

    @property
    def progress_bar_op(self):
        return ['-p', '--progressbar', False,
                {'dest': 'use_progress_bar',
                 'action': 'store_true',
                 'default': False,
                 'help': 'if provided, display progress bar'}]

    def _get_arg_config(self, cli_class: Type[FacadeCli]) -> Dict[str, Any]:
        return {'executors':
                [{'name': 'facade',
                  'executor': lambda params: cli_class(**params),
                  'actions': self._get_actions()}],
                'config_option': {'name': 'config',
                                  'expect': True,
                                  'opt': ['-c', '--config', False,
                                          {'dest': 'config',
                                           'metavar': 'FILE',
                                           'help': 'configuration file'}]},
                'whine': 0}

    def _get_actions(self) -> List[Dict[str, str]]:
        return [{'name': 'env',
                 'meth': 'print_environment',
                 'doc': 'action help explains how to do it'},
                {'name': 'info',
                 'meth': 'print_information',
                 'doc': 'print information about the model',
                 'opts': []},
                {'name': 'batch',
                 'meth': 'batch',
                 'doc': 'create batches (if not already) and print stats',
                 'opts': [self.overrides_op]},
                {'name': 'debug',
                 'meth': 'debug',
                 'doc': 'print debugging information about the model',
                 'opts': [self.overrides_op]},
                {'name': 'train',
                 'meth': 'train',
                 'doc': 'train the model',
                 'opts': [self.overrides_op, self.progress_bar_op]},
                {'name': 'test',
                 'meth': 'test',
                 'doc': 'test the model',
                 'opts': [self.overrides_op, self.progress_bar_op]},
                {'name': 'traintest',
                 'meth': 'train_test',
                 'doc': 'train and test the model',
                 'opts': [self.overrides_op, self.progress_bar_op]},
                {'name': 'trainprod',
                 'meth': 'train_production',
                 'doc': 'train the model on all data sets',
                 'opts': [self.overrides_op, self.progress_bar_op]},
                {'name': 'stop',
                 'meth': 'early_stop',
                 'doc': 'stop the training model execution',
                 'opts': []}]

    def _config_log_level(self, fmt, levelno):
        fmt = '%(asctime)s[%(levelname)s]:%(name)s %(message)s'
        super()._config_log_level(fmt, levelno)
        logging.getLogger(self.pkg_dist).setLevel(logging.INFO)
