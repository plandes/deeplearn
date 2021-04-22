"""Command line entry point to the application using the Zensols application
CLI.

"""
__author__ = 'plandes'

from typing import Dict, Any, List
from dataclasses import dataclass, field
import logging
import itertools as it
from enum import Enum, auto
from zensols.persist import dealloc
from zensols.config import Configurable, ImportConfigFactory
from zensols.cli import Application, ApplicationFactory, Invokable
from zensols.deeplearn import DeepLearnError
from zensols.deeplearn.model import ModelFacade
from zensols.deeplearn.batch import Batch

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
class FacadeApplication(object):
    config: Configurable = field()
    """The config used to create facade instances."""

    facade_name: str = field(default='facade')
    """The client facade."""

    config_factory_args: Dict[str, Any] = field(default_factory=dict)
    """The arguments given to the :class:`~zensols.config.ImportConfigFactory`,
    which could be useful for reloading all classes while debugingg.

    """
    def _create_facade(self) -> ModelFacade:
        """Create a new instance of the facade.

        """
        # we must create a new (non-shared) instance of the facade since it
        # will get deallcated after complete.
        cf = ImportConfigFactory(self.config, **self.config_factory_args)
        facade = cf.instance(self.facade_name)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'created facade: {facade}')
        return facade


@dataclass
class FacadeInfoApplication(FacadeApplication):
    CLI_META = {'mnemonic_overrides': {'print_information': 'info'},
                'option_includes': {'info_item'},
                'option_overrides': {'info_item': {'long_name': 'item',
                                                   'short_name': 'i'}}}

    def print_information(self, info_item: InfoItem = InfoItem.default):
        """Output facade data set, vectorizer and other configuration information.

        """
        # see :class:`.FacadeApplicationFactory'
        if not hasattr(self, '_no_op'):
            defs = 'executor metadata settings model config'.split()
            params = {f'include_{k}': False for k in defs}
            with dealloc(self._create_facade()) as facade:
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

    def debug(self):
        """Debug the model.

        """
        with dealloc(self._create_facade()) as facade:
            facade.debug()


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
                 'train_production': 'trainprod'}}

    use_progress_bar: bool = field(default=False)
    """Display the progress bar."""

    def _create_facade(self) -> ModelFacade:
        facade = super()._create_facade()
        facade.progress_bar = self.use_progress_bar
        if not self.use_progress_bar:
            facade.configure_cli_logging()
        return facade

    def clear_batches(self):
        """Clear all batch data."""
        with dealloc(self._create_facade()) as facade:
            logger.info('clearing batches')
            facade.batch_stash.clear()

    def batch(self, limit: int = 1):
        """Create batches (if not created already) and print statistics on the dataset.

        :param limit: the number of batches to print out

        """
        with dealloc(self._create_facade()) as facade:
            facade.executor.dataset_stash.write()
            batch: Batch
            for batch in it.islice(facade.batch_stash.values(), limit):
                batch.write()

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
        """Train, test the model, then dump the results with a graph.

        """
        with dealloc(self._create_facade()) as facade:
            facade.train()
            facade.test()
            facade.persist_result()

    def train_production(self):
        """Train, test the model on train and test datasets, then dump the results with
        a graph.

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
            facade.stop_training()


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
    def create_facade(self, args: List[str] = None) -> ModelFacade:
        """Create the facade tied to the application without invoking the command line.

        """
        create_args = ['info']
        if args is not None:
            create_args.extend(args)
        app: Application = self.create(create_args)
        inv: Invokable = app.invoke_but_second_pass()[1]
        fac_app: FacadeApplication = inv.instance
        return fac_app._create_facade()
