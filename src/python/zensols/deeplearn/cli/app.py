"""Command line entry point to the application using the Zensols application
CLI.

"""
__author__ = 'plandes'

from typing import Dict, Any
from dataclasses import dataclass, field
import logging
from zensols.persist import dealloc
from zensols.config import Configurable, ImportConfigFactory
from zensols.deeplearn.model import ModelFacade

logger = logging.getLogger(__name__)


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
                'option_includes': set()}

    def print_information(self):
        """Output facade data set, vectorizer and other configuration information.

        """
        with dealloc(self._create_facade()) as facade:
            facade.write(include_settings=True)

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
                 'batch': {'option_includes': set()},
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

    def batch(self):
        """Create batches (if not created already) and print statistics on the dataset.

        """
        with dealloc(self._create_facade()) as facade:
            facade.executor.dataset_stash.write()

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
            facade.executor.lifecycle_manager.stop()
