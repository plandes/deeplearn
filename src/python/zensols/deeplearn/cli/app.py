"""Command line entry point to the application using the Zensols application
CLI.

"""
__author__ = 'plandes'

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

    def _create_facade(self) -> ModelFacade:
        """Create a new instance of the facade.

        """
        # we must create a new (non-shared) instance of the facade since it
        # will get deallcated after complete.
        cf = ImportConfigFactory(self.config)
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
                                   'name': 'rmbatches'}}}

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

    def train(self):
        """Train the model and dump the results, including a graph of the
        train/validation loss.

        """
        with dealloc(self._create_facade()) as facade:
            facade.train()
            facade.persist_result()
