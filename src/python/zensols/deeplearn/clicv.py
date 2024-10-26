"""Cross-fold validation application classes.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field
from zensols.persist import Stash, dealloc
from .model import ModelFacade
from .cli import (
    ActionCliManager, ClearType, BatchReport,
    FacadeBatchApplication, FacadeModelApplication
)


@dataclass
class FacadeCrossValidateBatchApplication(FacadeBatchApplication):
    """Create and analyze batches.

    """
    CLI_META = ActionCliManager.combine_meta(
        FacadeBatchApplication,
        {'mnemonic_overrides': {'cross_validate_batch':
                                {'name': 'cvalbatch'}}})

    def _get_batch_stash(self, facade: ModelFacade) -> Stash:
        return facade.cross_fold_batch_stash

    def _get_dataset_stash(self, facade: ModelFacade) -> Stash:
        return facade.cross_fold_batch_stash

    def _get_batch_metrics(self, facade: ModelFacade) -> Stash:
        return facade.get_cross_fold_batch_metrics()

    def cross_validate_batch(self, limit: int = None,
                             clear_type: ClearType = ClearType.none,
                             report: BatchReport = BatchReport.none):
        """Create cross-validation batches if not already, print statistics on
        the dataset.

        :param clear_type: what to delete to force recreate

        :param limit: the number of batches to create

        :param report: also report label statistics

        """
        super().batch(limit, clear_type, report)


@dataclass
class FacadeCrossValidateModelApplication(FacadeModelApplication):
    """Test, train and validate models.

    """
    CLI_META = ActionCliManager.combine_meta(
        FacadeModelApplication,
        {'option_excludes': {'CLASS_INSPECTOR'},
         'mnemonic_overrides': {'cross_validate': 'cval'}})

    use_progress_bar: bool = field(default=False)
    """Display the progress bar."""

    def cross_validate(self, result_name: str = None):
        """Cross validate the model and dump the results.

        :param result_name: a descriptor used in the results

        """
        with dealloc(self.create_facade()) as facade:
            if result_name is not None:
                facade.result_name = result_name
            facade.cross_validate()
