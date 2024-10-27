"""Cross-fold validation application classes.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .result.manager import ModelResultManager
from dataclasses import dataclass, field
from zensols.persist import Stash, dealloc
from pathlib import Path
from .model import ModelFacade
from .cli import (
    ActionCliManager, ClearType, BatchReport, FacadeApplication,
    FacadeResultApplication, FacadeBatchApplication, FacadeModelApplication,
)


@dataclass
class _FacadeCrossValidateApplication(FacadeApplication):
    def _get_batch_stash(self, facade: ModelFacade) -> Stash:
        return facade.cross_fold_batch_stash

    def _get_dataset_stash(self, facade: ModelFacade) -> Stash:
        return facade.cross_fold_batch_stash

    def _get_batch_metrics(self, facade: ModelFacade) -> Stash:
        return facade.get_cross_fold_batch_metrics()

    def _get_result_manager(self, facade: ModelFacade) -> ModelResultManager:
        return facade.cross_fold_result_manager


@dataclass
class FacadeCrossValidateBatchApplication(
        _FacadeCrossValidateApplication, FacadeBatchApplication):
    """Create and analyze batches.

    """
    CLI_META = ActionCliManager.combine_meta(
        FacadeBatchApplication,
        {'mnemonic_overrides': {'cross_validate_batch':
                                {'name': 'cvalbatch'}}})

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
         'mnemonic_overrides': {'cross_validate': 'cvalrun'}})

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


@dataclass
class FacadeCrossValidateResultApplication(
        _FacadeCrossValidateApplication, FacadeResultApplication):
    """Cross validation results.

    """
    CLI_META = ActionCliManager.combine_meta(
        FacadeResultApplication,
        {'mnemonic_overrides': {'cross_validate_all_runs': 'cvalresall'}})

    def cross_validate_all_runs(self, out_file: Path = None):
        """Create a summary of all archived results.

        :param out_file: the output path or ``-`` for standard out

        """
        # TODO: rename val to test cols
        super().all_runs(out_file, True)
