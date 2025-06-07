"""Cross-fold validation application classes.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .result.report import ModelResultReporter
from dataclasses import dataclass, field
from enum import Enum, auto
from zensols.persist import Stash, dealloc
from pathlib import Path
from .model import ModelFacade
from .cli import (
    ActionCliManager, ClearType, BatchReport, Format, FacadeApplication,
    FacadeResultApplication, FacadeBatchApplication, FacadeModelApplication,
)


class CrossValidationReportType(Enum):
    results = auto()
    stats = auto()


@dataclass
class _FacadeCrossValidateApplication(FacadeApplication):
    def _get_batch_stash(self, facade: ModelFacade) -> Stash:
        return facade.cross_fold_batch_stash

    def _get_dataset_stash(self, facade: ModelFacade) -> Stash:
        return facade.cross_fold_batch_stash

    def _get_batch_metrics(self, facade: ModelFacade) -> Stash:
        return facade.get_cross_fold_batch_metrics()

    def _get_result_reporter(self, facade: ModelFacade) -> ModelResultReporter:
        return facade.get_result_reporter(cross_fold=True)


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

        :param report: the type of report to generate

        """
        super().batch(limit, clear_type, report)


@dataclass
class FacadeCrossValidateModelApplication(FacadeModelApplication):
    """Test, train and validate models.

    """
    CLI_META = ActionCliManager.combine_meta(
        FacadeModelApplication,
        {'option_excludes': {'CLASS_INSPECTOR'},
         'option_overrides': {
             'n_repeats': {'long_name': 'repeats',
                           'short_name': None}},
         'mnemonic_overrides': {'cross_validate': 'cvalrun'}})

    use_progress_bar: bool = field(default=False)
    """Display the progress bar."""

    def cross_validate(self, n_repeats: int = 1):
        """Cross validate the model and dump the results.

        :param n_repeats: the number of train/test iterations per fold

        """
        with dealloc(self.create_facade()) as facade:
            facade.cross_validate(n_repeats)


@dataclass
class FacadeCrossValidateResultApplication(
        _FacadeCrossValidateApplication, FacadeResultApplication):
    """Cross validation results.

    """
    CLI_META = ActionCliManager.combine_meta(
        FacadeResultApplication,
        {'mnemonic_overrides': {'cross_validate': 'cvalressum'}})

    def cross_validate(self, report_type: CrossValidationReportType,
                       out_file: Path = None, out_format: Format = None):
        """Create a summary of all archived results.

        :param out_file: the output path or ``-`` for standard out

        :param out_format: the output format

        """
        from zensols.datdesc import DataDescriber, DataFrameDescriber
        from zensols.deeplearn.result import ModelResultReporter

        out_format = Format.csv if out_format is None else out_format
        with dealloc(self.create_facade()) as facade:
            reporter: ModelResultReporter = self._get_result_reporter(facade)
            reporter.include_validation = True
            dd: DataDescriber = reporter.cross_validate_describer
            name: str = f'cross-validation-{report_type.name}'
            dfd: DataFrameDescriber = dd[name]
            dd.describers = (dfd,)
            self._process_data_describer(out_file, out_format, facade, dd)
            return dd
