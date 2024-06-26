"""A utility class to summarize all results in a directory.

"""
__author__ = 'Paul Landes'

from typing import Dict, Tuple, ClassVar
from dataclasses import dataclass, field
from pathlib import Path
import logging
import pandas as pd
from zensols.util.time import time
from zensols.datdesc import DataFrameDescriber
from zensols.deeplearn import DatasetSplitType
from . import (
    ModelResult, EpochResult, DatasetResult, ModelResultManager, ArchivedResult,
    Metrics, PredictionsDataFrameFactory,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelResultReporter(object):
    """Summarize all results in a directory from the output of model execution
    from :class:`~zensols.deeplearn.model.ModelExectuor`.

    The class iterates through the pickled binary output files from the run and
    summarizes in a Pandas dataframe, which is handy for reporting in papers.

    """
    METRIC_DESCRIPTIONS: ClassVar[Dict[str, str]] = \
        PredictionsDataFrameFactory.METRIC_DESCRIPTIONS
    """Dictionary of performance metrics column names to human readable
    descriptions.

    """
    result_manager: ModelResultManager = field()
    """Contains the results to report on--and specifically the path to directory
    where the results were persisted.

    """
    include_validation: bool = field(default=True)
    """Whether or not to include validation performance metrics."""

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return the summarized results (see class docs).

        :return: the Pandas dataframe of the results

        """
        rows = []
        cols = 'name file start train_duration converged features'.split()
        cols.extend('wF1t wPt wRt mF1t mPt mRt MF1t MPt MRt acct'.split())
        if self.include_validation:
            cols.extend('wF1v wPv wRv mF1v mPv mRv MF1v MPv MRv accv'.split())
        cols.extend('train_occurs validation_occurs test_occurs'.split())
        dpt_key = 'n_total_data_points'
        arch_res: ArchivedResult
        for fname, arch_res in self.result_manager.results_stash.items():
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'reading results from {fname}')
            res: ModelResult = arch_res.model_result
            train: DatasetResult = res.dataset_result.get(
                DatasetSplitType.train)
            validate: DatasetResult = res.dataset_result.get(
                DatasetSplitType.validation)
            test: DatasetResult = res.dataset_result.get(DatasetSplitType.test)
            if train is not None:
                dur = train.end_time - train.start_time
                hours, remainder = divmod(dur.seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                dur = f'{hours:02}:{minutes:02}:{seconds:02}'
            if validate is not None:
                conv_epoch: int = validate.statistics['n_epoch_converged']
                ver: EpochResult = validate.converged_epoch
            else:
                conv_epoch = None
                ver: EpochResult = None
            if test is not None:
                vm: Metrics = ver.metrics
                tm: Metrics = None
                if not test.is_ended:
                    # production models will not have test results
                    logger.warning(
                        f'no results found for {arch_res}--not reporting')
                else:
                    tm = test.metrics
                features = ', '.join(res.decoded_attributes)
                row = [res.name, fname, train.start_time, dur,
                       conv_epoch, features]
                if tm is None:
                    row.extend([float('nan')] * 10)
                else:
                    row.extend([
                        tm.weighted.f1, tm.weighted.precision, tm.weighted.recall,
                        tm.micro.f1, tm.micro.precision, tm.micro.recall,
                        tm.macro.f1, tm.macro.precision, tm.macro.recall,
                        tm.accuracy])
                if self.include_validation:
                    row.extend([
                        vm.weighted.f1, vm.weighted.precision,
                        vm.weighted.recall,
                        vm.micro.f1, vm.micro.precision, vm.micro.recall,
                        vm.macro.f1, vm.macro.precision, vm.macro.recall,
                        vm.accuracy])
                row.extend([
                    train.statistics[dpt_key], validate.statistics[dpt_key],
                    test.statistics[dpt_key]])
                rows.append(row)
                if logger.isEnabledFor(logging.INFO):
                    logger.info('result calculation complete for ' +
                                f'{res.name} ({fname})')
        return pd.DataFrame(rows, columns=cols)

    @property
    def dataframe_describer(self) -> DataFrameDescriber:
        """Get a dataframe describer of metrics (see :obj:`metrics_dataframe`).

        """
        df: pd.DataFrame = self.dataframe
        meta: Tuple[Tuple[str, str], ...] = \
            tuple(map(lambda c: (c, self.METRIC_DESCRIPTIONS[c]), df.columns))
        name: str = (self.result_manager.name.capitalize() +
                     ' Summarized Model Results')
        return DataFrameDescriber(
            name='Summarized Model Results',
            df=df,
            desc=name,
            meta=meta)

    def dump(self, path: Path) -> pd.DataFrame:
        """Create the summarized results and write them to the file system.

        """
        with time(f'wrote results summary: {path}'):
            df: pd.DataFrame = self.dataframe
            df.to_csv(path)
            return df
