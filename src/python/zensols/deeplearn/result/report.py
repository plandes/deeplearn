
"""A utility class to summarize all results in a directory.

"""
__author__ = 'Paul Landes'

from typing import Dict, Tuple, List, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging
import parse
from collections import OrderedDict
import pandas as pd
import numpy as np
import scipy
from zensols.util.time import time
from zensols.persist import FileTextUtil, Stash
from zensols.datdesc import DataFrameDescriber, DataDescriber
from zensols.deeplearn import DatasetSplitType
from . import (
    ModelResult, EpochResult, DatasetResult, ModelResultManager, ArchivedResult,
    Metrics, PredictionsDataFrameFactory
)

logger = logging.getLogger(__name__)


@dataclass
class ModelResultReporter(object):
    """Summarize all results in a directory from the output of model execution
    from :class:`~zensols.deeplearn.model.ModelExectuor`.

    The class iterates through the pickled binary output files from the run and
    summarizes in a Pandas dataframe, which is handy for reporting in papers.

    """
    result_manager: ModelResultManager = field()
    """Contains the results to report on--and specifically the path to directory
    where the results were persisted.

    """
    include_validation: bool = field(default=True)
    """Whether or not to include validation performance metrics."""

    def _add_rows(self, fname: str, arch_res: ArchivedResult) -> List[Any]:
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'reading results from {fname}')
        dpt_key: str = 'n_total_data_points'
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
                    f'no test results found for {arch_res}--not reporting')
            else:
                tm = test.metrics
            features = ', '.join(res.decoded_attributes)
            row: List[Any] = [res.name, fname, train.start_time, dur,
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
            if logger.isEnabledFor(logging.INFO):
                logger.info('result calculation complete for ' +
                            f'{res.name} ({fname})')
            return row

    def _get_archive_results(self) -> Tuple[Tuple[str, ArchivedResult], ...]:
        stash: Stash = self.result_manager.results_stash
        return sorted(stash.items(), key=lambda t: t[0])

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return the summarized results (see class docs).

        :return: the Pandas dataframe of the results

        """
        rows: List[List[Any]] = []
        cols = 'name resid start train_duration converged features'.split()
        cols.extend(PredictionsDataFrameFactory.TEST_METRIC_COLUMNS)
        if self.include_validation:
            cols.extend(PredictionsDataFrameFactory.VALIDATION_METRIC_COLUMNS)
        cols.extend('train_occurs validation_occurs test_occurs'.split())
        arch_res: ArchivedResult
        for fname, arch_res in self._get_archive_results():
            rows.append(self._add_rows(fname, arch_res))
        return pd.DataFrame(rows, columns=cols)

    def _create_data_frame_describer(self, df: pd.DataFrame,
                                     desc: str = 'Summary Model Results',
                                     metric_metadata: Dict[str, str] = None) \
            -> DataFrameDescriber:
        mdesc: Dict[str, str] = dict(
            PredictionsDataFrameFactory.METRIC_DESCRIPTIONS)
        if metric_metadata is not None:
            mdesc.update(metric_metadata)
        meta: Tuple[Tuple[str, str], ...] = tuple(map(
            lambda c: (c, mdesc[c]), df.columns))
        return DataFrameDescriber(
            name=FileTextUtil.normalize_text(desc),
            df=df,
            desc=f'{self.result_manager.name.capitalize()} {desc}',
            meta=meta)

    @property
    def dataframe_describer(self) -> DataFrameDescriber:
        """Get a dataframe describer of metrics (see :obj:`dataframe`)."""
        return self._create_data_frame_describer(df=self.dataframe)

    def _cross_validate_summary(self) -> DataFrameDescriber:
        from zensols.dataset import StratifiedCrossFoldSplitKeyContainer

        def map_name(name: str, axis: int) -> str:
            p: parse.Result = parse.parse(fold_format, name)
            return pd.Series((p['fold_ix'], p['iter_ix']))

        fold_format: str = StratifiedCrossFoldSplitKeyContainer.FOLD_FORMAT
        test_cols: List[str, ...] = \
            list(PredictionsDataFrameFactory.TEST_METRIC_COLUMNS)
        val_cols: List[str, ...] = \
            list(PredictionsDataFrameFactory.VALIDATION_METRIC_COLUMNS)
        test_cols.append('test_occurs')
        val_cols.append('validation_occurs')
        df: pd.DataFrame = self.dataframe.drop(columns=test_cols).\
            rename(columns=dict(zip(val_cols, test_cols)))
        fold_cols: List[str] = ['fold', 'iter']
        cols: List[str] = df.columns.to_list()
        df[fold_cols] = df['name'].apply(map_name, axis=1)
        df = df[fold_cols + cols]
        df = df.drop(columns=['name'])
        df = df.sort_values(fold_cols)
        dfd: DataFrameDescriber = self._create_data_frame_describer(
            df=df,
            desc='Cross Validation Results',
            metric_metadata=(('fold', 'fold number'),
                             ('iter', 'sub-fold iteration')))
        return dfd

    def _mean_conf_interval(self, data: np.ndarray, confidence: float = 0.95) \
            -> Tuple[int, float, float, float]:
        """Compute the mean confidence interval using a Student's t critical
        value on a sample.

        :param data: the sample

        :param confidence: the interval to use for confidence

        :return: the mean, lower and upper confidence interval

        """
        # ensure correct computation
        data = data.astype(float)
        # sample size
        n: int = len(data)
        # mu
        m: float = np.mean(data)
        # sigma
        se: float = scipy.stats.sem(data)
        # margin of error
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
        return n, m, m - h, m + h

    def _cross_validate_stats(self, dfd_res: DataFrameDescriber) -> \
            DataFrameDescriber:
        df = dfd_res.df[list(PredictionsDataFrameFactory.TEST_METRIC_COLUMNS)]
        rows: List[pd.Series] = []
        index_meta: Dict[str, str] = OrderedDict()
        desc = f'{len(df)}-Fold Cross Validation Statistics'
        stat: str
        for stat in 'mean min max std'.split():
            row: pd.Series = getattr(df, stat)()
            row.name = stat
            rows.append(row.to_frame().T)
            index_meta[stat] = f'the {stat} of the performance metric'
        cis: List[Tuple[float, float]] = []
        col_data: pd.Series
        for col_data in map(lambda t: t[1], df.items()):
            n, m, ci_min, ci_max = self._mean_conf_interval(col_data.to_numpy())
            cis.append((ci_min, ci_max))
        rows.append(pd.Series(data=map(lambda t: t[0], cis),
                              index=df.columns,
                              name='conf_low').to_frame().T)
        rows.append(pd.Series(data=map(lambda t: t[1], cis),
                              index=df.columns,
                              name='conf_high').to_frame().T)
        dfs: pd.DataFrame = pd.concat(rows)
        for name in 'low high'.split():
            index_meta.update({
                f'conf_{name}':
                f'the {name} bound of the confidence interval'})
        dfs.insert(0, 'stat', list(index_meta.keys()))
        dfs.index.name = 'description'
        meta: pd.DataFrame = dfd_res.meta[dfd_res.meta.index.isin(df.columns)]
        meta = pd.concat((
            pd.DataFrame([{'description': 'aggregate statistic'}],
                         index=['stat']),
            meta))
        return DataFrameDescriber(
            name='cross-validation-stats',
            df=dfs,
            desc=f'{self.result_manager.name.capitalize()} {desc}',
            meta=meta,
            index_meta=index_meta)

    @property
    def cross_validate_describer(self) -> DataDescriber:
        """Create a data describer with the results of a cross-validation.

        """
        dfd_sum: DataFrameDescriber = self._cross_validate_summary()
        dfd_stat: DataFrameDescriber = self._cross_validate_stats(dfd_sum)
        return DataDescriber(
            name='summary-model',
            describers=(dfd_sum, dfd_stat))

    def dump(self, path: Path) -> pd.DataFrame:
        """Create the summarized results and write them to the file system.

        """
        with time(f'wrote results summary: {path}'):
            df: pd.DataFrame = self.dataframe
            df.to_csv(path)
            return df
