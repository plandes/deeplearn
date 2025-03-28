"""A utility class to summarize all results in a directory.

"""
__author__ = 'Paul Landes'

from typing import Dict, Tuple, List, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging
import parse
from collections import OrderedDict
import pandas as pd
import numpy as np
import math
import scipy
from zensols.util.time import time
from zensols.persist import persisted, FileTextUtil, Stash
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

    @persisted('_archive_results')
    def _get_archive_results(self) -> Tuple[Tuple[str, ArchivedResult], ...]:
        stash: Stash = self.result_manager.results_stash
        return sorted(stash.items(), key=lambda t: t[0])

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
            row: List[Any] = [res.name, fname, train.start_time, train.end_time,
                              test.start_time, test.end_time,
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

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return the summarized results (see class docs).

        :return: the Pandas dataframe of the results

        """
        rows: List[List[Any]] = []
        res_stash: Stash = self.result_manager.results_stash
        n_res: int = len(res_stash)
        cols = 'name resid train_start train_end test_start test_end converged features'.split()
        cols.extend(PredictionsDataFrameFactory.TEST_METRIC_COLUMNS)
        if self.include_validation:
            cols.extend(PredictionsDataFrameFactory.VALIDATION_METRIC_COLUMNS)
        cols.extend('train_occurs validation_occurs test_occurs'.split())
        if n_res == 0:
            logger.warning(f'no results found in: {self.result_manager}')
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
            # TODO: iter_ix -> repeat_ix
            return pd.Series((int(p['fold_ix']), int(p['iter_ix'])))

        fold_format: str = StratifiedCrossFoldSplitKeyContainer.FOLD_FORMAT
        test_cols: List[str, ...] = \
            list(PredictionsDataFrameFactory.TEST_METRIC_COLUMNS)
        val_cols: List[str, ...] = \
            list(PredictionsDataFrameFactory.VALIDATION_METRIC_COLUMNS)
        test_cols.append('test_occurs')
        val_cols.append('validation_occurs')
        df: pd.DataFrame = self.dataframe.drop(columns=test_cols).\
            rename(columns=dict(zip(val_cols, test_cols)))
        fold_cols: List[str] = ['fold', 'repeat']
        cols: List[str] = df.columns.to_list()
        df[fold_cols] = df['name'].apply(map_name, axis=1)
        df = df[fold_cols + cols]
        df = df.drop(columns=['name'])
        df = df.sort_values(fold_cols)
        dfd: DataFrameDescriber = self._create_data_frame_describer(
            df=df,
            desc='Cross Validation Results',
            metric_metadata=(('fold', 'fold number'),
                             ('repeat', 'sub-fold repeat')))
        return dfd

    def _calc_t_ci(self, data: np.ndarray) -> Tuple[float, float]:
        """Compute the mean 95% confidence interval assuming a normal
        distribution of scores treating the scores as a mean distribution.

        :param data: the sample

        :param confidence: the interval to use for confidence

        :return: the confidence interval

        :see: :obj:`cross_validate_describer` for calculation reference

        :see: `Definition: <https://en.wikipedia.org/wiki/Confidence_interval>`_

        :see: `Mean distribution: <https://sebastianraschka.com/blog/2022/confidence-intervals-for-ml.html>`_

        """
        # standard reporting alpha
        confidence: float = 0.95
        # ensure correct computation
        data: np.ndarray = data.astype(float)
        # sample size
        n: int = len(data)
        # mu
        m: float = np.mean(data)
        # standard error on the distribution of means
        se_dist: float = scipy.stats.sem(data)
        # revert the mean computation denominator since our sores are already
        # a distribution of means
        se: float = se_dist * math.sqrt(n)
        # compute the t-value from the t-distribution
        t_value: float = scipy.stats.t.ppf((1 + confidence) / 2., df=n - 1)
        # margin of error
        ci_len: float = se * t_value
        return m - ci_len, m + ci_len

    def _get_metadata(self, df: pd.DataFrame) -> Dict[str, int]:
        return {'folds': df['fold'].max().item() + 1,
                'repeats': df['repeat'].max().item() + 1}

    def _cross_validate_stats(self, dfd_res: DataFrameDescriber) -> \
            DataFrameDescriber:
        cols: List[str] = list(PredictionsDataFrameFactory.TEST_METRIC_COLUMNS)
        cvm: Dict[str, int] = self._get_metadata(dfd_res.df)
        df: pd.DataFrame = dfd_res.df[cols]
        rows: List[pd.Series] = []
        index_meta: Dict[str, str] = OrderedDict()
        dfd_desc: str = (f"{cvm['folds']}-Fold Cross {cvm['repeats']} " +
                         'with Repeat(s) Validation Statistics')
        stat: str
        for stat in 'mean min max std'.split():
            row: pd.Series = getattr(df, stat)()
            row.name = stat
            rows.append(row.to_frame().T)
            index_meta[stat] = f'the {stat} of the performance metric'
        ci_meths: Tuple[Tuple[str, str, Callable]] = (
            ('t-ci', 't-distribution', self._calc_t_ci),)
        for (name, desc, ci_fn) in ci_meths:
            cis: List[Tuple[float, float]] = []
            col_data: pd.Series
            for col_data in map(lambda t: t[1], df.items()):
                ci_min, ci_max = ci_fn(col_data.to_numpy())
                # ci_max may be greater than 1, but it doesn't make sense to
                # *report* it as such
                ci_max = min(ci_max, 1.)
                cis.append((ci_min, ci_max))
            row: pd.Series = pd.Series(data=cis, index=df.columns, name=name)
            rows.append(row.to_frame().T)
            index_meta[name] = desc
        dfs: pd.DataFrame = pd.concat(rows)
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
            desc=f'{self.result_manager.name.capitalize()} {dfd_desc}',
            meta=meta,
            index_meta=index_meta)

    @property
    def cross_validate_describer(self) -> DataDescriber:
        """A data describer with the results of a cross-validation.

        The describer returned includes the metrics for each fold and summary
        statitics for all folds.  The statistics
        :class:`~zensols.datdesc.desc.DataFrameDescriber` (describer with name
        ``cross-validation-stats``) contain the following 95% mean confidence
        interval calculations given in the ``stat`` row:

          * ``t-ci``: use the t-scores from a t-distribution (assumes a normal
            distribution across scores)

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
