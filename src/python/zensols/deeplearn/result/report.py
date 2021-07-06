"""A utility class to summarize all results in a directory.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field
from pathlib import Path
import logging
import re
import pickle
import pandas as pd
from zensols.deeplearn import DatasetSplitType
from . import ModelResultError, ModelResult, ModelResultManager

logger = logging.getLogger(__name__)


@dataclass
class ModelResultReporter(object):
    """Summarize all results in a directory from the output of model execution from
    :class:`~zensols.deeplearn.model.ModelExectuor`.

    The class iterates through the pickled binary output files from the run and
    summarizes in a Pandas dataframe, which is handy for reporting in papers.

    """
    result_manager: ModelResultManager = field()
    """Contains the results to report on--and specifically the path to directory
    where the results were persisted.

    """
    @property
    def dataframe(self) -> pd.DataFrame:
        """Return the summarized results (see class docs).

        :return: the Pandas dataframe of the results

        """
        m = re.match(r'.*\.(.+?)$', self.result_manager.pattern)
        if m is None:
            raise ModelResultError(
                f'Results manager extension pattern incorrect: {self.pattern}')
        ext = m.group(1)
        path = self.result_manager.path
        paths = filter(lambda p: p.name.endswith(ext), path.iterdir())
        rows = []
        cols = ('name train_duration converged features ' +
                'wF1 wP wR mF1 mP mR MF1 MP MR ' +
                'train_occurs validation_occurs test_occurs').split()
        dpt_key = 'n_total_data_points'
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'reading results from {path}')
        for path in paths:
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'parsing results: {path}')
            with open(path, 'rb') as f:
                res: ModelResult = pickle.load(f)
            train = res.dataset_result.get(DatasetSplitType.train)
            validate = res.dataset_result.get(DatasetSplitType.validation)
            test = res.dataset_result.get(DatasetSplitType.test)
            if train is not None:
                dur = train.end_time - train.start_time
                hours, remainder = divmod(dur.seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                dur = f'{hours:02}:{minutes:02}:{seconds:02}'
            if validate is not None:
                conv_epoch = validate.statistics['n_epoch_converged']
            else:
                conv_epoch = None
            if test is not None:
                mets = test.metrics
                features = ', '.join(res.decoded_attributes)
                row = [res.name, dur, conv_epoch, features,
                       mets.weighted.f1, mets.weighted.precision, mets.weighted.recall,
                       mets.micro.f1, mets.micro.precision, mets.micro.recall,
                       mets.macro.f1, mets.macro.precision, mets.macro.recall,
                       train.statistics[dpt_key], validate.statistics[dpt_key],
                       test.statistics[dpt_key]]
                rows.append(row)
        return pd.DataFrame(rows, columns=cols)

    def dump(self, path: Path):
        """Create the summarized results and write them to the file system.

        """
        self.dataframe.to_csv(path)
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'wrote results summary: {path}')
