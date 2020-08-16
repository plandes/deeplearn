"""Analysis tools to compare results.

"""
__author__ = 'Paul Landes'


from typing import Tuple, Iterable
from dataclasses import dataclass, field
import logging
import sys
from io import TextIOBase, StringIO
import pandas as pd
from zensols.config import Dictable
from zensols.persist import PersistedWork, persisted
from zensols.deeplearn.result import ModelResult
from . import ModelFacade

logger = logging.getLogger(__name__)


@dataclass
class DataComparison(Dictable):
    key: str
    previous: ModelResult
    current: ModelResult
    compare_df: pd.DataFrame

    def _get_dictable_attributes(self) -> Iterable[str]:
        return self._split_str_to_attributes('key previous current')

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        converge_idx = self.previous.validation.converged_epoch.index
        sio = StringIO()
        self._write_line(f'result: {self.key}', depth, writer)
        self._write_line('loss:', depth, writer)
        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None):
            print(self.compare_df, file=sio)
        self._write_block(sio.getvalue().strip(), depth + 1, writer)
        self._write_line('previous:', depth, writer)
        self.previous.validation.write(depth + 1, writer)
        self._write_line(f'converged: {converge_idx}',
                         depth + 1, writer)
        self._write_line('current:', depth, writer)
        self.current.validation.write(depth + 1, writer)


@dataclass
class ResultAnalyzer(object):
    facade: ModelFacade
    previous_results_key: str
    cache_previous_results: bool = field(default=False)

    def __post_init__(self):
        self._previous_results = PersistedWork(
            '_previous_results', self,
            cache_global=self.cache_previous_results)

    def clear(self):
        self._previous_results.clear()

    @property
    @persisted('_previous_results')
    def previous_results(self) -> ModelResult:
        return self.facade.result_manager[self.previous_results_key]

    @property
    def current_results(self) -> Tuple[ModelResult, ModelResult]:
        executor = self.facade.executor
        executor.load()
        return executor.model_result

    @property
    def comparison(self) -> DataComparison:
        prev, cur = self.previous_results, self.current_results
        prev_losses = prev.validation.losses
        cur_losses = cur.validation.losses
        cur_len = len(cur_losses)
        df = pd.DataFrame({'epoch': range(cur_len),
                           'previous': prev_losses[:cur_len],
                           'current': cur_losses})
        df['improvement'] = df['previous'] - df['current']
        return DataComparison(self.previous_results_key, prev, cur, df)
