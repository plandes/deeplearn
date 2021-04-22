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
from zensols.deeplearn import ModelError
from zensols.deeplearn.result import ModelResult, ModelResultManager
from . import ModelExecutor

logger = logging.getLogger(__name__)


@dataclass
class DataComparison(Dictable):
    """Contains the results from two runs used to compare.  The data in this object
    is used to compare the validation loss from a previous run to a run that's
    currently in progress.  This is provided along with the performance metrics
    of the runs when written with :meth:`write`.

    """
    key: str = field()
    """The results key used with a :class:`.ModelResultManager`."""

    previous: ModelResult = field()
    """The previous resuls of the model from a previous run."""

    current: ModelResult = field()
    """The current results, which is probably a model currently running."""

    compare_df: pd.DataFrame = field()
    """A dataframe with the validation loss from the previous and current results
    and that difference.

    """

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
    """Load results from a previous run of the :class:`ModelExecutor` and a more
    recent run.  This run is usually a currently running model to compare the
    results during training.  This might provide meaningful information such as
    whether to early stop training.

    """
    executor: ModelExecutor = field()
    """The executor (not the running executor necessary) that will load the
    results if not already loadded.

    """

    previous_results_key: str = field()
    """The key given to retreive the previous results with
    :class:`ModelResultManager`.

    """

    cache_previous_results: bool = field()
    """If ``True``, globally cache the previous results to avoid having to
    reload each time.

    """

    def __post_init__(self):
        self._previous_results = PersistedWork(
            '_previous_results', self,
            cache_global=self.cache_previous_results)

    def clear(self):
        """Clear the previous results, if cached.

        """
        self._previous_results.clear()

    @property
    @persisted('_previous_results')
    def previous_results(self) -> ModelResult:
        """Return the previous results (see class docs).

        """
        rm: ModelResultManager = self.executor.result_manager
        if rm is None:
            rm = ModelError('No result manager available')
        return rm[self.previous_results_key]

    @property
    def current_results(self) -> Tuple[ModelResult, ModelResult]:
        """Return the current results (see class docs).

        """
        if self.executor.model_result is None:
            self.executor.load()
        return self.executor.model_result

    @property
    def comparison(self) -> DataComparison:
        """Load the results data and create a comparison instance read to write or
        jsonify.

        """
        prev, cur = self.previous_results, self.current_results
        prev_losses = prev.validation.losses
        cur_losses = cur.validation.losses
        cur_len = len(cur_losses)
        df = pd.DataFrame({'epoch': range(cur_len),
                           'previous': prev_losses[:cur_len],
                           'current': cur_losses})
        df['improvement'] = df['previous'] - df['current']
        return DataComparison(self.previous_results_key, prev, cur, df)
