"""Stashes that operate on a dataframe, which are useful to common machine
learning tasks.

"""
__author__ = 'Paul Landes'

from typing import Iterable, Dict, Set, Tuple
from dataclasses import dataclass, field
import logging
import sys
from abc import abstractmethod, ABCMeta
from collections import OrderedDict
from pathlib import Path
import pandas as pd
from zensols.config import Writable
from zensols.persist import (
    Deallocatable,
    PersistedWork,
    persisted,
    ReadOnlyStash,
    PrimeableStash,
)
from zensols.dataset import SplitKeyContainer

logger = logging.getLogger(__name__)


@dataclass
class DataframeStash(SplitKeyContainer, ReadOnlyStash, PrimeableStash,
                     Deallocatable, Writable, metaclass=ABCMeta):
    """A factory stash that uses a Pandas data frame from which to load.  It uses
    the data frame index as the keys.  The dataframe is usually constructed by
    reading a file (i.e.CSV) and doing some transformation before using it in
    an implementation of this stash.

    The dataframe created by :meth:`_get_dataframe` must have a string index
    since keys for all stashes are of type :class:`str`.

    This is can be done with::

        df.index = df.index.map(str)

    """
    dataframe_path: Path = field()
    """The path to store the pickeled version of the generated dataframe
    created with :meth:`_get_dataframe`.

    """

    key_path: Path = field()
    """The path where the key splits (as a ``dict``) is pickled."""

    split_col: str = field()
    """The column name in the dataframe used to indicate the split
    (i.e. ``train`` vs ``test``).

    """

    def __post_init__(self):
        super().__post_init__()
        Deallocatable.__init__(self)
        logger.debug(f'split stash post init: {self.dataframe_path}')
        self.dataframe_path.parent.mkdir(parents=True, exist_ok=True)
        self._dataframe = PersistedWork(self.dataframe_path, self)
        self._keys_by_split = PersistedWork(self.key_path, self)

    def deallocate(self):
        super().deallocate()
        self._dataframe.deallocate()
        self._keys_by_split.deallocate()

    @abstractmethod
    def _get_dataframe(self) -> pd.DataFrame:
        """Get or create the dataframe

        """
        pass

    def _create_keys_for_split(self, split_name: str, df: pd.DataFrame) -> \
            Iterable[str]:
        """Generate an iterable of string keys.  It is expected this method to be
        potentially very expensive, so the results are cached to disk.  This
        implementation returns the dataframe index.

        :param split_name: the name of the split (i.e. ``train`` vs ``test``)
        :param df: the data frame for the grouping of keys from CSV of data

        """
        return df.index

    def _get_counts_by_key(self) -> Dict[str, int]:
        sc = self.split_col
        return dict(self.dataframe.groupby([sc])[sc].count().items())

    @persisted('_split_names')
    def _get_split_names(self) -> Set[str]:
        return set(self.dataframe[self.split_col].unique())

    @persisted('_keys_by_split')
    def _get_keys_by_split(self) -> Dict[str, Tuple[str]]:
        keys_by_split = OrderedDict()
        split_col = self.split_col
        for split, df in self.dataframe.groupby([split_col]):
            logger.info(f'parsing keys for {split}')
            keys = self._create_keys_for_split(split, df)
            keys_by_split[split] = tuple(keys)
        return keys_by_split

    @property
    @persisted('_dataframe')
    def dataframe(self):
        df = self._get_dataframe()
        dt = df.index.dtype
        if dt != object:
            s = f'data frame index must be of type string, but got: {dt}'
            raise ValueError(s)
        return df

    def prime(self):
        super().prime()
        self.dataframe

    def clear(self):
        logger.debug('clearing split stash')
        self._dataframe.clear()
        self.clear_keys()

    def clear_keys(self):
        """Clear only the cache of keys generated from the group by.

        """
        self._keys_by_split.clear()

    def load(self, name: str) -> pd.Series:
        return self.dataframe.loc[name]

    def exists(self, name: str) -> bool:
        return name in self.dataframe.index

    def keys(self) -> Iterable[str]:
        return map(str, self.dataframe.index)

    def write(self, depth: int = 0, writer=sys.stdout):
        s = self._sp(depth)
        s2 = self._sp(depth + 1)
        total = self.dataframe.shape[0]
        writer.write(f'{s}data frame splits:\n')
        for split, cnt in self.counts_by_key.items():
            writer.write(f'{s2}{split}: {cnt} ({cnt/total*100:.1f}%)\n')
        writer.write(f'{s2}total: {total}\n')


@dataclass
class DefaultDataframeStash(DataframeStash):
    """A default implementation of :class:`.DataframeSplitStash` that creates the
    Pandas dataframe by simply reading it from a specificed CSV file.  The
    index is a string type appropriate for a stash.

    """
    input_csv_path: Path = field()
    """A path to the CSV of the source data."""

    def _get_dataframe(self) -> pd.DataFrame:
        df = pd.read_csv(self.input_csv_path)
        df.index = df.index.map(str)
        return df
