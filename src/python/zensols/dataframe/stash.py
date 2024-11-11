"""Stashes that operate on a dataframe, which are useful to common machine
learning tasks.

"""
__author__ = 'Paul Landes'

from typing import Iterable, Dict, Set, Tuple
from dataclasses import dataclass, field
import logging
import sys
from io import TextIOBase
from abc import abstractmethod, ABCMeta
from collections import OrderedDict
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from zensols.util import APIError
from zensols.config import Writable
from zensols.persist import (
    Deallocatable,
    PersistedWork,
    persisted,
    ReadOnlyStash,
    PrimeableStash,
)
from zensols.install import Installer, Resource
from zensols.dataset import SplitKeyContainer

logger = logging.getLogger(__name__)


class DataframeError(APIError):
    """Thrown for dataframe stash issues."""


@dataclass
class DataframeStash(ReadOnlyStash, Deallocatable, Writable,
                     PrimeableStash, metaclass=ABCMeta):
    """A factory stash that uses a Pandas data frame from which to load.  It
    uses the data frame index as the keys and :class:`pandas.Series` as values.
    The dataframe is usually constructed by reading a file (i.e.CSV) and doing
    some transformation before using it in an implementation of this stash.

    The dataframe created by :meth:`_get_dataframe` must have a string or
    integer index since keys for all stashes are of type :class:`str`.  The
    index will be mapped to a string if it is an int automatically.

    """
    dataframe_path: Path = field()
    """The path to store the pickeled version of the generated dataframe
    created with :meth:`_get_dataframe`.

    """
    def __post_init__(self):
        super().__post_init__()
        Deallocatable.__init__(self)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'split stash post init: {self.dataframe_path}')
        self._dataframe = PersistedWork(self.dataframe_path, self, mkdir=True)

    def deallocate(self):
        super().deallocate()
        self._dataframe.deallocate()

    @abstractmethod
    def _get_dataframe(self) -> pd.DataFrame:
        """Get or create the dataframe

        """
        pass

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        dt = df.index.dtype
        if dt != object:
            if dt != int:
                s = f'Data frame index must be a string or int, but got: {dt}'
                raise DataframeError(s)
            else:
                df.index = df.index.map(str)
        return df

    @property
    @persisted('_dataframe')
    def dataframe(self):
        df = self._get_dataframe()
        df = self._prepare_dataframe(df)
        return df

    def prime(self):
        super().prime()
        self.dataframe

    def clear(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('clearing dataframe stash')
        self._dataframe.clear()

    def load(self, name: str) -> pd.Series:
        return self.dataframe.loc[name]

    def exists(self, name: str) -> bool:
        return name in self.dataframe.index

    def keys(self) -> Iterable[str]:
        return map(str, self.dataframe.index)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        df = self.dataframe
        self._write_line(f'rows: {df.shape[0]}', depth, writer)
        self._write_line(f'cols: {", ".join(df.columns)}', depth, writer)


@dataclass
class SplitColumnDataframeStash(DataframeStash):
    """A stash that provides a way to get the labels and label count of the
    dataframe.

    """
    split_col: str = field()
    """The column name in the dataframe used to indicate the split
    (i.e. ``train`` vs ``test``).

    """
    @persisted('_labels')
    def get_labels(self) -> Tuple[str, ...]:
        return tuple(self.dataframe[self.split_col].drop_duplicates().to_list())

    def get_label_count(self) -> int:
        return len(self.get_labels())


@dataclass
class SplitKeyDataframeStash(SplitColumnDataframeStash, SplitKeyContainer):
    """A stash and split key container that reads from a dataframe.

    """
    key_path: Path = field()
    """The path where the key splits (as a ``dict``) is pickled."""

    def __post_init__(self):
        super().__post_init__()
        self._keys_by_split = PersistedWork(self.key_path, self, mkdir=True)

    def deallocate(self):
        super().deallocate()
        self._keys_by_split.deallocate()

    def _create_keys_for_split(self, split_name: str, df: pd.DataFrame) -> \
            Iterable[str]:
        """Generate an iterable of string keys.  It is expected this method to
        be potentially very expensive, so the results are cached to disk.  This
        implementation returns the dataframe index.

        :param split_name: the name of the split (i.e. ``train`` vs ``test``)
        :param df: the data frame for the grouping of keys from CSV of data

        """
        return df.index

    def _get_counts_by_key(self) -> Dict[str, int]:
        sc = self.split_col
        return dict(self.dataframe.groupby(sc)[sc].count().items())

    @persisted('_split_names')
    def _get_split_names(self) -> Set[str]:
        return set(self.dataframe[self.split_col].unique())

    @persisted('_keys_by_split')
    def _get_keys_by_split(self) -> Dict[str, Tuple[str]]:
        keys_by_split = OrderedDict()
        split_col = self.split_col
        split: str
        df: pd.DataFrame
        for split, df in self.dataframe.groupby(split_col):
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'parsing keys for {split}')
            keys = self._create_keys_for_split(split, df)
            keys_by_split[split] = tuple(keys)
        return keys_by_split

    def clear(self):
        super().clear()
        self.clear_keys()

    def clear_keys(self):
        """Clear only the cache of keys generated from the group by.

        """
        self._keys_by_split.clear()

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        total = self.dataframe.shape[0]
        self._write_line('data frame splits:', depth, writer)
        for split, cnt in self.counts_by_key.items():
            self._write_line(f'{split}: {cnt} ({cnt/total*100:.1f}%)',
                             depth, writer)
        self._write_line(f'total: {total}', depth, writer)


@dataclass
class AutoSplitDataframeStash(SplitKeyDataframeStash):
    """Automatically a dataframe in to train, test and validation datasets by
    adding a :obj:`split_col` with the split name.

    """
    distribution: Dict[str, float] = field()
    """The distribution as a percent across all key splits.  The distribution
    values must add to 1.  The keys must have ``train``, ``test`` and
    ``validate``.

    """
    def __post_init__(self):
        super().__post_init__()
        sm = float(sum(self.distribution.values()))
        err_low, err_high, errm = (1. - sm), (1. + sm), 1e-1
        if err_low > errm:
            raise APIError('distriubtion must add to 1: ' +
                           f'{self.distribution} (err={err_low} > errm)')
        if err_high < errm:
            raise APIError('distriubtion must add to 1: ' +
                           f'{self.distribution} (err={err_low} > errm)')

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        n_train = self.distribution['train']
        n_test = self.distribution['test']
        n_val = self.distribution['validate']
        n_test_val = n_test + n_val
        n_test = n_test / n_test_val
        train, test_val = train_test_split(df, test_size=1 - n_train)
        test, val = train_test_split(test_val, test_size=n_test)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'split dataframe: train: {train.size}, ' +
                         f'test: {test.size}, validation: {val.size}')
        # pandas complains about modifying a slice
        train = train.copy()
        test = test.copy()
        val = val.copy()
        train[self.split_col] = 'train'
        test[self.split_col] = 'test'
        val[self.split_col] = 'validation'
        df = pd.concat([train, test, val], ignore_index=False)
        df = super()._prepare_dataframe(df)
        return df


@dataclass
class DefaultDataframeStash(SplitKeyDataframeStash):
    """A default implementation of :class:`.DataframeSplitStash` that creates
    the Pandas dataframe by simply reading it from a specificed CSV file.  The
    index is a string type appropriate for a stash.

    """
    input_csv_path: Path = field()
    """A path to the CSV of the source data."""

    def _get_dataframe(self) -> pd.DataFrame:
        return pd.read_csv(self.input_csv_path)


@dataclass
class ResourceFeatureDataframeStash(SplitColumnDataframeStash):
    """A dataframe that installs a corpus and then reads a file to create the
    Pandas dataframe.

    """
    installer: Installer = field()
    """The installer used to download and uncompress dataset."""

    resource: Resource = field()
    """Use to resolve the corpus file."""

    def _get_dataframe(self) -> pd.DataFrame:
        self.installer()
        path: Path = self.installer[self.resource]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'loading CSV from resource path: {path}')
        df = pd.read_csv(path)
        df = df.rename(columns=dict(
            zip(df.columns, map(str.lower, df.columns))))
        return df
