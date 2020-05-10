"""Utility stashes useful to common machine learning tasks.

"""
__author__ = 'Paul Landes'

import sys
import logging
from typing import Iterable, Dict, Set
from dataclasses import dataclass
from abc import abstractmethod, ABCMeta, ABC
from itertools import chain
from pathlib import Path
import numpy as np
import pandas as pd
from zensols.util import time
from zensols.config import Writable
from zensols.persist import (
    PersistedWork,
    persisted,
    Stash,
    DelegateStash,
    ReadOnlyStash,
    PrimeableStash,
)

logger = logging.getLogger(__name__)


@dataclass
class SplitKeyContainer(ABC):
    """An interface defining a container that partitions data sets (i.e. ``train``
    vs ``test``).  For instances of this class, that data are the unique keys
    that point at the data.

    """
    def _get_split_names(self) -> Set[str]:
        return self._get_keys_by_split().keys()

    def _get_counts_by_key(self) -> Dict[str, int]:
        ks = self._get_keys_by_split()
        return {k: len(ks[k]) for k in ks.keys()}

    @abstractmethod
    def _get_keys_by_split(self) -> Dict[str, Set[str]]:
        pass

    @property
    def split_names(self) -> Set[str]:
        """Return the names of each split in the dataset.

        """
        return self._get_split_names()

    @property
    def counts_by_key(self) -> Dict[str, int]:
        """Return data set splits name to count for that respective split.

        """
        return self._get_counts_by_key()

    @property
    def keys_by_split(self) -> Dict[str, Set[str]]:
        """Generate a dictionary of split name to keys for that split.  It is expected
        this method will be very expensive.

        """
        return self._get_keys_by_split()


@dataclass
class SplitStashContainer(PrimeableStash, SplitKeyContainer,
                          metaclass=ABCMeta):
    """An interface like ``SplitKeyContainer``, but whose implementations are of
    ``Stash`` containing the instance data.

    """
    @abstractmethod
    def _get_split_name(self) -> str:
        pass

    @abstractmethod
    def _get_splits(self) -> Dict[str, Stash]:
        pass

    @property
    def split_name(self) -> str:
        """Return the name of the split this stash contains.  Thus, all data/items
        returned by this stash are in the data set given by this name
        (i.e. ``train``).

        """
        return self._get_split_name()

    @property
    def splits(self) -> Dict[str, Stash]:
        """Return a dictionary with keys as split names and values as the stashes
        represented by that split.

        :see split_name:

        """
        return self._get_splits()


@dataclass
class DataframeStash(SplitKeyContainer, ReadOnlyStash, PrimeableStash,
                     Writable, metaclass=ABCMeta):
    """A factory stash that uses a Pandas data frame from which to load.  It uses
    the data frame index as the keys.  The dataframe is usually constructed by
    reading a file (i.e.CSV) and doing some transformation before using it in
    an implementation of this stash.

    The dataframe created by ``_get_dataframe`` must have a string index since
    keys for all stashes are of type ``str``.  This is can be done with:
    ``
        df.index = df.index.map(str)
    ``

    :param dataframe_path: the path to store the pickeled version of the
                           generated dataframe created with ``_get_dataframe``.
    :param split_col: the column name in the dataframe used to indicate
                      the split (i.e. ``train`` vs ``test``)
    :param key_path: the path where the key splits (as a ``dict``) is pickled

    """
    dataframe_path: Path
    key_path: Path
    split_col: str

    def __post_init__(self):
        logger.debug(f'split stash post init: {self.dataframe_path}')
        self.dataframe_path.parent.mkdir(parents=True, exist_ok=True)
        self._dataframe = PersistedWork(self.dataframe_path, self)
        self._keys_by_split = PersistedWork(self.key_path, self)

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
    def _get_keys_by_split(self) -> Dict[str, Set[str]]:
        keys_by_split = {}
        split_col = self.split_col
        for split, df in self.dataframe.groupby([split_col]):
            logger.info(f'parsing keys for {split}')
            keys = self._create_keys_for_split(split, df)
            keys_by_split[split] = set(keys)
        return keys_by_split

    @property
    @persisted('_dataframe')
    def dataframe(self):
        df = self._get_dataframe()
        dt = df.index.dtype
        if dt != np.object:
            s = f'data frame must be of type string, but got: {dt}'
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
    """A default implementation of ``DataframeSplitStash`` that creates the Pandas
    dataframe by simply reading it from a specificed CSV file.  The index is a
    string type appropriate for a stash.

    """
    input_csv_path: Path

    def _get_dataframe(self) -> pd.DataFrame:
        df = pd.read_csv(self.input_csv_path)
        df.index = df.index.map(str)
        return df


@dataclass
class DatasetSplitStash(DelegateStash, SplitStashContainer, Writable):
    """Generates a separate stash instance for each data set split (i.e. ``train``
    vs ``test).  Each split instance holds the data (keys and values) for each
    split as indicated in a dataframe colum.

    Stash instances by split are obtained with ``splits``, and will have
    a ``split`` attribute that give the name of the split.

    :param split_container: the instance that provides the data frame for
                            the splits in the data set

    :see splits:

    """
    split_container: SplitKeyContainer

    def __post_init__(self):
        super().__post_init__()
        self.inst_split_name = None
        self._keys_by_split = PersistedWork('_keys_by_split', self)

    @persisted('_keys_by_split')
    def _get_keys_by_split(self) -> Dict[str, Set[str]]:
        """Return keys by split type (i.e. ``train`` vs ``test``) for only those keys
        available by the delegate backing stash.

        """
        logger.debug('creating in memory available keys data structure')
        with time('created key data structures', logging.DEBUG):
            delegate_keys = set(self.delegate.keys())
            avail_kbs = {}
            for split, keys in self.split_container.keys_by_split.items():
                ks = keys & delegate_keys
                #logger.debug(f'{keys} & {delegate_keys}')
                logger.debug(f'{split} has {len(ks)} keys')
                avail_kbs[split] = ks
            return avail_kbs

    def _get_counts_by_key(self) -> Dict[str, int]:
        return dict(map(lambda i: (i[0], len(i[1])),
                        self.keys_by_split.items()))

    def check_key_consistent(self):
        return self.counts_by_key == self.split_container.counts_by_key

    def keys(self) -> Iterable[str]:
        self.prime()
        logger.debug(f'keys for {self.split_name}')
        kbs = self.keys_by_split
        logger.debug(f'obtained keys for {self.split_name}')
        if self.split_name is None:
            return chain.from_iterable(kbs.values())
        else:
            return kbs[self.split_name]

    def prime(self):
        logger.debug('priming ds split stash')
        super().prime()
        self.keys_by_split

    def _delegate_has_data(self):
        return not isinstance(self.delegate, PrimeableStash) or \
            self.delegate.has_data

    def clear(self):
        """Clear and destory key and delegate data.

        """
        del_has_data = self._delegate_has_data()
        logger.debug(f'clearing: {del_has_data}')
        if del_has_data:
            logger.debug('clearing delegate and split container')
            super().clear()
            self.split_container.clear()
            self._keys_by_split.clear()

    def _get_split_names(self) -> Set[str]:
        return self.split_container.split_names

    def _get_split_name(self):
        return self.inst_split_name

    @persisted('_splits')
    def _get_splits(self) -> Dict[str, Stash]:
        """Return an instance of ta stash that contains only the data for a split.

        :param split: the name of the split of the instance to get
                      (i.e. ``train``, ``test``).

        """
        self.prime()
        stashes = {}
        for split_name in self.split_names:
            clone = self.__class__(
                delegate=self.delegate, split_container=self.split_container)
            clone.inst_split_name = split_name
            clone._keys_by_split = self._keys_by_split
            stashes[split_name] = clone
        return stashes

    def write(self, depth: int = 0, writer=sys.stdout):
        s = self._sp(depth)
        s2 = self._sp(depth + 1)
        writer.write(f'{s}split stash splits:\n')
        t = 0
        for ks in self.split_container.keys_by_split.values():
            t += len(ks)
        for k, ks in self.split_container.keys_by_split.items():
            ln = len(ks)
            writer.write(f'{s2}{k}: {ln} ({ln/t*100:.1f}%)\n')
        writer.write(f'{s2}total: {t}\n')
        ckc = self.check_key_consistent()
        writer.write(f'{s}total this instance: {len(self)}\n')
        writer.write(f'{s}keys consistent: {ckc}\n')
