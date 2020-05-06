"""Utility stashes useful to common machine learning tasks.

"""
__author__ = 'Paul Landes'

import sys
import logging
from typing import Iterable, Dict, Set
from dataclasses import dataclass
from abc import abstractmethod, ABCMeta
from itertools import chain
from pathlib import Path
import numpy as np
import pandas as pd
from zensols.util import time
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
class SplitStash(ReadOnlyStash, PrimeableStash, metaclass=ABCMeta):
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

    """
    dataframe_path: Path
    split_col: str

    def __post_init__(self):
        logger.debug(f'split stash post init: {self.dataframe_path}')
        self.dataframe_path.parent.mkdir(parents=True, exist_ok=True)
        self._dataframe = PersistedWork(self.dataframe_path, self)

    @abstractmethod
    def _get_dataframe(self) -> pd.DataFrame:
        """Get or create the dataframe
        """
        pass

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

    @property
    @persisted('_split_names')
    def split_names(self) -> Set[str]:
        return set(self.dataframe[self.split_col].unique())

    def clear(self):
        logger.debug('clearing split stash')
        self._dataframe.clear()

    def load(self, name: str) -> pd.Series:
        return self.dataframe.loc[name]

    def exists(self, name: str) -> bool:
        return name in self.dataframe.index

    def keys(self) -> Iterable[str]:
        return map(str, self.dataframe.index)

    @property
    def counts_by_key(self) -> Dict[str, int]:
        sc = self.split_col
        return dict(self.dataframe.groupby([sc])[sc].count().items())

    def write(self, depth: int = 0, writer=sys.stdout):
        s = ' ' * (depth * 2)
        s2 = ' ' * ((depth + 1) * 2)
        writer.write(f'{s}data frame splits:\n')
        for split, cnt in self.counts_by_key.items():
            writer.write(f'{s2}{split}: {cnt}\n')
        writer.write(f'{s2}total: {self.dataframe.shape[0]}\n')


@dataclass
class SplitDatasetStash(DelegateStash, PrimeableStash):
    """Generates a separate stash instance for each data set split (i.e. ``train``
    vs ``test).  Each split instance holds the data (keys and values) for each
    split as indicated in a dataframe colum.

    Stash instances by split are obtained with ``splits``, and will have
    a ``split`` attribute that give the name of the split.

    :param split_stash: the instance that provides the data frame for the
                        splits in the data set
    :param key_path: the path where the key splits (as a ``dict``) is pickled

    :see splits:

    """
    split_stash: SplitStash
    key_path: Path

    def __post_init__(self):
        super().__post_init__()
        self._dataframe_keys_by_split = PersistedWork(self.key_path, self)
        self.split = None

    def _get_keys_for_split(self, split_name: str, df: pd.DataFrame) -> \
            Iterable[str]:
        """Generate an iterable of string keys.  It is expected this method to be
        potentially very expensive, so the results are cached to disk.  This
        implementation returns the dataframe index.

        :param split_name: the name of the split (i.e. ``train`` vs ``test``)
        :param df: the data frame for the grouping of keys from CSV of data

        """
        return df.index

    @property
    def dataframe(self) -> pd.DataFrame:
        return self.split_stash.dataframe

    @property
    def split_names(self) -> Set[str]:
        """Return the names of each split in the dataset.

        """
        return self.split_stash.split_names

    @property
    @persisted('_dataframe_keys_by_split')
    def dataframe_keys_by_split(self) -> Dict[str, Set[str]]:
        """Generate a dictionary of split name to keys for that split.  It is expected
        this method will be very expensive.

        """
        logger.info(f'creating key splits; cache to {self.key_path}')
        keys_by_split = {}
        split_col = self.split_stash.split_col
        for split, df in self.split_stash.dataframe.groupby([split_col]):
            logger.info(f'parsing keys for {split}')
            keys = self._get_keys_for_split(split, df)
            keys_by_split[split] = set(keys)
        return keys_by_split

    @property
    @persisted('_keys_by_split')
    def keys_by_split(self) -> Dict[str, Set[str]]:
        """Return keys by split type (i.e. ``train`` vs ``test``) for only those keys
        available by the delegate backing stash.

        """
        logger.debug('creating in memory available keys data structure')
        with time('created key data structures', logging.DEBUG):
            delegate_keys = set(self.delegate.keys())
            avail_kbs = {}
            for split, keys in self.dataframe_keys_by_split.items():
                avail_kbs[split] = keys & delegate_keys
            return avail_kbs

    def prime(self):
        super().prime()
        self.keys_by_split

    def keys(self) -> Iterable[str]:
        self.prime()
        logger.debug(f'keys for {self.split}')
        kbs = self.keys_by_split
        logger.debug(f'obtained keys for {self.split}')
        if self.split is None:
            return chain.from_iterable(kbs.values())
        else:
            return kbs[self.split]

    def clear(self):
        """Clear and destory key and delegate data.

        """
        if self.has_data:
            super().clear()
            self.clear_keys()
            self.split_stash.clear()

    def clear_keys(self):
        """Clear only the cache of keys generated from the group by.

        """
        self._dataframe_keys_by_split.clear()

    def clear_docs(self):
        if self.has_data:
            super().clear()

    @property
    @persisted('_splits')
    def splits(self) -> Dict[str, Stash]:
        """Return an instance of ta stash that contains only the data for a split.

        :param split: the name of the split of the instance to get
                      (i.e. ``train``, ``test``).

        """
        self.prime()
        stashes = {}
        for split in self.split_names:
            clone = self.__class__(
                self.delegate, self.split_stash, self.key_path)
            clone.split = split
            clone._keys_by_split = self._keys_by_split
            stashes[split] = clone
        return stashes

    @property
    def counts_by_key(self) -> Dict[str, int]:
        return dict(map(lambda i: (i[0], len(i[1])),
                        self.keys_by_split.items()))

    def check_key_consistent(self):
        return self.counts_by_key == self.split_stash.counts_by_key

    def write(self, depth: int = 0, writer=sys.stdout):
        s = ' ' * (depth * 2)
        s2 = ' ' * ((depth + 1) * 2)
        writer.write(f'{s}data frame splits:\n')
        t = 0
        for k, ks in self.dataframe_keys_by_split.items():
            ln = len(ks)
            writer.write(f'{s2}{k}: {ln}\n')
            t += ln
        writer.write(f'{s2}total: {t}\n')

        writer.write(f'{s}delegate available splits:\n')
        t = 0
        for k, ln in self.counts_by_key.items():
            writer.write(f'{s2}{k}: {ln}\n')
            t += ln
        writer.write(f'{s2}total: {t}\n')
        self.split_stash.write(depth, writer)
        ckc = self.check_key_consistent()
        writer.write(f'{s}total this instance: {len(self)}, ' +
                     f'keys consistent: {ckc}\n')
