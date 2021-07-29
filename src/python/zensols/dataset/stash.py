"""Utility stashes useful to common machine learning tasks.

"""
__author__ = 'Paul Landes'

import sys
import logging
from typing import Iterable, Dict, Set, Callable, Tuple, Any, List
from dataclasses import dataclass, field
from itertools import chain
from collections import OrderedDict
from io import TextIOBase
from zensols.util import time
from zensols.config import Writable
from zensols.persist import (
    PersistedWork,
    PersistableContainer,
    persisted,
    Stash,
    DelegateStash,
    PreemptiveStash,
)
from zensols.dataset import SplitStashContainer, SplitKeyContainer

logger = logging.getLogger(__name__)


@dataclass
class DatasetSplitStash(DelegateStash, SplitStashContainer,
                        PersistableContainer, Writable):
    """A default implementation of :class:`.SplitStashContainer`.  However, it
    needs an instance of a :class:`.SplitKeyContainer`.  This implementation
    generates a separate stash instance for each data set split (i.e. ``train``
    vs ``test``).  Each split instance holds the data (keys and values) for
    each split.

    Stash instances by split are obtained with ``splits``, and will have
    a ``split`` attribute that give the name of the split.

    :see: :meth:`.SplitStashContainer.splits`

    """
    split_container: SplitKeyContainer = field()
    """The instance that provides the splits in the dataset."""

    def __post_init__(self):
        super().__post_init__()
        PersistableContainer.__init__(self)
        self.inst_split_name = None
        self._keys_by_split = PersistedWork('_keys_by_split', self)
        self._splits = PersistedWork('_splits', self)

    def _add_keys(self, split_name: str, to_populate: Dict[str, str],
                  keys: List[str]):
        to_populate[split_name] = tuple(keys)

    @persisted('_keys_by_split')
    def _get_keys_by_split(self) -> Dict[str, Tuple[str]]:
        """Return keys by split type (i.e. ``train`` vs ``test``) for only those keys
        available by the delegate backing stash.

        """
        logger.debug('creating in memory available keys data structure')
        with time('created key data structures', logging.DEBUG):
            delegate_keys = set(self.delegate.keys())
            avail_kbs = OrderedDict()
            for split, keys in self.split_container.keys_by_split.items():
                ks = list()
                for k in keys:
                    if k in delegate_keys:
                        ks.append(k)
                logger.debug(f'{split} has {len(ks)} keys')
                self._add_keys(split, avail_kbs, ks)
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
        return not isinstance(self.delegate, PreemptiveStash) or \
            self.delegate.has_data

    def deallocate(self):
        if id(self.delegate) != id(self.split_container):
            self._try_deallocate(self.delegate)
        self._try_deallocate(self.split_container)
        self._keys_by_split.deallocate()
        if self._splits.is_set():
            splits = tuple(self._splits().values())
            self._splits.clear()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'deallocating: {len(splits)} stash data splits')
            for v in splits:
                self._try_deallocate(v, recursive=True)
        self._splits.deallocate()
        super().deallocate()

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

    def _get_split_name(self) -> str:
        return self.inst_split_name

    @persisted('_splits')
    def _get_splits(self) -> Dict[str, Stash]:
        """Return an instance of ta stash that contains only the data for a split.

        :param split: the name of the split of the instance to get
                      (i.e. ``train``, ``test``).

        """
        self.prime()
        stashes = OrderedDict()
        for split_name in self.split_names:
            clone = self.__class__(
                delegate=self.delegate, split_container=self.split_container)
            clone._keys_by_split.deallocate()
            clone._splits.deallocate()
            clone.__dict__.update(self.__dict__)
            clone.inst_split_name = split_name
            stashes[split_name] = clone
        return stashes

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line('split stash splits:', depth, writer)
        t = 0
        for ks in self.split_container.keys_by_split.values():
            t += len(ks)
        for k, ks in self.split_container.keys_by_split.items():
            ln = len(ks)
            self._write_line(f'{k}: {ln} ({ln/t*100:.1f}%)',
                             depth + 1, writer)
        self._write_line(f'total: {t}', depth + 1, writer)
        ckc = self.check_key_consistent()
        self._write_line(f'total this instance: {len(self)}', depth, writer)
        self._write_line(f'keys consistent: {ckc}', depth, writer)
        if isinstance(self.delegate, Writable):
            self._write_line('delegate:', depth, writer)
            self.delegate.write(depth + 1, writer)


@dataclass
class SortedDatasetSplitStash(DatasetSplitStash):
    """A sorted version of a :class:`DatasetSplitStash`, where keys, values, items
    and iterations are sorted by key.  This is important for reproducibility of
    results.

    Any shuffling of the dataset, for the sake of training on non-uniform data,
    needs to come before this step.

    *Implementation note:* trying to reuse :class:`zensols.persist.SortedStash`
    would over complicate, so this (minor) functionality overlap is redundant
    in this class.

    """
    ATTR_EXP_META = ('sort_function',)

    sort_function: Callable = field(default=None)
    """A function, such as ``int``, used to sort keys per data set split."""

    def __iter__(self):
        return map(lambda x: (x, self.__getitem__(x),), self.keys())

    def values(self) -> Iterable[Any]:
        return map(lambda k: self.__getitem__(k), self.keys())

    def items(self) -> Tuple[str, Any]:
        return map(lambda k: (k, self.__getitem__(k)), self.keys())

    def _add_keys(self, split_name: str, to_populate: Dict[str, str],
                  keys: List[str]):
        to_populate[split_name] = tuple(sorted(keys, key=self.sort_function))

    def keys(self) -> Iterable[str]:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'sort function: {self.sort_function} ' +
                         f'({self.sort_function})')
        keys = super().keys()
        if self.sort_function is None:
            keys = sorted(keys)
        else:
            keys = sorted(keys, key=self.sort_function)
        return keys
