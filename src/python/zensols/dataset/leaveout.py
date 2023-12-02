"""A split key container for leave-one-out dataset splits.

"""
__author__ = 'Paul Landes'

from typing import Dict, Tuple, Set
from dataclasses import dataclass, field
import sys
import random as rand
import itertools as it
from collections import deque
from pathlib import Path
from zensols.persist import persisted, PersistedWork, Stash
from . import DatasetError, SplitKeyContainer


@dataclass
class LeaveNOutSplitKeyContainer(SplitKeyContainer):
    """A split key container that leaves one out of the dataset.  By default,
    this creates a dataset that has one data point for validation, another for
    test, and the rest of the data for training.

    """
    delegate: Stash = field()
    """The source for keys to generate the splits."""

    distribution: Dict[str, int] = field(
        default_factory=lambda: {'train': -1, 'validation': 1, 'test': 1})
    """The number of data points by each split type.  If the value is an
    integer, that number of data points are used.  Otherwise, if it is a float,
    then that percentage of the entire key set is used.

    """
    shuffle: bool = field(default=True)
    """If ``True``, shuffle the keys obtained from :obj:`delegate` before
    creating the splits.

    """
    path: Path = field(default=None)
    """If not ``None``, persist the keys after shuffling (if enabled) to the
    path specified, for reproducibility of key partitions.

    """
    def __post_init__(self):
        path = '_key_queue' if self.path is None else self.path
        self._key_queue = PersistedWork(path, self, mkdir=True)
        self._iter = 0

    @persisted('_key_queue')
    def _get_key_queue(self) -> deque:
        keys = list(self.delegate.keys())
        if self.shuffle:
            rand.shuffle(keys)
        return deque(keys)

    def next_split(self) -> bool:
        """Create the next split so that the next access to properties such as
        :obj:`keys_by_split` provide the next key split permutation.

        """
        key_queue = self._get_key_queue()
        key_queue.rotate(-1)
        self._iter += 1
        return (self._iter % len(key_queue)) == 0

    @persisted('_split_names')
    def _get_split_names(self) -> Set[str]:
        return frozenset(self.distribution.keys())

    def _get_keys_by_split(self) -> Dict[str, Tuple[str]]:
        kq = self._get_key_queue()
        keys = iter(kq)
        klen = len(kq)
        ds = self.distribution.items()
        filled = False
        by_split = {}
        for name, n in sorted(ds, key=lambda x: x[1], reverse=True):
            if n < 0:
                if filled:
                    raise DatasetError("Distribution has more than one " +
                                       f"'fill' (-1) value: {ds}")
                filled = True
                n = sys.maxsize
            elif isinstance(n, float):
                n = int(n * klen)
            by_split[name] = tuple(it.islice(keys, n))
        total = sum(map(lambda x: len(x), by_split.values()))
        if total != klen:
            raise DatasetError(
                f'Number of allocated keys to the distribution ({total}) ' +
                f'does not equal total keys ({klen})')
        return by_split
