"""Utility stashes useful to common machine learning tasks.

"""
__author__ = 'Paul Landes'

import sys
import logging
from typing import Iterable, Dict, Set
from dataclasses import dataclass
from itertools import chain
from collections import OrderedDict
from zensols.util import time
from zensols.config import Writable
from zensols.persist import (
    PersistedWork,
    persisted,
    Stash,
    DelegateStash,
    PreemptiveStash,
)
from zensols.dataset import SplitStashContainer, SplitKeyContainer

logger = logging.getLogger(__name__)


@dataclass
class DatasetSplitStash(DelegateStash, SplitStashContainer, Writable):
    """Generates a separate stash instance for each data set split (i.e. ``train``
    vs ``test).  Each split instance holds the data (keys and values) for each
    split.

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
            avail_kbs = OrderedDict()
            for split, keys in self.split_container.keys_by_split.items():
                ks = keys & delegate_keys
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
        return not isinstance(self.delegate, PreemptiveStash) or \
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
        stashes = OrderedDict()
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
