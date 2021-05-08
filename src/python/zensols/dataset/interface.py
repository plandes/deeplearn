"""Interfaces used for dealing with dataset splits.

"""
__author__ = 'Paul Landes'

import logging
from typing import Dict, Set, Tuple
from dataclasses import dataclass
from abc import abstractmethod, ABCMeta, ABC
from zensols.persist import Stash, PrimeableStash

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
    def _get_keys_by_split(self) -> Dict[str, Tuple[str]]:
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
    def keys_by_split(self) -> Dict[str, Tuple[str]]:
        """Generate a dictionary of split name to keys for that split.  It is expected
        this method will be very expensive.

        """
        return self._get_keys_by_split()


@dataclass
class SplitStashContainer(PrimeableStash, SplitKeyContainer,
                          metaclass=ABCMeta):
    """An interface like ``SplitKeyContainer``, but whose implementations are of
    ``Stash`` containing the instance data.

    For a default implemetnation, see :class:`.DatasetSplitStash`.

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

        :see: :meth:`split_name`

        """
        return self._get_splits()
