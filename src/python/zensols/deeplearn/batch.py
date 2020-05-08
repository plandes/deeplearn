"""This file contains a stash used to load an embedding layer.  It creates
features in batches of matrices and persists matrix only (sans features) for
efficient retrival.

"""
__author__ = 'Paul Landes'

import logging
import traceback
from typing import List, Any, Dict, Iterable, Tuple, Set
from dataclasses import dataclass, field
import itertools as it
from itertools import chain
import collections
import copy as cp
from abc import ABCMeta, abstractmethod
import numpy as np
from pathlib import Path
import torch
from zensols.deeplearn import TorchConfig
from zensols.util import time
from zensols.config import Configurable
from zensols.persist import (
    chunks,
    persisted,
    PersistedWork,
    PersistableContainer
)
from zensols.multi import MultiProcessStash
from zensols.deeplearn import (
    FeatureVectorizerManager,
    FeatureVectorizerManagerSet,
    SplitStashContainer,
    SplitKeyContainer,
)

logger = logging.getLogger(__name__)


@dataclass
class DataPointIDSet(object):
    batch_id: str
    data_point_ids: Set[str]
    split_name: str

    def __str__(self):
        return (f'{self.batch_id}: split={self.split_name}, ' +
                f'num keys: {len(self.data_point_ids)}')


@dataclass
class BatchStash(MultiProcessStash, SplitKeyContainer, metaclass=ABCMeta):
    ATTR_EXP_META = ('data_point_type',)

    config: Configurable
    name: str
    #data_point_type: type
    split_stashes: SplitStashContainer
    vec_managers: FeatureVectorizerManagerSet
    data_point_id_sets: Path
    batch_size: int
    data_point_id_set_limit: int

    def __post_init__(self):
        super().__post_init__()
        self._batch_data_point_sets = PersistedWork(
            self.data_point_id_sets, self)

    @property
    @persisted('_batch_data_point_sets')
    def batch_data_point_sets(self) -> List[DataPointIDSet]:
        psets = []
        batch_id = 0
        for split, keys in self.split_stashes.keys_by_split.items():
            logger.debug(f'keys for {split}: {len(keys)}')
            for chunk in chunks(keys, self.batch_size):
                psets.append(DataPointIDSet(batch_id, tuple(chunk), split))
                batch_id += 1
        return psets

    def _get_keys_by_split(self) -> Dict[str, Set[str]]:
        by_set = collections.defaultdict(lambda: set())
        for dps in self.batch_data_point_sets:
            by_set[dps.split_name].add(dps.batch_id)
        return by_set

    def _create_data(self) -> List[DataPointIDSet]:
        return it.islice(map(lambda s: (s.batch_id, s),
                             self.batch_data_point_sets),
                         self.data_point_id_set_limit)

    def _process(self, chunk: List[DataPointIDSet]) -> \
            Iterable[Tuple[str, Any]]:
        print(f'process: {chunk}')
        return chunk

    def load(self, name: str):
        obj = super().load(name)
        # add back the container of the batch to reconstitute the original
        # features and use the CUDA for tensor device transforms
        if obj is not None and not hasattr(obj, 'batch_stash'):
            obj.batch_stash = self
        return obj

    def prime(self):
        self.batch_data_point_sets
        super().prime()

    def clear(self):
        super().clear()
        self._batch_data_point_sets.clear()


@dataclass
class DataPoint(metaclass=ABCMeta):
    """Abstract class that makes up a container class for features created from
    sentences.

    The ``get_label_matrices`` method needs an implementation in subclasses.

    """
    id: int
    batch_stash: BatchStash

    @abstractmethod
    def get_label_matrices(self) -> np.ndarray:
        """Return the labels for the data points.  This will be a singleton unless the
        data point expands.

        """
        pass

    def __str__(self):
        return f'{self.id}: labels: {self.get_label_matrices()}'

    def __repr__(self):
        return self.__str__()


@dataclass
class Batch(PersistableContainer):
    """Contains a batch of data used in the first layer of a net.  This class holds
    the labels, but is otherwise useless without at least one embedding layer
    matrix defined.

    """
    batch_stash: BatchStash = field(repr=False)
    id: int
    split_type: str
    data_points: List[DataPoint] = field(repr=False)

    def __post_init__(self):
        if self.data_points is not None:
            self.data_point_ids = tuple(map(lambda d: d.id, self.data_points))
