"""This file contains a stash used to load an embedding layer.  It creates
features in batches of matrices and persists matrix only (sans features) for
efficient retrival.

"""
__author__ = 'Paul Landes'

import logging
from typing import Tuple, List, Any, Dict, Union, Set, Iterable
import torch
from dataclasses import dataclass, field
import itertools as it
import collections
from abc import ABCMeta, abstractmethod
import numpy as np
from pathlib import Path
from zensols.config import Configurable
from zensols.persist import (
    chunks,
    persisted,
    PersistedWork,
    PersistableContainer
)
from zensols.multi import MultiProcessStash
from zensols.deeplearn import (
    FeatureContext,
    FeatureVectorizer,
    FeatureVectorizerManager,
    FeatureVectorizerManagerSet,
    SplitKeyContainer,
    SplitStashContainer,
    TorchConfig,
)

from multiprocessing import Pool
from typing import Callable

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
    data_point_type: type
    batch_type: type
    split_stash_container: SplitStashContainer
    vectorizer_manager_set: FeatureVectorizerManagerSet
    torch_config: TorchConfig
    data_point_id_sets: Path
    batch_size: int
    data_point_id_set_limit: int

    def __post_init__(self):
        super().__post_init__()
        self.data_point_id_sets.parent.mkdir(parents=True, exist_ok=True)
        self._batch_data_point_sets = PersistedWork(
            self.data_point_id_sets, self)
        self.priming = False

    def _invoke_pool(self, pool: Pool, fn: Callable, data: iter) -> int:
        m = pool.imap_unordered(fn, data)
        return tuple(m)

    @property
    @persisted('_batch_data_point_sets')
    def batch_data_point_sets(self) -> List[DataPointIDSet]:
        psets = []
        batch_id = 0
        for split, keys in self.split_stash_container.keys_by_split.items():
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
        return it.islice(self.batch_data_point_sets,
                         self.data_point_id_set_limit)

    def reconstitute_batch(self, batch: Any) -> Any:
        data_point_id: str
        dpcls = self.data_point_type
        cont = self.split_stash_container
        points = tuple(map(lambda dpid: dpcls(dpid, self, cont[dpid]),
                           batch.data_point_ids))
        bcls = self.batch_type
        return bcls(self, batch.id, batch.split_name, points)

    def _process(self, chunk: List[DataPointIDSet]) -> \
            Iterable[Tuple[str, Any]]:
        #return tuple(map(lambda d: (d.batch_id, d), chunk))
        logger.debug(f'processing: {chunk} {type(chunk)}')
        dpcls = self.data_point_type
        bcls = self.batch_type
        cont = self.split_stash_container
        batches: List[Batch] = []
        dpid_set: DataPointIDSet
        points: Tuple[DataPoint]
        batch: Batch
        for dset in chunk:
            points = tuple(map(lambda dpid: dpcls(dpid, self, cont[dpid]),
                               dset.data_point_ids))
            batch = bcls(self, dset.batch_id, dset.split_name, points)
            batches.append((dset.batch_id, batch))
        return batches

    def load(self, name: str):
        obj = super().load(name)
        # add back the container of the batch to reconstitute the original
        # features and use the CUDA for tensor device transforms
        if obj is not None and not hasattr(obj, 'batch_stash'):
            obj.batch_stash = self
        return obj

    def prime(self):
        logger.debug(f'priming {self.__class__}, is child: {self.is_child}, ' +
                     f'currently priming: {self.priming}')
        if not self.priming:
            self.priming = True
            try:
                self.batch_data_point_sets
                super().prime()
            finally:
                self.priming = False

    def clear(self):
        logger.debug('clear: calling super')
        super().clear()
        self._batch_data_point_sets.clear()


@dataclass
class FieldFeatureMapping(object):
    attr: str
    feature_type: str
    is_agg: bool = field(default=False)


@dataclass
class ManagerFeatureMapping(object):
    vectorizer_manager_name: str
    fields: Tuple[FieldFeatureMapping]


@dataclass
class BatchFeatureMapping(object):
    label_feature_type: str
    manager_mappings: List[ManagerFeatureMapping]


@dataclass
class DataPoint(metaclass=ABCMeta):
    """Abstract class that makes up a container class for features created from
    sentences.

    The ``get_label_matrices`` method needs an implementation in subclasses.

    """
    id: int
    batch_stash: BatchStash

    # @abstractmethod
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
    split_name: str
    data_points: Tuple[DataPoint] = field(repr=False)

    def __post_init__(self):
        if self.data_points is not None:
            self.data_point_ids = tuple(map(lambda d: d.id, self.data_points))

    @abstractmethod
    def _get_batch_feature_mappings(self) -> BatchFeatureMapping:
        pass

    def get_labels(self) -> torch.Tensor:
        label_attr = self._get_batch_feature_mappings().label_feature_type
        return self.attributes[label_attr]

    def _encode_field(self, vec: FeatureVectorizer, fm: FieldFeatureMapping,
                      vals: List[Any]) -> FeatureContext:
        if fm.is_agg:
            logger.debug(f'encoding aggregate with {vec}')
            ctx = vec.encode(vals)
        else:
            ctx = tuple(map(lambda v: vec.encode(v), vals))
        return ctx

    def _encode(self):
        """Called to create all matrices/arrays needed for the layer.  After this is
        called, features in this instance are removed for so pickling is fast.

        """
        vms = self.batch_stash.vectorizer_manager_set
        by_manager = {}
        fnames = set()
        bmap = self._get_batch_feature_mappings()
        mmap: ManagerFeatureMapping
        for mmap in bmap.manager_mappings:
            by_vec = {}
            vm: FeatureVectorizerManager = vms[mmap.vectorizer_manager_name]
            fm: FieldFeatureMapping
            for fm in mmap.fields:
                if fm.feature_type in fnames:
                    raise ValueError(f'duplicate feature name: {fm.feature_type}')
                fnames.add(fm.feature_type)
                vec = vm[fm.feature_type]
                avals = []
                dp: DataPoint
                ctx = None
                for dp in self.data_points:
                    aval = getattr(dp, fm.attr)
                    avals.append(aval)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f'attr: {fm.attr} => {aval.__class__}')
                ctx = self._encode_field(vec, fm, avals)
                if ctx is not None:
                    by_vec[fm.attr] = ctx
            if len(by_vec) > 0:
                by_manager[mmap.vectorizer_manager_name] = by_vec
        # from pprint import pprint
        # pprint(by_manager)
        return by_manager

    def _decode_context(self, vec: FeatureVectorizer, ctx: FeatureContext):
        if isinstance(ctx, tuple):
            arrs = tuple(map(vec.decode, ctx))
            arr = torch.cat(arrs)
        else:
            arr = vec.decode(ctx)
        return arr

    def _decode(self, ctx: Dict[str, Dict[str, Union[FeatureContext,
                                                     Tuple[FeatureContext]]]]):
        """Called to create all matrices/arrays needed for the layer.  After this is
        called, features in this instance are removed for so pickling is fast.

        """
        attribs = {}
        feats = {}
        vms = self.batch_stash.vectorizer_manager_set
        mmap: ManagerFeatureMapping
        for mmap_name, feature_ctx in ctx.items():
            vm: FeatureVectorizerManager = vms[mmap_name]
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'mng: {mmap_name} -> {vm}')
            for attrib, ctx in feature_ctx.items():
                if isinstance(ctx, tuple):
                    feature_type = ctx[0].feature_type
                else:
                    feature_type = ctx.feature_type
                vec = vm[feature_type]
                arr = self._decode_context(vec, ctx)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'decoded: {attrib} -> {arr.shape}')
                if attrib in feats:
                    raise ValueError(f'feature name collision on decode: {attrib}')
                if feature_type in feats:
                    raise ValueError(f'feature name collision on decode: {feature_type}')
                attribs[attrib] = arr
                feats[feature_type] = arr
        return attribs, feats

    def __getstate__(self):
        ctx = self._encode()
        state = super().__getstate__()
        state.pop('batch_stash', None)
        state.pop('data_points', None)
        state['ctx'] = ctx
        return state

    @persisted('_decoded_state', transient=True)
    def _get_decoded_state(self):
        attribs, feats = self._decode(self.ctx)
        delattr(self, 'ctx')
        return attribs, feats

    @property
    def attributes(self):
        return self._get_decoded_state()[0]

    @property
    def features(self):
        return self._get_decoded_state()[1]
