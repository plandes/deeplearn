"""This file contains a stash used to load an embedding layer.  It creates
features in batches of matrices and persists matrix only (sans features) for
efficient retrival.

"""
__author__ = 'Paul Landes'

from typing import Tuple, List, Any, Dict, Union
from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
import sys
import logging
from io import TextIOWrapper
import torch
import collections
from zensols.util import time
from zensols.config import Writable
from zensols.persist import (
    persisted,
    PersistedWork,
    PersistableContainer
)
from zensols.deeplearn.vectorize import (
    FeatureContext,
    FeatureVectorizer,
    FeatureVectorizerManager,
)
from zensols.deeplearn.batch import (
    BatchStash,
    FieldFeatureMapping,
    ManagerFeatureMapping,
    BatchFeatureMapping,
)

logger = logging.getLogger(__name__)


@dataclass
class DataPoint(metaclass=ABCMeta):
    """Abstract class that makes up a container class for features created from
    sentences.

    :param id: the ID of this data point, which maps back to the ``BatchStash``
               instance's subordinate stash

    :param batch_stash: ephemeral instance of the stash used during encoding
                        only

    """
    id: int
    batch_stash: BatchStash


@dataclass
class Batch(PersistableContainer, Writable):
    """Contains a batch of data used in the first layer of a net.  This class holds
    the labels, but is otherwise useless without at least one embedding layer
    matrix defined.

    The user must subclass, add mapping meta data, and optionally (suggested)
    add getters and/or properties for the specific data so the model can by
    more *Pythonic* in the PyTorch ``nn.Module``.

    :param batch_stash: ephemeral instance of the stash used during
                        encoding and decoding

    :param id: the ID of this batch instance, which is the sequence number of
               the batch given during child processing of the chunked data
               point ID setes

    :param split_name: the name of the split for this batch (i.e. ``train`` vs
                       ``test``)

    :param data_points: the list of the data points given on creation for
                        encoding, and ``None``'d out after encoding/pickinglin

    :param data_point_ids: populated on instance creation and pickled along
                           with the class

    """
    batch_stash: BatchStash = field(repr=False)
    id: int
    split_name: str
    data_points: Tuple[DataPoint] = field(repr=False)

    def __post_init__(self):
        if self.data_points is not None:
            self.data_point_ids = tuple(map(lambda d: d.id, self.data_points))
        self._decoded_state = PersistedWork(
            '_decoded_state', self, transient=True)
        self.state = 'n'

    @abstractmethod
    def _get_batch_feature_mappings(self) -> BatchFeatureMapping:
        """Return the feature mapping meta data for this batch and it's data points.
        It is best to define a class level instance of the mapping and return
        it here to avoid instancing for each batch.

        :see BatchFeatureMapping:

        """
        pass

    def get_labels(self) -> torch.Tensor:
        """Return the label tensor for this batch.

        """
        label_attr = self._get_batch_feature_mappings().label_attribute_name
        return self.attributes[label_attr]

    def size(self) -> int:
        """Return the size of this batch, which is the number of data points.

        """
        return len(self.data_point_ids)

    @property
    def attributes(self) -> Dict[str, torch.Tensor]:
        """Return the attribute batched tensors as a dictionary using the attribute
        names as the keys.

        """
        return self._get_decoded_state()[0]

    @property
    def feature_types(self) -> Dict[str, str]:
        """Return a mapping from available feature name to attribute name.

        """
        return self._get_decoded_state()[1]

    def write(self, depth: int = 0, writer: TextIOWrapper = sys.stdout):
        self._write_line(self.__class__.__name__, depth, writer)
        self._write_line(f'size: {self.size()}', depth + 1, writer)
        for k, v in self.attributes.items():
            self._write_line(f'{k}: {v.shape}', depth + 2, writer)

    def __getstate__(self):
        if self.state != 'n':
            raise ValueError(
                f'expecting nascent state, but state is {self.state}')
        with time(f'encoded batch {self.id}'):
            ctx = self._encode()
        self.state = 'e'
        state = super().__getstate__()
        state.pop('batch_stash', None)
        state.pop('data_points', None)
        state['ctx'] = ctx
        return state

    @persisted('_decoded_state')
    def _get_decoded_state(self):
        """Decode the pickeled attriubtes after loaded by containing ``BatchStash`` and
        remove the context information to save memory.

        """
        if self.state != 'e':
            raise ValueError(
                f'expecting enocded state, but state is {self.state}')
        with time(f'decoded batch {self.id}'):
            attribs, feats = self._decode(self.ctx)
        delattr(self, 'ctx')
        self.state = 'd'
        return attribs, feats

    def to(self) -> Any:
        """Clone this instance and copy data to the CUDA device configured in the batch
        stash.

        :return: a clone of this instance with all attribute tensors copied
                 to the given torch configuration device

        """
        torch_config = self.batch_stash.model_torch_config
        attribs, feats = self._get_decoded_state()
        attribs = {k: torch_config.to(attribs[k]) for k in attribs.keys()}
        inst = self.__class__(self.batch_stash, self.id, self.split_name, None)
        inst.data_point_ids = self.data_point_ids
        inst._decoded_state.set((attribs, feats))
        inst.state = 't'
        return inst

    def _encode_field(self, vec: FeatureVectorizer, fm: FieldFeatureMapping,
                      vals: List[Any]) -> FeatureContext:
        """Encode a set of features in to feature contexts:

        :param vec: the feature vectorizer to use to create the context
        :param fm: the field metadata for the feature values
        :param vals: a list of feature input values used to create the context

        :see BatchFeatureMapping:
        """
        if fm.is_agg:
            logger.debug(f'encoding aggregate with {vec}')
            ctx = vec.encode(vals)
        else:
            ctx = tuple(map(lambda v: vec.encode(v), vals))
        logger.debug(f'encoded: {ctx.__class__}')
        return ctx

    def _encode(self) -> Dict[str, Dict[str, Union[FeatureContext,
                                                   Tuple[FeatureContext]]]]:
        """Called to create all matrices/arrays needed for the layer.  After this is
        called, features in this instance are removed for so pickling is fast.

        The returned data structure has the following form:

        - feature vector manager name
          - attribute name -> feature context

        where feature context can be either a single context or a tuple of
        context.  If it is a tuple, then each is decoded in turn and the
        resulting matrices will be concatenated together at decode time with
        ``_decode_context``.  Note that the feature type is an attribute of the
        feature context.

        :see _decode_context:

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
        return by_manager

    def _decode_context(self, vec: FeatureVectorizer, ctx: FeatureContext) \
            -> torch.Tensor:
        """Decode ``ctx`` in to a tensor using vectorizer ``vec``.

        """
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
        attribs = collections.OrderedDict()
        feats = collections.OrderedDict()
        attrib_keeps = self.batch_stash.decoded_attributes
        vms = self.batch_stash.vectorizer_manager_set
        mmap: ManagerFeatureMapping
        for mmap_name, feature_ctx in ctx.items():
            vm: FeatureVectorizerManager = vms[mmap_name]
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'mng: {mmap_name} -> {vm}')
            for attrib, ctx in feature_ctx.items():
                # keep only the desired feature subset for speed up
                if attrib_keeps is not None and attrib not in attrib_keeps:
                    continue
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
                feats[feature_type] = attrib
        return attribs, feats

    def __str__(self):
        return f'{super().__str__()}: state={self.state}'

    def __repr__(self):
        return self.__str__()
