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
from io import TextIOBase
import torch
import collections
from zensols.util import time
from zensols.config import Writable
from zensols.persist import (
    persisted,
    PersistedWork,
    PersistableContainer,
    Deallocatable,
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
class DataPoint(Writable, metaclass=ABCMeta):
    """Abstract class that makes up a container class for features created from
    sentences.

    :param id: the ID of this data point, which maps back to the ``BatchStash``
               instance's subordinate stash

    :param batch_stash: ephemeral instance of the stash used during encoding
                        only

    """
    id: int
    batch_stash: BatchStash = field(repr=False)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(f'id: {id}', depth, writer)


@dataclass
class Batch(PersistableContainer, Deallocatable, Writable):
    """Contains a batch of data used in the first layer of a net.  This class holds
    the labels, but is otherwise useless without at least one embedding layer
    matrix defined.

    The user must subclass, add mapping meta data, and optionally (suggested)
    add getters and/or properties for the specific data so the model can by
    more *Pythonic* in the PyTorch :class:`torch.nn.Module`.

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
    STATES = {'n': 'nascent',
              'e': 'encoded',
              'd': 'decoded',
              't': 'memory copied',
              'k': 'deallocated'}
    batch_stash: BatchStash = field(repr=False)
    id: int
    split_name: str
    data_points: Tuple[DataPoint] = field(repr=False)

    def __post_init__(self):
        super().__init__()
        if self.data_points is not None:
            self.data_point_ids = tuple(map(lambda d: d.id, self.data_points))
        self._decoded_state = PersistedWork(
            '_decoded_state', self, transient=True)
        self.state = 'n'

    def get_data_points(self) -> Tuple[DataPoint]:
        """Return the data points used to create this batch.  If the batch does not
        contain the data points (it has been decoded), then they are retrieved
        from the :obj:`batch_stash` instance's feature stash.

        """
        if not hasattr(self, 'data_points') or self.data_points is None:
            self.data_points = self.batch_stash._get_data_points_for_batch(self)
        return self.data_points

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

    def get_label_classes(self) -> List[str]:
        """Return the vectorizer that encodes labels.

        """
        vec = self.batch_stash.get_label_feature_vectorizer(self)
        return vec.get_classes(self.get_labels().cpu())

    def size(self) -> int:
        """Return the size of this batch, which is the number of data points.

        """
        return len(self.data_point_ids)

    @property
    def attributes(self) -> Dict[str, torch.Tensor]:
        """Return the attribute batched tensors as a dictionary using the attribute
        names as the keys.

        """
        return self._get_decoded_state()

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              include_data_points: bool = False):
        self._write_line(self.__class__.__name__, depth, writer)
        self._write_line(f'size: {self.size()}', depth + 1, writer)
        for k, v in self.attributes.items():
            self._write_line(f'{k}: {v.shape}', depth + 2, writer)
        if include_data_points:
            self._write_line('data points:', depth + 1, writer)
            for dp in self.get_data_points():
                dp.write(depth + 2, writer)

    @property
    def _feature_contexts(self):
        has_ctx = hasattr(self, '_feature_context_inst')
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'has feature contexts: {has_ctx}')
        if has_ctx:
            if self._feature_context_inst is None:
                raise ValueError('bad state transition, null contexts')
        else:
            with time(f'encoded batch {self.id}'):
                self._feature_context_inst = self._encode()
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'access context: (state={self.state}), num keys=' +
                        f'{len(self._feature_context_inst.keys())}')
        return self._feature_context_inst

    @_feature_contexts.setter
    def _feature_contexts(self, contexts):
        if logger.isEnabledFor(logging.DEBUG):
            obj = 'None' if contexts is None else contexts.keys()
            logger.debug(f'setting context: {obj}')
        self._feature_context_inst = contexts

    def __getstate__(self):
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'pickling batch {self.id} (state={self.state})')
        assert self.state == 'n'
        if not hasattr(self, '_feature_context_inst'):
            self._feature_contexts
        self.state = 'e'
        state = super().__getstate__()
        state.pop('batch_stash', None)
        state.pop('data_points', None)
        if logger.isEnabledFor(logging.DEBUG):
            logger(f'state keys: {state.keys()}')
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'unpickling batch: {self.id}')

    @persisted('_decoded_state')
    def _get_decoded_state(self):
        """Decode the pickeled attriubtes after loaded by containing ``BatchStash`` and
        remove the context information to save memory.

        """
        assert self.state == 'e'
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'decoding ctxs: {self._feature_context_inst.keys()}')
        assert self._feature_context_inst is not None
        with time(f'decoded batch {self.id}'):
            attribs = self._decode(self._feature_contexts)
        self._feature_contexts = None
        assert self._feature_context_inst is None
        self.state = 'd'
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'return decoded attributes: {attribs.keys()}')
        return attribs

    def to(self) -> Any:
        """Clone this instance and copy data to the CUDA device configured in the batch
        stash.

        :return: a clone of this instance with all attribute tensors copied
                 to the given torch configuration device

        """
        if self.state == 't':
            inst = self
        else:
            torch_config = self.batch_stash.model_torch_config
            attribs = self._get_decoded_state()
            attribs = {k: torch_config.to(attribs[k]) for k in attribs.keys()}
            inst = self.__class__(
                self.batch_stash, self.id, self.split_name, None)
            inst.data_point_ids = self.data_point_ids
            inst._decoded_state.set(attribs)
            inst.state = 't'
        return inst

    def deallocate(self):
        with time('deallocated attribute', logging.DEBUG):
            if self.state == 'd' or self.state == 't':
                attrs = self.attributes
                for arr in tuple(attrs.values()):
                    del arr
                del attrs
            self._decoded_state.deallocate()
        if hasattr(self, 'batch_stash'):
            del self.batch_stash
        if hasattr(self, 'data_point_ids'):
            del self.data_point_ids
        if hasattr(self, 'data_points'):
            del self.data_points
        with time('deallocated feature context', logging.DEBUG):
            if hasattr(self, '_feature_context_inst') and \
               self._feature_context_inst is not None:
                for ctx in self._feature_context_inst.values():
                    self._try_deallocate(ctx)
                self._feature_context_inst.clear()
                del self._feature_context_inst
        self.state = 'k'
        super().deallocate()
        logger.debug(f'deallocated batch: {self.id}')

    def _encode_field(self, vec: FeatureVectorizer, fm: FieldFeatureMapping,
                      vals: List[Any]) -> FeatureContext:
        """Encode a set of features in to feature contexts:

        :param vec: the feature vectorizer to use to create the context

        :param fm: the field metadata for the feature values

        :param vals: a list of feature input values used to create the context

        :see BatchFeatureMapping:
        """
        if fm.is_agg:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'encoding aggregate with {vec}')
            ctx = vec.encode(vals)
        else:
            ctx = tuple(map(lambda v: vec.encode(v), vals))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'encoded: {ctx.__class__}')
        return ctx

    def _decode_context(self, vec: FeatureVectorizer, ctx: FeatureContext,
                        fm: FieldFeatureMapping) -> torch.Tensor:
        """Decode ``ctx`` in to a tensor using vectorizer ``vec``.

        """
        if isinstance(ctx, tuple):
            arrs = map(vec.decode, ctx)
            if fm.add_dim is not None:
                arrs = map(lambda v: v.unsqueeze(fm.add_dim), arrs)
            arr = torch.cat(tuple(arrs))
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'decodeed shape for {fm}: {arr.shape}')
        else:
            arr = vec.decode(ctx)
        return arr

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
        ``_decode_context``.  Note that the feature id is an attribute of the
        feature context.

        :see _decode_context:

        """
        vms = self.batch_stash.vectorizer_manager_set
        attrib_to_ctx = collections.OrderedDict()
        bmap = self._get_batch_feature_mappings()
        mmap: ManagerFeatureMapping
        for mmap in bmap.manager_mappings:
            vm: FeatureVectorizerManager = vms[mmap.vectorizer_manager_name]
            fm: FieldFeatureMapping
            for fm in mmap.fields:
                if fm.feature_id in attrib_to_ctx:
                    raise ValueError(f'duplicate feature: {fm.feature_id}')
                vec = vm[fm.feature_id]
                avals = []
                dp: DataPoint
                ctx = None
                for dp in self.data_points:
                    aval = getattr(dp, fm.attribute_accessor)
                    avals.append(aval)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f'attr: {fm.attr} => {aval.__class__}')
                ctx = self._encode_field(vec, fm, avals)
                if ctx is not None:
                    attrib_to_ctx[fm.attr] = ctx
        return attrib_to_ctx

    def _decode(self, ctx: Dict[str, Dict[str, Union[FeatureContext,
                                                     Tuple[FeatureContext]]]]):
        """Called to create all matrices/arrays needed for the layer.  After this is
        called, features in this instance are removed for so pickling is fast.

        """
        attribs = collections.OrderedDict()
        attrib_keeps = self.batch_stash.decoded_attributes
        if attrib_keeps is not None:
            attrib_keeps = set(attrib_keeps)
        bmap = self._get_batch_feature_mappings()
        vms = self.batch_stash.vectorizer_manager_set
        mmap: ManagerFeatureMapping
        for attrib, ctx in ctx.items():
            mng_fmap = bmap.get_field_map_by_attribute(attrib)
            if mng_fmap is None:
                raise ValueError(
                    f'missing mapped attribute \'{attrib}\' on decode')
            mng, fmap = mng_fmap
            mmap_name = mng.vectorizer_manager_name
            feature_id = fmap.feature_id
            vm: FeatureVectorizerManager = vms[mmap_name]
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'vec manager: {mmap_name} -> {vm}')
            # keep only the desired feature subset for speed up
            if attrib_keeps is not None and attrib not in attrib_keeps:
                continue
            if attrib_keeps is not None:
                attrib_keeps.remove(attrib)
            if isinstance(ctx, tuple):
                feature_id = ctx[0].feature_id
            else:
                feature_id = ctx.feature_id
            vec = vm[feature_id]
            arr = self._decode_context(vec, ctx, fmap)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'decoded: {attrib} -> {arr.shape}')
            if attrib in attribs:
                raise ValueError(
                    f'attribute collision on decode: {attrib}')
            attribs[attrib] = arr
        if attrib_keeps is not None and len(attrib_keeps) > 0:
            raise ValueError(f'unknown attriubtes: {attrib_keeps}')
        return attribs

    @property
    def state_name(self):
        return self.STATES.get(self.state, 'unknown: ' + self.state)

    def __len__(self):
        return len(self.data_point_ids)

    def keys(self) -> Tuple[str]:
        return tuple(self.attributes.keys())

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.attributes[key]

    def __str__(self):
        return f'{super().__str__()}: size: {self.size()}, state={self.state}'

    def __repr__(self):
        return self.__str__()
