from __future__ import annotations
"""This file contains a stash used to load an embedding layer.  It creates
features in batches of matrices and persists matrix only (sans features) for
efficient retrival.

"""
__author__ = 'Paul Landes'

from typing import Tuple, List, Any, Dict, Union, Set
from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
import sys
import logging
from io import TextIOBase
import torch
from torch import Tensor
import collections
from zensols.util import time
from zensols.config import Writable
from zensols.persist import (
    persisted,
    PersistedWork,
    PersistableContainer,
    Deallocatable,
)
from zensols.deeplearn import DeepLearnError, TorchConfig
from zensols.deeplearn.vectorize import (
    FeatureContext,
    NullFeatureContext,
    FeatureVectorizer,
    FeatureVectorizerManager,
    FeatureVectorizerManagerSet,
    CategoryEncodableFeatureVectorizer,
)
from . import (
    BatchError,
    FieldFeatureMapping,
    ManagerFeatureMapping,
    BatchFeatureMapping,
)

logger = logging.getLogger(__name__)


@dataclass
class DataPoint(Writable, metaclass=ABCMeta):
    """Abstract class that makes up a container class for features created from
    sentences.

    """
    id: int = field()
    """The ID of this data point, which maps back to the ``BatchStash`` instance's
    subordinate stash.

    """

    batch_stash: BatchStash = field(repr=False)
    """Ephemeral instance of the stash used during encoding only."""

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(f'id: {self.id}', depth, writer)

    def __getstate__(self):
        raise DeepLearnError('Data points should not be pickeled')


@dataclass
class Batch(PersistableContainer, Deallocatable, Writable):
    """Contains a batch of data used in the first layer of a net.  This class holds
    the labels, but is otherwise useless without at least one embedding layer
    matrix defined.

    The user must subclass, add mapping meta data, and optionally (suggested)
    add getters and/or properties for the specific data so the model can by
    more *Pythonic* in the PyTorch :class:`torch.nn.Module`.

    """
    STATES = {'n': 'nascent',
              'e': 'encoded',
              'd': 'decoded',
              't': 'memory copied',
              'k': 'deallocated'}
    """A human friendly mapping of the encoded states."""

    batch_stash: BatchStash = field(repr=False)
    """Ephemeral instance of the stash used during encoding and decoding."""

    id: int = field()
    """The ID of this batch instance, which is the sequence number of the batch
    given during child processing of the chunked data point ID setes.

    """

    split_name: str = field()
    """The name of the split for this batch (i.e. ``train`` vs ``test``)."""

    data_points: Tuple[DataPoint] = field(repr=False)
    """The list of the data points given on creation for encoding, and
    ``None``'d out after encoding/pickinglin.

    """

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
            stash: BatchStash = self.batch_stash
            self.data_points = stash._get_data_points_for_batch(self)
        return self.data_points

    @abstractmethod
    def _get_batch_feature_mappings(self) -> BatchFeatureMapping:
        """Return the feature mapping meta data for this batch and it's data points.
        It is best to define a class level instance of the mapping and return
        it here to avoid instancing for each batch.

        :see: :class:`.BatchFeatureMapping`

        """
        pass

    def get_labels(self) -> torch.Tensor:
        """Return the label tensor for this batch.

        """
        bmap: BatchFeatureMapping = self._get_batch_feature_mappings()
        label_attr = bmap.label_attribute_name
        return self.attributes[label_attr]

    @property
    @persisted('_has_labels', transient=True)
    def has_labels(self) -> bool:
        """Return whether or not this batch has labels.  If it doesn't, it is a
        batch used for prediction.

        """
        return self.get_labels() is not None

    def get_label_classes(self) -> List[str]:
        """Return the labels in this batch in their string form.  This assumes the
        label vectorizer is instance of
        :class:`~zensols.deeplearn.vectorize.CategoryEncodableFeatureVectorizer`.

        :return: the reverse mapped, from nominal values, labels

        """
        vec: FeatureVectorizer = self.get_label_feature_vectorizer()
        if not isinstance(vec, CategoryEncodableFeatureVectorizer):
            raise BatchError(
                'Reverse label decoding is only supported with type of ' +
                'CategoryEncodableFeatureVectorizer, but got: ' +
                f'{vec} ({(type(vec))})')
        return vec.get_classes(self.get_labels().cpu())

    def get_label_feature_vectorizer(self) -> FeatureVectorizer:
        """Return the label vectorizer used in the batch.  This assumes there's only
        one vectorizer found in the vectorizer manager.

        :param batch: used to access the vectorizer set via the batch stash

        """
        mapping: BatchFeatureMapping = self._get_batch_feature_mappings()
        field_name: str = mapping.label_attribute_name
        mng, f = mapping.get_field_map_by_attribute(field_name)
        vec_name: str = mng.vectorizer_manager_name
        vec_mng_set = self.batch_stash.vectorizer_manager_set
        vec: FeatureVectorizerManager = vec_mng_set[vec_name]
        return vec[f.feature_id]

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
            shape = None if v is None else v.shape
            self._write_line(f'{k}: {shape}', depth + 2, writer)
        if include_data_points:
            self._write_line('data points:', depth + 1, writer)
            for dp in self.get_data_points():
                dp.write(depth + 2, writer)

    @property
    def _feature_contexts(self) -> \
            Dict[str, Dict[str, Union[FeatureContext, Tuple[FeatureContext]]]]:
        has_ctx = hasattr(self, '_feature_context_inst')
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'has feature contexts: {has_ctx}')
        if has_ctx:
            if self._feature_context_inst is None:
                raise BatchError('Bad state transition, null contexts')
        else:
            with time(f'encoded batch {self.id}'):
                self._feature_context_inst = self._encode()
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'access context: (state={self.state}), num keys=' +
                        f'{len(self._feature_context_inst.keys())}')
        return self._feature_context_inst

    @_feature_contexts.setter
    def _feature_contexts(self,
                          contexts: Dict[str, Dict[
                              str, Union[FeatureContext,
                                         Tuple[FeatureContext]]]]):
        if logger.isEnabledFor(logging.DEBUG):
            obj = 'None' if contexts is None else contexts.keys()
            logger.debug(f'setting context: {obj}')
        self._feature_context_inst = contexts

    def __getstate__(self):
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'create batch state {self.id} (state={self.state})')
        assert self.state == 'n'
        if not hasattr(self, '_feature_context_inst'):
            self._feature_contexts
        self.state = 'e'
        state = super().__getstate__()
        state.pop('batch_stash', None)
        state.pop('data_points', None)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'state keys: {state.keys()}')
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

    @property
    def torch_config(self) -> TorchConfig:
        """The torch config used to copy from CPU to GPU memory."""
        return self.batch_stash.model_torch_config

    def to(self) -> Any:
        """Clone this instance and copy data to the CUDA device configured in the batch
        stash.

        :return: a clone of this instance with all attribute tensors copied
                 to the given torch configuration device

        """
        def to(arr: Tensor) -> Tensor:
            if arr is not None:
                arr = torch_config.to(arr)
            return arr

        if self.state == 't':
            inst = self
        else:
            torch_config = self.torch_config
            attribs = self._get_decoded_state()
            attribs = {k: to(attribs[k]) for k in attribs.keys()}
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
                attrs.clear()
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

        :see: :class:`.BatchFeatureMapping`

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
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'decode with {fm}')
        if isinstance(ctx, tuple):
            arrs = tuple(map(vec.decode, ctx))
            try:
                arr = torch.cat(arrs)
            except Exception as e:
                raise BatchError(
                    'Batch has inconsistent data point length, eg magic ' +
                    f'bedding or using combine_sentences for NLP for: {vec}') \
                    from e
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'decodeed shape for {fm}: {arr.shape}')
        else:
            arr = vec.decode(ctx)
        return arr

    def _is_missing(self, aval: Union[Any, Tuple[Any]]):
        return (aval is None) or \
            (isinstance(aval, (tuple, list)) and all(v is None for v in aval))

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

        :see: meth:`_decode_context`

        """
        vms = self.batch_stash.vectorizer_manager_set
        attrib_to_ctx = collections.OrderedDict()
        bmap: BatchFeatureMapping = self._get_batch_feature_mappings()
        label_attr: str = bmap.label_attribute_name
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"encoding with label: '{label_attr}' using {vms}")
        mmap: ManagerFeatureMapping
        for mmap in bmap.manager_mappings:
            vm: FeatureVectorizerManager = vms[mmap.vectorizer_manager_name]
            fm: FieldFeatureMapping
            for fm in mmap.fields:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'field: {fm}')
                if fm.feature_id in attrib_to_ctx:
                    raise BatchError(f'Duplicate feature: {fm.feature_id}')
                vec = vm[fm.feature_id]
                avals = []
                ctx = None
                dp: DataPoint
                for dp in self.data_points:
                    aval = getattr(dp, fm.attribute_accessor)
                    avals.append(aval)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f'attr: {fm.attr} => {aval.__class__}')
                try:
                    is_label = fm.is_label or (label_attr == fm.attr)
                    if is_label and self._is_missing(aval):
                        # assume prediction
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug('skipping missing label')
                        ctx = NullFeatureContext(fm.feature_id)
                    else:
                        ctx = self._encode_field(vec, fm, avals)
                except Exception as e:
                    raise BatchError(
                        f'Could not vectorize {fm} using {vec}') from e
                if ctx is not None:
                    attrib_to_ctx[fm.attr] = ctx
        return attrib_to_ctx

    def _decode(self, ctx: Dict[str, Dict[str, Union[FeatureContext,
                                                     Tuple[FeatureContext]]]]):
        """Called to create all matrices/arrays needed for the layer.  After this is
        called, features in this instance are removed for so pickling is fast.

        :param ctx: the context to decode

        """
        attribs = collections.OrderedDict()
        attrib_keeps: Set[str] = self.batch_stash.decoded_attributes
        if attrib_keeps is not None:
            attrib_keeps = set(attrib_keeps)
        bmap: BatchFeatureMapping = self._get_batch_feature_mappings()
        label_attr: str = bmap.label_attribute_name
        vms: FeatureVectorizerManagerSet = \
            self.batch_stash.vectorizer_manager_set
        mmap: ManagerFeatureMapping
        for attrib, ctx in ctx.items():
            mng_fmap = bmap.get_field_map_by_attribute(attrib)
            if mng_fmap is None:
                raise BatchError(
                    f'Missing mapped attribute \'{attrib}\' on decode')
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
            elif ctx is None:
                feature_id = None
            else:
                feature_id = ctx.feature_id
            vec: FeatureVectorizer = vm.get(feature_id)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'decoding {ctx} with {vec}')
            arr = self._decode_context(vec, ctx, fmap)
            if arr is None and fmap.attr != label_attr:
                raise BatchError(
                    f'No decoded value for {fmap}, which is not ' +
                    f"the label attribute '{label_attr}'")
            if logger.isEnabledFor(logging.DEBUG):
                shape = '<none>' if arr is None else arr.shape
                logger.debug(f'decoded: {attrib} -> {shape}')
            if attrib in attribs:
                raise BatchError(
                    f'Attribute collision on decode: {attrib}')
            attribs[attrib] = arr
        if attrib_keeps is not None and len(attrib_keeps) > 0:
            raise BatchError(f'Unknown attriubtes: {attrib_keeps}')
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
