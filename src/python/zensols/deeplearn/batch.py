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

logger = logging.getLogger(__name__)


@dataclass
class DataPointIDSet(object):
    """Set of subordinate stash IDs with feature values to be vectorized with
    ``BatchStash``.  Groups of these are sent to subprocesses for processing in
    to ``Batch`` instances.

    """
    batch_id: str
    data_point_ids: Set[str]
    split_name: str

    def __str__(self):
        return (f'{self.batch_id}: split={self.split_name}, ' +
                f'num keys: {len(self.data_point_ids)}')


@dataclass
class BatchStash(MultiProcessStash, SplitKeyContainer, metaclass=ABCMeta):
    """A stash that vectorizes features in to easily consumable tensors for
    training and testing.  This stash produces instances of ``Batch``, which is
    a batch in the machine learning sense, and the first dimension of what will
    become the tensor used in PyTorch.  Each of these batches has a logical one
    to many relationship to that batche's respective set of data points, which
    is encapsulated in the ``DataPoint`` class.

    The stash creates subprocesses to vectorize features in to tensors in
    chunks of IDs (data point IDs) from the subordinate stash using
    ``DataPointIDSet`` instances.

    The lifecycle of the data follows:

    1. Feature data created by the client, which could be language features,
       row data etc.

    2. Vectorize the feature data using the vectorizers in
       ``vectorizer_manager_set``.  This creates the feature contexts
       (``FeatureContext``) specifically meant to be pickeled.

    3. Pickle the feature contexts when dumping to disk, which is invoked in
       the child processes of this class.

    4. At train time, load the feature contexts from disk.

    5. Decode the feature contexts in to PyTorch tensors.

    6. The model manager uses the ``to`` method to copy the CPU tensors to the
       GPU (where GPUs are available).

    :see _process: for details on the pickling of the batch instances

    """
    ATTR_EXP_META = ('data_point_type',)

    config: Configurable
    name: str
    data_point_type: type
    batch_type: type
    split_stash_container: SplitStashContainer
    vectorizer_manager_set: FeatureVectorizerManagerSet
    decoded_attributes: Set[int]
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
        """Data created for the sub proceesses are the first N data point ID sets.

        """
        return it.islice(self.batch_data_point_sets,
                         self.data_point_id_set_limit)

    def _process(self, chunk: List[DataPointIDSet]) -> \
            Iterable[Tuple[str, Any]]:
        """Create the batches by creating the set of data points for each
        ``DataPointIDSet`` instance.  When the subordinate stash dumps the
        batch (specifically a subclass of ``Batch), the overrided pickle logic
        is used to *detatch* the batch by encoded all data in to
        ``EncodedFeatureContext`` instances.

        """
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

    def reconstitute_batch(self, batch: Any) -> Any:
        """Return a new instance of a batch, which is some subclass of ``Batch``, with
        instances of it's respective data points repopulated.  This is useful
        after a batch is decoded and the original data point data is needed.

        """
        data_point_id: str
        dpcls = self.data_point_type
        cont = self.split_stash_container
        points = tuple(map(lambda dpid: dpcls(dpid, self, cont[dpid]),
                           batch.data_point_ids))
        bcls = self.batch_type
        return bcls(self, batch.id, batch.split_name, points)

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
    """Meta data describing an attribute of the data point.

    :attributes attr: the attribute name, which is used to identify the
                      feature that is vectorized
    :attribute feature_type: indicates which vectorizer to use

    :attribute is_agg: if ``True``, tuplize across all data points and encode
                       as one tuple of data to create the batched tensor on
                       decode; otherwise, each data point feature is encoded
                       and concatenated on decode

    """
    attr: str
    feature_type: str
    is_agg: bool = field(default=False)


@dataclass
class ManagerFeatureMapping(object):
    """Meta data for a vectorizer manager with fields describing attributes to be
    vectorized from features in to feature contests.

    :attribute vectorizer_manager_name: the configuration name that identifiees
                                        an instance of ``FeatureVectorizerManager``
    :attribute field: the fields of the data point to be vectorized
    """
    vectorizer_manager_name: str
    fields: Tuple[FieldFeatureMapping]


@dataclass
class BatchFeatureMapping(object):
    """The meta data used to encode and decode each feature in to tensors.  It is
    best to define a class level instance of this in the ``Batch`` class and
    return it with ``_get_batch_feature_mappings``.

    An example from the iris data set test:

        MAPPINGS = BatchFeatureMapping(
            'label',
            [ManagerFeatureMapping(
                'iris_vectorizer_manager',
                (FieldFeatureMapping('label', 'ilabel', True),
                 FieldFeatureMapping('flower_dims', 'iseries')))])

    :attribute label_feature_type: the name of the attribute used for labels
    :attribute manager_mappings: the manager level attribute mapping meta data
    """
    label_feature_type: str
    manager_mappings: List[ManagerFeatureMapping]


@dataclass
class DataPoint(metaclass=ABCMeta):
    """Abstract class that makes up a container class for features created from
    sentences.

    :attribute id: the ID of this data point, which maps back to the
                   ``BatchStash`` instance's subordinate stash
    :attribute batch_stash: ephemeral instance of the stash used during
                            encoding only

    """
    id: int
    batch_stash: BatchStash


@dataclass
class Batch(PersistableContainer):
    """Contains a batch of data used in the first layer of a net.  This class holds
    the labels, but is otherwise useless without at least one embedding layer
    matrix defined.

    :attribute batch_stash: ephemeral instance of the stash used during
                            encoding and decoding
    :attribute id: the ID of this batch instance, which is the sequence number
                   of the batch given during child processing of the chunked
                   data point ID setes
    :split_name: the name of the split for this batch (i.e. ``train`` vs
                 ``test``)
    :data_points: the list of the data points given on creation for encoding,
                  and ``None``'d out after encoding/pickinglin
    :data_point_ids: populated on instance creation and pickled along with the
                     class

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
        """Return the feature mapping meta data for this batch and it's data points.
        It is best to define a class level instance of the mapping and return
        it here to avoid instancing for each batch.

        :see BatchFeatureMapping:

        """
        pass

    def get_labels(self) -> torch.Tensor:
        """Return the label tensor for this batch.

        """
        label_attr = self._get_batch_feature_mappings().label_feature_type
        return self.attributes[label_attr]

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
        attribs = {}
        feats = {}
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
        """Decode the pickeled attriubtes after loaded by containing ``BatchStash`` and
        remove the context information to save memory.

        """
        attribs, feats = self._decode(self.ctx)
        delattr(self, 'ctx')
        return attribs, feats

    @property
    def attributes(self):
        return self._get_decoded_state()[0]

    @property
    def features(self):
        return self._get_decoded_state()[1]
