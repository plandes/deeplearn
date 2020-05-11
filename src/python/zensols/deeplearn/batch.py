"""This file contains a stash used to load an embedding layer.  It creates
features in batches of matrices and persists matrix only (sans features) for
efficient retrival.

"""
__author__ = 'Paul Landes'

from typing import Tuple, List, Any, Dict, Union, Set, Iterable
from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
import sys
import logging
from io import TextIOWrapper
import torch
import collections
from pathlib import Path
from zensols.util import time
from zensols.config import Configurable, Writable
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

    def __post_init__(self):
        if not isinstance(self.batch_id, str):
            raise ValueError(f'wrong id type: {type(self.batch_id)}')

    def __str__(self):
        return (f'{self.batch_id}: s={self.split_name} ' +
                f'({len(self.data_point_ids)})')
                #f'({self.data_point_ids})')

    def __repr__(self):
        return self.__str__()


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

    To speed up experiements, all available features configured in
    ``vectorizer_manager_set`` are encoded on disk.  However, only the
    ``decoded_attributes`` (see attribute below) are avilable to the model
    regardless of what was created during encoding time.

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


    :param config: the application configuration meant to be populated by
                   ``ImportClassFactory``

    :param name: the name of this stash in the application configuration

    :param data_point_type: a subclass type of ``DataPoint`` implemented
                            for the specific feature

    :param split_stash_container: the container that has the data set keys for
                                  each split (i.e. ``train`` vs ``test``)

    :param vectorizer_manager_set: used to vectorize features in to tensors

    :param decoded_attributes: the attributes to decode; only these are
                               avilable to the model regardless of what was
                               created during encoding time; if None, all are
                               available

    :param batch_size: the number of data points in each batch, except the last
                       (unless the data point cardinality divides the batch
                       size)

    :param model_torch_config: the PyTorch configuration used to
                               (optionally) copy CPU to GPU memory

    :param data_point_id_sets_path: the path of where to store key data for the
                                    splits; note that the container might store
                                    it's key splits in some other location

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
    batch_size: int
    model_torch_config: TorchConfig
    data_point_id_sets_path: Path
    batch_limit: int

    def __post_init__(self):
        super().__post_init__()
        self.data_point_id_sets_path.parent.mkdir(parents=True, exist_ok=True)
        self._batch_data_point_sets = PersistedWork(
            self.data_point_id_sets_path, self)
        self.priming = False

    @property
    @persisted('_batch_data_point_sets')
    def batch_data_point_sets(self) -> List[DataPointIDSet]:
        """Create the data point ID sets.  Each instance returned will correlate to a
        batch and each set of keys point to a feature ``DataPoint``.

        """
        psets = []
        batch_id = 0
        cont = self.split_stash_container
        logger.info(f'creating keys with {cont.__class__.__name__} ' +
                    f'using batch size of {self.batch_size}')
        for split, keys in cont.keys_by_split.items():
            logger.info(f'keys for split {split}: {len(keys)}')
            for chunk in chunks(keys, self.batch_size):
                chunk = tuple(chunk)
                logger.debug(f'chunked size: {len(chunk)}')
                psets.append(DataPointIDSet(str(batch_id), chunk, split))
                batch_id += 1
        psettrunc = psets[:self.batch_limit]
        logger.info(f'created {len(psets)} dp sets and truncated ' +
                    f'to {len(psettrunc)}, batch_limit={self.batch_limit}')
        return psettrunc

    def _get_keys_by_split(self) -> Dict[str, Set[str]]:
        by_set = collections.defaultdict(lambda: set())
        for dps in self.batch_data_point_sets:
            by_set[dps.split_name].add(dps.batch_id)
        return dict(by_set)

    def _create_data(self) -> List[DataPointIDSet]:
        """Data created for the sub proceesses are the first N data point ID sets.

        """
        return self.batch_data_point_sets

    def _process(self, chunk: List[DataPointIDSet]) -> \
            Iterable[Tuple[str, Any]]:
        """Create the batches by creating the set of data points for each
        ``DataPointIDSet`` instance.  When the subordinate stash dumps the
        batch (specifically a subclass of ``Batch), the overrided pickle logic
        is used to *detatch* the batch by encoded all data in to
        ``EncodedFeatureContext`` instances.

        """
        logger.info(f'processing: {chunk}')
        dpcls = self.data_point_type
        bcls = self.batch_type
        cont = self.split_stash_container
        batches: List[Batch] = []
        points: Tuple[DataPoint]
        batch: Batch
        dset: DataPointIDSet
        for dset in chunk:
            batch_id = dset.batch_id
            points = tuple(map(lambda dpid: dpcls(dpid, self, cont[dpid]),
                               dset.data_point_ids))
            batch = bcls(self, batch_id, dset.split_name, points)
            logger.info(f'created batch: {batch}')
            batches.append((batch_id, batch))
        return batches

    def reconstitute_batch(self, batch: Any) -> Any:
        """Return a new instance of a batch, which is some subclass of ``Batch``, with
        instances of it's respective data points repopulated.  This is useful
        after a batch is decoded and the original data point data is needed.

        """
        dpcls = self.data_point_type
        cont = self.split_stash_container
        points = tuple(map(lambda dpid: dpcls(dpid, self, cont[dpid]),
                           batch.data_point_ids))
        bcls = self.batch_type
        return bcls(self, batch.id, batch.split_name, points)

    def load(self, name: str):
        logger.info(f'loading {name}, child={self.is_child}')
        with time(f'loaded batch {name}'):
            obj = super().load(name)
        # add back the container of the batch to reconstitute the original
        # features and use the CUDA for tensor device transforms
        if obj is not None and not hasattr(obj, 'batch_stash'):
            obj.batch_stash = self
        return obj

    def prime(self):
        logger.debug(f'priming {self.__class__}, is child: {self.is_child}, ' +
                     f'currently priming: {self.priming}')
        if self.priming:
            raise ValueError('already priming')
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

    :params attr: the attribute name, which is used to identify the
                  feature that is vectorized

    :param feature_type: indicates which vectorizer to use

    :param is_agg: if ``True``, tuplize across all data points and encode as
                   one tuple of data to create the batched tensor on decode;
                   otherwise, each data point feature is encoded and
                   concatenated on decode

    """
    attr: str
    feature_type: str
    is_agg: bool = field(default=False)


@dataclass
class ManagerFeatureMapping(object):
    """Meta data for a vectorizer manager with fields describing attributes to be
    vectorized from features in to feature contests.

    :param vectorizer_manager_name: the configuration name that identifiees
                                    an instance of ``FeatureVectorizerManager``
    :param field: the fields of the data point to be vectorized
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

    :param label_feature_type: the name of the attribute used for labels
    :param manager_mappings: the manager level attribute mapping meta data
    """
    label_feature_type: str
    manager_mappings: List[ManagerFeatureMapping]


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
        label_attr = self._get_batch_feature_mappings().label_feature_type
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
                feats[feature_type] = attrib
        return attribs, feats

    def __str__(self):
        return f'{super().__str__()}: state={self.state}'

    def __repr__(self):
        return self.__str__()
