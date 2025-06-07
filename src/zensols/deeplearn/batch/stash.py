"""This file contains a stash used to load an embedding layer.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import Tuple, List, Any, Dict, Set, Iterable, Union, Sequence, Type
from dataclasses import dataclass, field
from abc import ABCMeta
import sys
import logging
import collections
from functools import reduce
import itertools as it
from itertools import chain
from pathlib import Path
from zensols.util import time
from zensols.config import Writeback
from zensols.persist import (
    chunks,
    Deallocatable,
    persisted,
    PersistedWork,
    Primeable,
    Stash,
)
from zensols.dataset import (
    SplitKeyContainer,
    SplitStashContainer,
)
from zensols.deeplearn import TorchConfig, DeepLearnError
from zensols.deeplearn.vectorize import (
    FeatureVectorizerManagerSet,
    FeatureVectorizerManager,
)
from . import (
    BatchDirectoryCompositeStash, BatchFeatureMapping, DataPointIDSet,
    DataPoint, Batch, BatchMetadata, BatchFieldMetadata,
    ManagerFeatureMapping, FieldFeatureMapping, TorchMultiProcessStash,
)

logger = logging.getLogger(__name__)


@dataclass
class BatchStash(TorchMultiProcessStash, SplitKeyContainer, Writeback,
                 Deallocatable, metaclass=ABCMeta):
    """A stash that vectorizes features in to easily consumable tensors for
    training and testing.  This stash produces instances of :class:`.Batch`,
    which is a batch in the machine learning sense, and the first dimension of
    what will become the tensor used in PyTorch.  Each of these batches has a
    logical one to many relationship to that batche's respective set of data
    points, which is encapsulated in the :class:`.DataPoint` class.

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

    Use the :obj:`split_stash_container` to get dataset which as a ``splits``
    property for the feature data.  Use the ``dataset_stash`` from the
    application context :class:`~zensols.util.config.ConfigFactory` for the
    batch splits.

    :see _process: for details on the pickling of the batch instances

    .. document private functions
    .. automethod:: _process

    """
    _DICTABLE_WRITE_EXCLUDES = {'batch_feature_mappings'}

    data_point_type: Type[DataPoint] = field()
    """A subclass type of :class:`.DataPoint` implemented for the specific
    feature.

    """
    batch_type: Type[Batch] = field()
    """The batch class to be instantiated when created batchs.

    """
    split_stash_container: SplitStashContainer = field()
    """The source data stash that has both the data and data set keys for each
    split (i.e. ``train`` vs ``test``).

    """
    vectorizer_manager_set: FeatureVectorizerManagerSet = field()
    """Used to vectorize features in to tensors."""

    batch_size: int = field()
    """The number of data points in each batch, except the last (unless the
    data point cardinality divides the batch size).

    """
    model_torch_config: TorchConfig = field()
    """The PyTorch configuration used to (optionally) copy CPU to GPU memory.

    """
    data_point_id_sets_path: Path = field()
    """The path of where to store key data for the splits; note that the
    container might store it's key splits in some other location.

    """
    decoded_attributes: Union[Set[str], Sequence[str]] = field()
    """The attributes to decode; only these are avilable to the model regardless
    of what was created during encoding time; if None, all are available.
    Sequences are converted to sets, which makes configuration easier in YAML
    files.

    """
    batch_feature_mappings: BatchFeatureMapping = field(default=None)
    """The meta data used to encode and decode each feature in to tensors.

    """
    batch_limit: int = field(default=sys.maxsize)
    """The max number of batches to process, which is useful for debugging."""

    def __post_init__(self):
        super().__post_init__()
        Deallocatable.__init__(self)
        # TODO: this class conflates key split and delegate stash functionality
        # in the `split_stash_container`.  An instance of this type serves the
        # purpose, but it need not be.  Instead it just needs to be both a
        # SplitKeyContainer and a Stash.  This probably should be split out in
        # to two different fields.
        cont = self.split_stash_container
        if not isinstance(cont, SplitStashContainer) \
           and (not isinstance(cont, SplitKeyContainer) or
                not isinstance(cont, Stash)):
            raise DeepLearnError('Expecting SplitStashContainer but got ' +
                                 f'{self.split_stash_container.__class__}')
        self._batch_data_point_sets = PersistedWork(
            self.data_point_id_sets_path, self, mkdir=True)
        self.priming = False
        self._update_comp_stash_attribs()

    @property
    def _decoded_attributes(self) -> Set[str]:
        """The attributes to decode.  Only these are avilable to the model
        regardless of what was created during encoding time; if None, all are
        available

        """
        return self._decoded_attributes_val

    @_decoded_attributes.setter
    def _decoded_attributes(self, attribs: Union[Set[str], Sequence[str]]):
        """The attributes to decode.  Only these are avilable to the model
        regardless of what was created during encoding time; if None, all are
        available

        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'setting decoded attributes: {attribs}')
        if isinstance(attribs, (tuple, list)):
            attribs = set(attribs)
        self._decoded_attributes_val = attribs
        if isinstance(self.delegate, BatchDirectoryCompositeStash):
            self.delegate.load_keys = attribs

    @property
    @persisted('_batch_metadata')
    def batch_metadata(self) -> BatchMetadata:
        mapping: BatchFeatureMapping
        if self.batch_feature_mappings is not None:
            mapping = self.batch_feature_mappings
        else:
            batch: Batch = self.batch_type(None, None, None, None)
            batch.batch_stash = self
            mapping = batch._get_batch_feature_mappings()
            batch.deallocate()
        vec_mng_set: FeatureVectorizerManagerSet = self.vectorizer_manager_set
        attrib_keeps = self.decoded_attributes
        vec_mng_names = set(vec_mng_set.keys())
        by_attrib = {}
        mmng: ManagerFeatureMapping
        for mmng in mapping.manager_mappings:
            vec_mng_name: str = mmng.vectorizer_manager_name
            if vec_mng_name in vec_mng_names:
                vec_mng: FeatureVectorizerManager = vec_mng_set[vec_mng_name]
                field: FieldFeatureMapping
                for field in mmng.fields:
                    if field.attr in attrib_keeps:
                        vec = vec_mng[field.feature_id]
                        by_attrib[field.attr] = BatchFieldMetadata(field, vec)
        return BatchMetadata(self.data_point_type, self.batch_type,
                             mapping, by_attrib)

    def _update_comp_stash_attribs(self):
        """Update the composite stash grouping if we're using one and if this
        class is already configured.

        """
        if isinstance(self.delegate, BatchDirectoryCompositeStash):
            meta: BatchMetadata = self.batch_metadata
            meta_attribs: Set[str] = set(
                map(lambda f: f.attr, meta.mapping.get_attributes()))
            groups: Tuple[Set[str]] = self.delegate.groups
            gattribs = reduce(lambda x, y: x | y, groups)
            to_remove = gattribs - meta_attribs
            new_groups = []
            if len(to_remove) > 0:
                group: Set[str]
                for group in groups:
                    ng: Set[str] = meta_attribs & group
                    if len(ng) > 0:
                        new_groups.append(ng)
                self.delegate.groups = tuple(new_groups)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'meta attribs: {meta_attribs}, groups: {groups}')

    @property
    @persisted('_batch_data_point_sets')
    def batch_data_point_sets(self) -> List[DataPointIDSet]:
        """Create the data point ID sets.  Each instance returned will correlate
        to a batch and each set of keys point to a feature :class:`.DataPoint`.

        """
        psets = []
        batch_id = 0
        cont = self.split_stash_container
        tc_seed = TorchConfig.get_random_seed_context()
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'{self.name}: creating keys with ({type(cont)}) ' +
                        f'using batch size of {self.batch_size}')
        for split, keys in cont.keys_by_split.items():
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'keys for split {split}: {len(keys)}')
            # keys are ordered and needed to be as such for consistency
            # keys = sorted(keys, key=int)
            cslice = it.islice(chunks(keys, self.batch_size), self.batch_limit)
            for chunk in cslice:
                chunk = tuple(chunk)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'chunked size: {len(chunk)}')
                dp_set = DataPointIDSet(str(batch_id), chunk, split, tc_seed)
                psets.append(dp_set)
                batch_id += 1
        logger.info(f'created {len(psets)} each set limited with ' +
                    f'{self.batch_limit} with batch_limit={self.batch_limit}')
        return psets

    def _get_keys_by_split(self) -> Dict[str, Tuple[str]]:
        by_batch = collections.defaultdict(lambda: [])
        for dps in self.batch_data_point_sets:
            by_batch[dps.split_name].append(dps.batch_id)
        return {k: tuple(by_batch[k]) for k in by_batch.keys()}

    def _create_data(self) -> List[DataPointIDSet]:
        """Data created for the sub proceesses are the first N data point ID
        sets.

        """
        return self.batch_data_point_sets

    def populate_batch_feature_mapping(self, batch: Batch):
        """Add batch feature mappings to a batch instance."""
        if self.batch_feature_mappings is not None:
            batch.batch_feature_mappings = self.batch_feature_mappings

    def create_batch(self, points: Tuple[DataPoint], split_name: str = None,
                     batch_id: str = None):
        """Create a new batch instance with data points, which happens when
        primed.

        """
        bcls: Type[Batch] = self.batch_type
        batch: Batch = bcls(self, batch_id, split_name, points)
        self.populate_batch_feature_mapping(batch)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'created batch: {batch}')
        return batch

    def _process(self, chunk: List[DataPointIDSet]) -> \
            Iterable[Tuple[str, Any]]:
        """Create the batches by creating the set of data points for each
        :class:`.DataPointIDSet` instance.  When the subordinate stash dumps
        the batch (specifically a subclass of :class:`.Batch`), the overrided
        pickle logic is used to *detach* the batch by encoded all data in to
        :class:`.FeatureContext` instances.

        """
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'{self.name}: processing: {len(chunk)} data points')
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'chunk data points: {chunk}')
        tseed = chunk[0].torch_seed_context
        dpcls: Type[DataPoint] = self.data_point_type
        cont = self.split_stash_container
        if tseed is not None:
            TorchConfig.set_random_seed(
                tseed['seed'], tseed['disable_cudnn'], False)
        dset: DataPointIDSet
        for dset in chunk:
            batch_id: str = dset.batch_id
            points: Tuple[DataPoint] = tuple(
                map(lambda dpid: dpcls(dpid, self, cont[dpid]),
                    dset.data_point_ids))
            batch: Batch = self.create_batch(points, dset.split_name, batch_id)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'created batch: {batch}')
            yield (batch_id, batch)

    def _get_data_points_for_batch(self, batch: Any) -> Tuple[Any]:
        """Return the data points that were used to create ``batch``.

        """
        dpcls = self.data_point_type
        cont = self.split_stash_container
        return tuple(map(lambda dpid: dpcls(dpid, self, cont[dpid]),
                         batch.data_point_ids))

    def load(self, name: str):
        with time('loaded batch {name} ({obj.split_name})', logging.DEBUG):
            obj = super().load(name)
        # add back the container of the batch to reconstitute the original
        # features and use the CUDA for tensor device transforms
        if obj is not None:
            if not hasattr(obj, 'batch_stash'):
                obj.batch_stash = self
            if (not hasattr(obj, 'batch_feature_mappings') or
               obj.batch_feature_mappings is None):
                self.populate_batch_feature_mapping(obj)
        return obj

    def _prime_vectorizers(self):
        vec_mng_set: FeatureVectorizerManagerSet = self.vectorizer_manager_set
        vecs = map(lambda v: v.values(), vec_mng_set.values())
        for vec in chain.from_iterable(vecs):
            if isinstance(vec, Primeable):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'priming {vec}')
                vec.prime()

    def prime(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'priming {self.__class__}, is child: ' +
                         f'{self.is_child}, currently priming: {self.priming}')
        if self.priming:
            raise DeepLearnError('Already priming')
        self.priming = True
        try:
            self.batch_data_point_sets
            self._prime_vectorizers()
            super().prime()
        finally:
            self.priming = False

    def deallocate(self):
        self._batch_data_point_sets.deallocate()
        if id(self.delegate) != id(self.split_stash_container):
            self._try_deallocate(self.delegate)
        self._try_deallocate(self.split_stash_container)
        self.vectorizer_manager_set.deallocate()
        super().deallocate()

    def _from_dictable(self, *args, **kwargs):
        # avoid long Wriable.write output
        dct = super()._from_dictable(*args, **kwargs)
        rms = tuple(filter(lambda k: k.startswith('_'), dct.keys()))
        for k in rms:
            del dct[k]
        return dct

    def clear(self):
        """Clear the batch, batch data point sets."""
        logger.debug('clearing')
        super().clear()
        self._batch_data_point_sets.clear()

    def clear_all(self):
        """Clear the batch, batch data point sets, and the source data
        (:obj:`split_stash_container`).

        """
        self.clear()
        self.split_stash_container.clear()


BatchStash.decoded_attributes = BatchStash._decoded_attributes
