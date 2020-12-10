"""This file contains a stash used to load an embedding layer.

"""
__author__ = 'Paul Landes'

from typing import Tuple, List, Any, Dict, Set, Iterable
from dataclasses import dataclass, InitVar
from abc import ABCMeta
import logging
import collections
import itertools as it
from itertools import chain
from pathlib import Path
from zensols.util import time
from zensols.config import Writeback, Configurable
from zensols.persist import (
    chunks,
    Deallocatable,
    persisted,
    PersistedWork,
    Primeable,
    Stash,
    DirectoryCompositeStash,
)
from zensols.multi import MultiProcessStash
from zensols.dataset import (
    SplitKeyContainer,
    SplitStashContainer,
)
from zensols.deeplearn import TorchConfig
from zensols.deeplearn.vectorize import (
    FeatureVectorizer,
    FeatureVectorizerManager,
    FeatureVectorizerManagerSet,
)
from . import BatchFeatureMapping

logger = logging.getLogger(__name__)


class BatchDirectoryCompositeStash(DirectoryCompositeStash):
    """A composite stash used for instances of :class:`BatchStash`.

    """
    def __init__(self, path: Path, groups: Tuple[Set[str]]):
        super().__init__(path, groups, '_feature_contexts')


@dataclass
class DataPointIDSet(object):
    """Set of subordinate stash IDs with feature values to be vectorized with
    :class:`.BatchStash`.  Groups of these are sent to subprocesses for
    processing in to :class:`.Batch` instances.

    :param batch_id: the ID of the batch

    :param data_point_ids: the IDs each data point in the setLevel

    :param split_name: the split (i.e. ``train``, ``test``, ``validation``)

    :param torch_seed_context: the seed context given by :class:`.TorchConfig`

    """
    batch_id: str
    data_point_ids: Tuple[str]
    split_name: str
    torch_seed_context: Dict[str, Any]

    def __post_init__(self):
        if not isinstance(self.batch_id, str):
            raise ValueError(f'wrong id type: {type(self.batch_id)}')

    def __str__(self):
        return (f'{self.batch_id}: s={self.split_name} ' +
                f'({len(self.data_point_ids)})')

    def __repr__(self):
        return self.__str__()


@dataclass
class BatchStash(MultiProcessStash, SplitKeyContainer, Writeback,
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


    :param config: the application configuration meant to be populated by
                   :class:`zensols.config.factory.ImportClassFactory`

    :param name: the name of this stash in the application configuration

    :param data_point_type: a subclass type of :class:`.DataPoint` implemented
                            for the specific feature

    :param split_stash_container: the source data stash that has both the data
                                  and data set keys for each split
                                  (i.e. ``train`` vs ``test``)

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
    config: Configurable
    data_point_type: type
    batch_type: type
    split_stash_container: SplitStashContainer
    vectorizer_manager_set: FeatureVectorizerManagerSet
    batch_size: int
    model_torch_config: TorchConfig
    data_point_id_sets_path: Path
    batch_limit: int
    decoded_attributes: InitVar[Set[str]]

    def __post_init__(self, decoded_attributes):
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
            raise ValueError('expecting SplitStashContainer but got ' +
                             f'{self.split_stash_container.__class__}')
        self.data_point_id_sets_path.parent.mkdir(parents=True, exist_ok=True)
        self._batch_data_point_sets = PersistedWork(
            self.data_point_id_sets_path, self)
        self.decoded_attributes = decoded_attributes
        self.priming = False

    @property
    def decoded_attributes(self) -> Set[str]:
        """The attributes to decode.  Only these are avilable to the model regardless
        of what was created during encoding time; if None, all are available

        """
        return self._decoded_attributes

    @decoded_attributes.setter
    def decoded_attributes(self, attribs):
        """The attributes to decode.  Only these are avilable to the model regardless
        of what was created during encoding time; if None, all are available

        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'setting decoded attributes: {attribs}')
        self._decoded_attributes = attribs
        if isinstance(self.delegate, BatchDirectoryCompositeStash):
            self.delegate.load_keys = attribs

    @property
    @persisted('_batch_data_point_sets')
    def batch_data_point_sets(self) -> List[DataPointIDSet]:
        """Create the data point ID sets.  Each instance returned will correlate to a
        batch and each set of keys point to a feature :class:`.DataPoint`.

        """
        psets = []
        batch_id = 0
        cont = self.split_stash_container
        tc_seed = TorchConfig.get_random_seed_context()
        logger.info(f'{self.name}: creating keys with ({type(cont)}) ' +
                    f'using batch size of {self.batch_size}')
        for split, keys in cont.keys_by_split.items():
            logger.info(f'keys for split {split}: {len(keys)}')
            #keys = sorted(keys, key=int)
            cslice = it.islice(chunks(keys, self.batch_size), self.batch_limit)
            for chunk in cslice:
                chunk = tuple(chunk)
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
        """Data created for the sub proceesses are the first N data point ID sets.

        """
        return self.batch_data_point_sets

    def _process(self, chunk: List[DataPointIDSet]) -> \
            Iterable[Tuple[str, Any]]:
        """Create the batches by creating the set of data points for each
        :class:`.DataPointIDSet` instance.  When the subordinate stash dumps
        the batch (specifically a subclass of :class:`.Batch`), the overrided
        pickle logic is used to *detatch* the batch by encoded all data in to
        ``EncodedFeatureContext`` instances.

        """
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'{self.name}: processing: {len(chunk)} data points')
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'chunk data points: {chunk}')
        tseed = chunk[0].torch_seed_context
        dpcls = self.data_point_type
        bcls = self.batch_type
        cont = self.split_stash_container
        points: Tuple[DataPoint]
        batch: Batch
        dset: DataPointIDSet
        if tseed is not None:
            TorchConfig.set_random_seed(
                tseed['seed'], tseed['disable_cudnn'], False)
        for dset in chunk:
            batch_id = dset.batch_id
            points = tuple(map(lambda dpid: dpcls(dpid, self, cont[dpid]),
                               dset.data_point_ids))
            batch = bcls(self, batch_id, dset.split_name, points)
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

    def reconstitute_batch(self, batch: Any) -> Any:
        """Return a new instance of a batch, which is some subclass of :class:`.Batch`,
        with instances of it's respective data points repopulated.  This is
        useful after a batch is decoded and the original data point data is
        needed.

        """
        points = self._get_data_points_for_batch(batch)
        return self.batch_type(self, batch.id, batch.split_name, points)

    def get_label_feature_vectorizer(self, batch) -> FeatureVectorizer:
        """Return the label vectorizer used in the batch.  This assumes there's only
        one vectorizer found in the vectorizer manager.

        :param batch: used to access the vectorizer set via the batch stash

        """
        batch = self.reconstitute_batch(batch)
        mapping: BatchFeatureMapping = batch._get_batch_feature_mappings()
        field_name: str = mapping.label_attribute_name
        mng, f = mapping.get_field_map_by_attribute(field_name)
        vec_name: str = mng.vectorizer_manager_name
        vec_mng_set = self.vectorizer_manager_set
        vec: FeatureVectorizerManager = vec_mng_set[vec_name]
        return vec[f.feature_id]

    def load(self, name: str):
        with time('loaded batch {name} ({obj.split_name})'):
            obj = super().load(name)
        # add back the container of the batch to reconstitute the original
        # features and use the CUDA for tensor device transforms
        if obj is not None and not hasattr(obj, 'batch_stash'):
            obj.batch_stash = self
        return obj

    def _prime_vectorizers(self):
        vec_mng_set = self.vectorizer_manager_set
        vecs = map(lambda v: v.vectorizers.values(), vec_mng_set.values())
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
            raise ValueError('already priming')
        self.priming = True
        try:
            self.batch_data_point_sets
            self._prime_vectorizers()
            super().prime()
        finally:
            self.priming = False

    def deallocate(self):
        super().deallocate()
        self._batch_data_point_sets.deallocate()
        if id(self.delegate) != id(self.split_stash_container):
            self._try_deallocate(self.delegate)
        self._try_deallocate(self.split_stash_container)
        self.vectorizer_manager_set.deallocate()

    def clear(self):
        logger.debug('clear: calling super')
        super().clear()
        self._batch_data_point_sets.clear()
