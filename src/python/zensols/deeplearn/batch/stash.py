"""This file contains a stash used to load an embedding layer.

"""
__author__ = 'Paul Landes'

from typing import Tuple, List, Any, Dict, Set, Iterable
from dataclasses import dataclass
from abc import ABCMeta
import logging
import collections
from pathlib import Path
from zensols.util import time
from zensols.config import Configurable
from zensols.persist import (
    chunks,
    persisted,
    PersistedWork,
    DirectoryCompositeStash,
)
from zensols.multi import MultiProcessStash
from zensols.dataset import (
    SplitKeyContainer,
    SplitStashContainer,
)
from zensols.deeplearn import TorchConfig
from zensols.deeplearn.vectorize import FeatureVectorizerManagerSet

logger = logging.getLogger(__name__)


class BatchDirectoryCompositeStash(DirectoryCompositeStash):
    def __init__(self, path: Path, groups: Tuple[Set[str]]):
        super().__init__(path, groups, '_feature_contexts')


@dataclass
class DataPointIDSet(object):
    """Set of subordinate stash IDs with feature values to be vectorized with
    ``BatchStash``.  Groups of these are sent to subprocesses for processing in
    to ``Batch`` instances.

    """
    batch_id: str
    data_point_ids: Set[str]
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
    decoded_attributes: Set[str]
    batch_size: int
    model_torch_config: TorchConfig
    data_point_id_sets_path: Path
    batch_limit: int

    def __post_init__(self):
        super().__post_init__()
        self.data_point_id_sets_path.parent.mkdir(parents=True, exist_ok=True)
        self._batch_data_point_sets = PersistedWork(
            self.data_point_id_sets_path, self)
        if isinstance(self.delegate, BatchDirectoryCompositeStash):
            self.delegate.load_keys = self.decoded_attributes
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
        tc_seed = TorchConfig.get_random_seed_context()
        logger.info(f'creating keys with {cont.__class__.__name__} ' +
                    f'using batch size of {self.batch_size}')
        for split, keys in cont.keys_by_split.items():
            logger.info(f'keys for split {split}: {len(keys)}')
            keys = sorted(keys)
            for chunk in chunks(keys, self.batch_size):
                chunk = tuple(chunk)
                logger.debug(f'chunked size: {len(chunk)}')
                dp_set = DataPointIDSet(str(batch_id), chunk, split, tc_seed)
                psets.append(dp_set)
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
        tseed = chunk[0].torch_seed_context
        dpcls = self.data_point_type
        bcls = self.batch_type
        cont = self.split_stash_container
        batches: List[Batch] = []
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
