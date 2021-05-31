from __future__ import annotations
"""This file contains a stash used to load an embedding layer.

"""
__author__ = 'Paul Landes'

from typing import Tuple, List, Any, Dict, Set, Iterable, Type
from dataclasses import dataclass, InitVar, field
from abc import ABCMeta
import sys
import logging
import collections
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
from zensols.deeplearn import TorchConfig
from zensols.deeplearn.vectorize import FeatureVectorizerManagerSet
from . import (
    BatchDirectoryCompositeStash, DataPointIDSet,
    DataPoint, Batch, TorchMultiProcessStash,
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

    :see _process: for details on the pickling of the batch instances

    """
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

    decoded_attributes: InitVar[Set[str]] = field()
    """The attributes to decode; only these are avilable to the model
    regardless of what was created during encoding time; if None, all are
    available.

    """

    batch_limit: int = field(default=sys.maxsize)
    """The max number of batches to process, which is useful for debugging."""

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
        """Data created for the sub proceesses are the first N data point ID sets.

        """
        return self.batch_data_point_sets

    def _process(self, chunk: List[DataPointIDSet]) -> \
            Iterable[Tuple[str, Any]]:
        """Create the batches by creating the set of data points for each
        :class:`.DataPointIDSet` instance.  When the subordinate stash dumps
        the batch (specifically a subclass of :class:`.Batch`), the overrided
        pickle logic is used to *detatch* the batch by encoded all data in to
        :class:`.FeatureContext` instances.

        """
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'{self.name}: processing: {len(chunk)} data points')
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'chunk data points: {chunk}')
        tseed = chunk[0].torch_seed_context
        dpcls: Type[DataPoint] = self.data_point_type
        bcls: Type[Batch] = self.batch_type
        cont = self.split_stash_container
        points: Tuple[DataPoint]
        batch: Batch
        if tseed is not None:
            TorchConfig.set_random_seed(
                tseed['seed'], tseed['disable_cudnn'], False)
        dset: DataPointIDSet
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

    def load(self, name: str):
        with time('loaded batch {name} ({obj.split_name})'):
            obj = super().load(name)
        # add back the container of the batch to reconstitute the original
        # features and use the CUDA for tensor device transforms
        if obj is not None and not hasattr(obj, 'batch_stash'):
            obj.batch_stash = self
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
            raise ValueError('already priming')
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

    def clear(self):
        logger.debug('clearing')
        super().clear()
        self._batch_data_point_sets.clear()
