"""This file contains a stash used to load an embedding layer.  It creates
features in batches of matrices and persists matrix only (sans features) for
efficient retrival.

"""
__author__ = 'Paul Landes'

import sys
import logging
import traceback
from typing import List, Any, Dict, Iterable, Tuple
from dataclasses import dataclass, field
import itertools as it
from itertools import chain
import copy as cp
from abc import ABCMeta, abstractmethod
import numpy as np
import torch
from zensols.deeplearn import TorchConfig
from zensols.util import time
from zensols.config import Configurable
from zensols.persist import (
    chunks,
    persisted,
    PersistedWork,
    PersistableContainer
)
from zensols.multi import MultiProcessStash
from zensols.deeplearn import FeatureVectorizerManager

logger = logging.getLogger(__name__)


@dataclass
class BatchStash(MultiProcessStash, metaclass=ABCMeta):
    """Load the data sets in memory and collate each word with a word vector for
    the word embedding layer of the model.  The word embedding layer is created
    by "Magic Bedding" the width of the matrix: if the sentence is too short we
    pad with zeros (stretch), if it's too long we cut it off (the feet/legs).

    :param torch_config: defines how/if we're using CUDA acceleration; if we
                        are create the word embeding layer directly as GPU
                        tensors

    """
    #ATTR_EXP_META = ('split_type',)

    config: Configurable
    name: str
    data_point_type: type
    vec_manager: FeatureVectorizerManager
    article_limit: int = field(default=sys.maxsize)

    @abstractmethod
    def _create_batch(self, batch_id: int, chunk: list):
        pass

    def _create_data(self) -> List[str]:
        with time(f'created batch keys'):
            keys = it.islice(self.factory.keys(), self.article_limit)
            logger.debug(f'creating data set {self.split_type}')
            return keys

    def _process(self, ids: List[str]) -> Iterable[Tuple[str, Any]]:
        logger.info(f'creating data sets {self.split_type} ' +
                    f'batch size: {self.batch_size}')
        dtype = self.data_point_type
        dpoints = map(lambda i: dtype(i, self, self.factory[i]), ids)
        for i, chunk in enumerate(chunks(dpoints, self.batch_size)):
            batch_id = f'{chunk[0].id}.{i}'
            batch = self._create_batch(batch_id, self.split_type, chunk)
            logger.debug(f'created batch: {batch} with ' +
                         f'{len(batch.data_points)} data points')
            yield (batch.id, batch)

    def load(self, name: str):
        obj = super().load(name)
        # add back the container of the batch to reconstitute the original
        # features and use the CUDA for tensor device transforms
        if obj is not None and not hasattr(obj, 'batch_stash'):
            obj.batch_stash = self
        return obj


@dataclass
class DataPoint(metaclass=ABCMeta):
    """Abstract class that makes up a container class for features created from
    sentences.

    The ``get_label_matrices`` method needs an implementation in subclasses.

    """
    id: int
    batch_stash: BatchStash

    @abstractmethod
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
    split_type: str
    data_points: List[DataPoint] = field(repr=False)

    def __post_init__(self):
        if self.data_points is not None:
            self.data_point_ids = tuple(map(lambda d: d.id, self.data_points))

    @property
    def torch_config(self) -> TorchConfig:
        """Return the CUDA configuration used to generate a CUDA version of this batch.

        :see ``to``
        """
        return self.batch_stash.torch_config

    def _add_detach_attributes(self, attribs: List[str]):
        attribs.append('label')

    def _add_present_attributes(self, attribs: List[str]):
        attribs.append('label')

    def _add_embedded_matrices(self, attribs: Dict[str, np.ndarray]):
        pass

    def _get_matrix_property(self, name) -> np.ndarray:
        """Return a matrix/array by name in the batch.

        """
        try:
            meth = f'get_{name}_matrices'
            logger.debug(f'getting method: {meth} on {self}')
            return getattr(self, meth)()
        except Exception as e:
            traceback.print_exc()
            if hasattr(self, 'data_points'):
                dps = ','.join(map(lambda d: str(d.id), self.data_points))
            else:
                dps = '<no data points>'
            msg = f'could not create {name} for batch {self.id} with {dps}: {e}'
            logger.error(msg)
            raise e

    def _batch_matrices(self, mats):
        """Stack matrices created from data points in to the batch matrix.

        """
        mats = chain.from_iterable(map(tuple, mats))
        mats = torch.stack(tuple(mats))
        logger.debug(f'batched matrices: {mats.shape}')
        return mats

    def detach(self):
        """Called to create all matrices/arrays needed for the layer.  After this is
        called, features in this instance are removed for so pickling is fast.

        """
        attribs = []
        meta = self._get_persistable_metadata()
        self._add_detach_attributes(attribs)
        for name in attribs:
            self._get_matrix_property(name)
        for k, v in meta.persisted.items():
            val = v()
            if isinstance(val, torch.Tensor):
                v.set(val.detach().cpu())
            elif (isinstance(val, tuple) or isinstance(val, list)) and isinstance(val[0], torch.Tensor):
                v.set(tuple(map(lambda t: t.cpu(), val)))

    def deallocate(self):
        """Used to deallocate all resources in the batch.  Useful for quickly getting
        memory back from CUDA allocated resources.  This is nulls out all
        reference cycles.

        """
        if hasattr(self, 'is_clone') and self.is_clone:
            del self.split_type
            del self.id
            if hasattr(self, 'batch_stash'):
                delattr(self, 'batch_stash')
            if hasattr(self, 'data_points'):
                del self.data_points
            meta = self._get_persistable_metadata()
            meta.clear()
            for k, v in meta.persisted.items():
                delattr(self, k)
                del v

    def reconstitute_features(self) -> iter:
        """Return an iterator of the original features used in the data points that
        created the original batch and parse prime time.

        """
        fac = self.batch_stash.factory
        return map(lambda d: fac[d], self.data_point_ids)

    @persisted('__embed_matrices', transient=True)
    def _embed_matricies(self):
        """Return all matricies that are embedded, which get passed through the torch
        library's ``nn.Embeddded`` class to create the embedding to be cashed
        by this instance.

        """
        repl_mats = {}
        embedded_matrices = {}
        self._add_embedded_matrices(embedded_matrices)
        for name, layer in embedded_matrices.items():
            logger.debug(f'layer.config: {layer.config}')
            if layer.config.cached:
                mat = self._get_matrix_property(name)
                mat = layer.preemptive_foward(mat).clone().detach().cpu()
                repl_mats[name] = mat
        return repl_mats

    def _reconstitute_map_to_matrix(self, other: Any):
        raise ValueError(f'unknown unmapped object: {type(other)}')

    def to(self):
        """Return a clone of this batch with all CUDA memory mapped resources.

        :see ``torch_config``

        """
        labs = self.get_label_matrices()
        embedded_matrices = self._embed_matricies()
        if not self.torch_config.same_device(labs):
            present_attributes = []
            self._add_present_attributes(present_attributes)
            # can't use import copy here since persistables are forever linked
            # to their instances
            clone = self.__class__(self.batch_stash, self.id, self.split_type, None)
            clone.batch_stash = self.batch_stash
            clone.data_point_ids = cp.copy(self.data_point_ids)
            for name in present_attributes:
                if name in embedded_matrices:
                    mat = embedded_matrices[name]
                else:
                    mat = self._get_matrix_property(name)
                pw_name = f'_{name}_matrices'
                pw = PersistedWork(pw_name, clone)
                if not isinstance(mat, torch.Tensor):
                    mat = self._reconstitute_map_to_matrix(mat)
                else:
                    mat = mat.clone().detach()
                    mat = self.torch_config.to(mat)
                logger.debug(f'name={name}, shape={mat.shape}, device={mat.device}')
                pw.set(mat)
                setattr(clone, pw_name, pw)
                clone.is_clone = True
            return clone
        return self

    def get_matrices(self) -> Dict[str, np.ndarray]:
        """Return a dictionary of tensors by name.

        """
        mats = {}
        attribs = []
        self._add_present_attributes(attribs)
        for name in attribs:
            mat = self._get_matrix_property(name)
            mats[name] = mat
        return mats

    def __getstate__(self):
        self.detach()
        state = super().__getstate__()
        state.pop('batch_stash', None)
        state.pop('data_points', None)
        return state

    @persisted('_label_matrices')
    def get_label_matrices(self):
        """Return the labels for all data points in the batch.

        :see: ``get_labels``

        """
        labels = it.chain(*map(lambda p: p.get_label_matrices(), self.data_points))
        return torch.stack(tuple(labels))

    def get_labels(self) -> np.ndarray:
        """Convenience method for ``get_label_matrices``, which must retain the naming
        for propery persistence mechanisms.

        """
        return self.get_label_matrices()

    def __len__(self):
        return len(self.get_label_matrices())

    def write(self, writer=sys.stdout, indent=0):
        """Write a human readable representation of the batch.

        :param writer: the writer to dump the human readable text
        :param indent: the indent to write the data points (if not detached)

        """
        sp = ' ' * (indent * 2)
        writer.write(f'{sp}{self}\n')
        if hasattr(self, 'data_points'):
            sp = ' ' * ((indent + 1) * 2)
            writer.write('data points:\n')
            for nf in self.data_points:
                writer.write(f'{sp}{nf}\n')

    def __str__(self):
        return (f'id: {self.id} ({self.split_type})')
