"""Vectorization base classes and basic functionality.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Any, Union
from dataclasses import dataclass
from abc import abstractmethod, ABCMeta
import logging
import sys
from io import TextIOBase
from scipy import sparse
from scipy.sparse.csr import csr_matrix
import torch
from torch import Tensor
from zensols.config import Writeback
from zensols.persist import PersistableContainer
from zensols.deeplearn import TorchConfig

logger = logging.getLogger(__name__)


@dataclass
class FeatureVectorizer(Writeback, metaclass=ABCMeta):
    """An asbstrct base class that transforms a Python object in to a PyTorch
    tensor.

    """
    feature_id: str

    def __post_init__(self, *args, **kwargs):
        pass

    def _allow_config_adds(self) -> bool:
        return True

    @abstractmethod
    def _get_shape(self) -> Tuple[int, int]:
        pass

    @abstractmethod
    def transform(self, data: Any) -> Tensor:
        """Transform ``data`` to a tensor data format.

        """
        pass

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the tensor created by ``transform``.

        """
        return self._get_shape()

    @property
    def description(self) -> str:
        """A short human readable name.

        :see feature_id:

        """
        return self.DESCRIPTION

    def __str__(self):
        return f'{self.feature_id} ({self.description})'

    def __repr__(self):
        return f'{self.__class__}: {self.__str__()}'

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(f'{self}, shape: {self.shape}', depth, writer)


@dataclass
class FeatureContext(PersistableContainer):
    """Data created by coding and meant to be pickled on the file system.

    :param feature_id: the feature id of the ``FeatureVectorizer`` that created
                       this context.

    :see EncodableFeatureVectorizer.encode:

    """
    feature_id: str

    def __str__(self):
        return f'{self.__class__.__name__} ({self.feature_id})'


@dataclass
class TensorFeatureContext(FeatureContext):
    """A context that encodes data directly to a tensor.  This tensor could be a
    sparse matrix becomes dense during the decoding process.

    :param tensor: the output tensor of the encoding phase

    """
    tensor: Tensor

    def __str__(self):
        tstr = f'{self.tensor.shape}' if self.tensor is not None else '<none>'
        return f'{super().__str__()}: {tstr}'

    def __repr__(self):
        return self.__str__()


@dataclass
class SparseTensorFeatureContext(FeatureContext):
    """Contains data that was encded from a dense matrix as a sparse matrix and
    back.  Using torch sparse matrices currently lead to deadlocking in child
    proceesses, so use scipy :class:``csr_matrix`` is used instead.

    """
    USE_SPARSE = True
    sparse_arr: Union[csr_matrix, Tensor]

    @classmethod
    def instance(cls, feature_id: str, arr: Tensor,
                 torch_config: TorchConfig):
        arr = arr.cpu()
        if cls.USE_SPARSE:
            narr = arr.numpy()
            sarr = sparse.csr_matrix(narr)
        else:
            sarr = arr
        return cls(feature_id, sarr)

    def to_tensor(self, torch_config: TorchConfig) -> Tensor:
        if isinstance(self.sparse_arr, Tensor):
            tarr = self.sparse_arr
        else:
            dense = self.sparse_arr.todense()
            tarr = torch.from_numpy(dense)
        return tarr


@dataclass
class MultiFeatureContext(FeatureContext):
    """A composite context that contains a tuple of other contexts.

    """
    contexts: Tuple[FeatureContext]
