"""Vectorization base classes and basic functionality.

"""
__author__ = 'Paul Landes'


import logging
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Tuple, Any
import torch
from zensols.persist import PersistableContainer
from zensols.deeplearn import TorchConfig

logger = logging.getLogger(__name__)


@dataclass
class FeatureVectorizer(ABC):
    """An asbstrct base class that transforms a Python object in to a PyTorch
    tensor.

    """
    def __post_init__(self):
        if not hasattr(self, '_feature_type') and \
           hasattr(self.__class__, 'FEATURE_TYPE'):
            self.feature_type = self.FEATURE_TYPE
        self._name = self.NAME

    @abstractmethod
    def _get_shape(self) -> Tuple[int, int]:
        pass

    @abstractmethod
    def transform(self, data: Any) -> torch.tensor:
        """Transform ``data`` to a tensor data format.

        """
        pass

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the tensor created by ``transform``.

        """
        return self._get_shape()

    @property
    def name(self) -> str:
        """A short human readable name.

        :see feature_type:

        """
        return self._name

    @property
    def feature_type(self) -> str:
        """A short unique symbol of the feature.  The name should be somewhat
        undstandable.  However, meaning of the vectorizer comes from the
        ``name`` attriubte.

        :see name:

        """
        return self._feature_type

    @feature_type.setter
    def feature_type(self, feature_type):
        self._feature_type = feature_type

    def __str__(self):
        return f'{self.feature_type} ({self._name})'

    def __repr__(self):
        return f'{self.__class__}: {self.__str__()}'


@dataclass
class FeatureContext(PersistableContainer):
    """Data created by coding and meant to be pickled on the file system.

    :param feature_type: the feature type of the ``FeatureVectorizer`` that
                         created this context.

    :see EncodableFeatureVectorizer.encode:

    """
    feature_type: str

    def __str__(self):
        return f'{self.__class__.__name__} ({self.feature_type})'


@dataclass
class TensorFeatureContext(FeatureContext):
    """A context that encodes data directly to a tensor.  This tensor could be a
    sparse matrix becomes dense during the decoding process.

    :param tensor: the output tensor of the encoding phase

    """
    tensor: torch.Tensor

    def __str__(self):
        tstr = f'{self.tensor.shape}' if self.tensor is not None else '<none>'
        return f'{super().__str__()}: {tstr}'

    def __repr__(self):
        return self.__str__()


@dataclass
class SparseTensorFeatureContext(FeatureContext):
    """Contains data that was encded from a dense matrix as a sparse matrix and
    back.  Using torch sparse matrices currently lead to deadlocking in child
    proceesses, so the sparse algorithm has been reimplemenated.

    :todo: the encoding of the sparse matrix to bypass instantiating a torch
           sparse matrix

    """
    indices: torch.Tensor
    values: torch.Tensor
    shape: Tuple[int, int]

    def __post_init__(self):
        if self.indices[0].shape != self.indices[1].shape:
            raise ValueError(
                'sparse index coordiates size do not match: ' +
                f'{self.indices[0].shape} != {self.indices[1].shape}')
        if self.indices[0].shape != self.values.shape:
            raise ValueError(
                'size of indicies and length of values do not match: ' +
                f'{self.indices[0].shape} != {self.values.shape}')

    @classmethod
    def instance(cls, feature_type: str, arr: torch.Tensor,
                 torch_config: TorchConfig):
        if not torch_config.is_sparse(arr):
            arr = arr.to_sparse()
        indices = arr.indices()
        vals = arr.values()
        shape = tuple(arr.size())
        return cls(feature_type, indices, vals, shape)

    def to_tensor(self, torch_config: TorchConfig) -> torch.Tensor:
        arr = torch_config.zeros(self.shape)
        idx = self.indices
        for i, val in enumerate(self.values):
            r, c = idx[0][i], idx[1][i]
            arr[r, c] = val
        return arr
