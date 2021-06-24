"""Vectorization base classes and basic functionality.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Any, Union
from dataclasses import dataclass, field
from abc import abstractmethod, ABCMeta
import logging
import sys
from io import TextIOBase
from scipy import sparse
from scipy.sparse.csr import csr_matrix
import torch
from torch import Tensor
from zensols.persist import PersistableContainer
from zensols.config import ConfigFactory, Writable
from zensols.deeplearn import DeepLearnError, TorchConfig

logger = logging.getLogger(__name__)


class VectorizerError(DeepLearnError):
    """Thrown by instances of :class:`.FeatureVectorizer` during encoding or
    decoding operations.

    """
    pass


@dataclass
class ConfigurableVectorization(PersistableContainer, Writable):
    name: str = field()
    """The name of the section given in the configuration.

    """
    config_factory: ConfigFactory = field(repr=False)
    """The configuration factory that created this instance and used for
    serialization functions.

    """

    def __post_init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@dataclass
class FeatureVectorizer(ConfigurableVectorization, metaclass=ABCMeta):
    """An asbstrct base class that transforms a Python object in to a PyTorch
    tensor.

    """
    feature_id: str = field()
    """Uniquely identifies this vectorizer."""

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

        :see: obj:`feature_id`

        """
        return self.DESCRIPTION

    def __str__(self):
        return f'{self.feature_id} ({self.description}), shape: {self.shape}'

    def __repr__(self):
        return f'{self.__class__}: {self.__str__()}'

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(str(self), depth, writer)


@dataclass
class FeatureContext(PersistableContainer):
    """Data created by coding and meant to be pickled on the file system.

    :see EncodableFeatureVectorizer.encode:

    """

    feature_id: str = field()
    """The feature id of the :class:`.FeatureVectorizer` that created this context.

    """

    def __str__(self):
        return f'{self.__class__.__name__} ({self.feature_id})'


@dataclass
class NullFeatureContext(FeatureContext):
    """A no-op feature context used for cases such as prediction batches with data
    points that have no labels.

    :see: :meth:`~zensols.deeplearn.batch.BatchStash.create_prediction`

    :see: :class:`~zensols.deeplearn.batch.Batch`

    """
    pass


@dataclass
class TensorFeatureContext(FeatureContext):
    """A context that encodes data directly to a tensor.  This tensor could be a
    sparse matrix becomes dense during the decoding process.

    """

    tensor: Tensor = field()
    """The output tensor of the encoding phase."""

    def deallocate(self):
        super().deallocate()
        if hasattr(self, 'tensor'):
            del self.tensor

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

    sparse_data: Union[Tuple[Tuple[csr_matrix, int]], Tensor] = field()
    """The sparse array data."""

    @property
    def sparse_arr(self) -> Tuple[csr_matrix]:
        assert isinstance(self.sparse_data[0], tuple)
        return self.sparse_data[0]

    @classmethod
    def to_sparse(cls, arr: Tensor) -> Tuple[csr_matrix]:
        narr = arr.numpy()
        tdim = len(arr.shape)
        if tdim == 3:
            narrs = tuple(map(lambda i: narr[i], range(narr.shape[0])))
        elif tdim == 2 or tdim == 1:
            narrs = (narr,)
        else:
            raise VectorizerError('Tensors of dimensions higher than ' +
                                  f'3 not supported: {arr.shape}')
        mats = tuple(map(lambda m: sparse.csr_matrix(m), narrs))
        return (mats, tdim)

    @classmethod
    def instance(cls, feature_id: str, arr: Tensor,
                 torch_config: TorchConfig):
        arr = arr.cpu()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'encoding in to sparse tensor: {arr.shape}')
        if cls.USE_SPARSE:
            sarr = cls.to_sparse(arr)
        else:
            sarr = arr
        return cls(feature_id, sarr)

    def to_tensor(self, torch_config: TorchConfig) -> Tensor:
        if isinstance(self.sparse_arr, Tensor):
            tarr = self.sparse_arr
        else:
            narr, tdim = self.sparse_data
            narrs = tuple(map(lambda sm: torch.from_numpy(sm.todense()), narr))
            if len(narrs) == 1:
                tarr = narrs[0]
            else:
                tarr = torch.stack(narrs)
            dim_diff = len(tarr.shape) - tdim
            if dim_diff > 0:
                for _ in range(dim_diff):
                    tarr = tarr.squeeze(0)
            elif dim_diff < 0:
                for _ in range(-dim_diff):
                    tarr = tarr.unsqueeze(0)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'decoded sparce matrix to: {tarr.shape}')
        return tarr


@dataclass
class MultiFeatureContext(FeatureContext):
    """A composite context that contains a tuple of other contexts.

    """
    contexts: Tuple[FeatureContext]

    @property
    def is_empty(self) -> bool:
        cnt = sum(1 for _ in filter(
            lambda c: not isinstance(c, NullFeatureContext), self.contexts))
        return cnt == 0

    def deallocate(self):
        super().deallocate()
        if hasattr(self, 'contexts'):
            self._try_deallocate(self.contexts)
            del self.contexts
