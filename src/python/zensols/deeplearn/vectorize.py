"""Vectorization base classes and basic functionality

"""
__author__ = 'Paul Landes'


import logging
from abc import abstractmethod, ABC, ABCMeta
from dataclasses import dataclass, field
from typing import Tuple, Any, Set, Type, Dict, List, Iterable
from itertools import chain
import collections
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from zensols.persist import PersistableContainer, persisted
from zensols.config import ConfigFactory
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
    indices: torch.Tensor
    values: torch.Tensor
    shape: Tuple[int, int]

    def __post_init__(self):
        if self.indices[0].shape != self.indices[1].shape:
            raise ValueError(
                f'sparse index coordiates size do not match: ' +
                f'{self.indices[0].shape} != {self.indices[1].shape}')
        if self.indices[0].shape != self.values.shape:
            raise ValueError(
                f'size of indicies and length of values do not match: ' +
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


@dataclass
class EncodableFeatureVectorizer(FeatureVectorizer, metaclass=ABCMeta):
    """This vectorizer splits transformation up in to encoding and decoding.  The
    encoded state as a ``FeatureContext``, in cases where encoding is
    prohibitively expensive, is computed once and pickled to the file system.
    It is then loaded and finally decoded into a tensor.

    Examples include computing an encoding as indexes of a word embedding
    during the encoding phase.  Then generating the full embedding layer during
    decoding.  Note that this decoding is done with a ``TorchConfig`` so the
    output tensor goes directly to the GPU.

    This abstract base class only needs the ``_encode`` method overridden.  The
    ``_decode`` must be overridden if the context is not of type
    ``TensorFeatureContext``.

    :params manager: the manager used to create this vectorizer that has
                     resources needed to encode and decode

    :type manager: FeatureDocumentVectorizer

    """
    manager: Any

    def transform(self, data: Any) -> torch.Tensor:
        """Use the output of the encoding as input to the decoding to directly produce
        the output tensor ready to be used in testing, training, validation
        etc.

        """
        context = self.encode(data)
        return self.decode(context)

    def encode(self, data: Any) -> FeatureContext:
        """Encode data to a context ready to (potentially) be pickled.

        """
        return self._encode(data)

    def decode(self, context: FeatureContext) -> torch.Tensor:
        """Decode a (potentially) unpickled context and return a tensor using the
        manager's ``torch_config``.

        """
        self._validate_context(context)
        return self._decode(context)

    @property
    def torch_config(self) -> TorchConfig:
        return self.manager.torch_config

    @abstractmethod
    def _encode(self, data: Any) -> FeatureContext:
        pass

    def _decode(self, context: FeatureContext) -> torch.Tensor:
        if isinstance(context, TensorFeatureContext):
            return context.tensor
        else:
            cstr = str(context) if context is None else context.__class__
            raise ValueError(f'unknown context: {cstr}')

    def _validate_context(self, context: FeatureContext):
        if context.feature_type != self.feature_type:
            raise ValueError(f'context meant for {context.feature_type} ' +
                             f'routed to {self.feature_type}')



# manager
@dataclass
class FeatureVectorizerManager(object):
    """Creates and manages instances of ``EncodableFeatureVectorizer`` and
    parses text in to feature based document.

    This handles encoding data into a context, which is data ready to be
    pickled on the file system with the idea this intermediate state is
    expensive to create.  At training time, the context is brought back in to
    memory and efficiently decoded in to a tensor.

    This class keeps track of two kinds of vectorizers:
    - module: registered with ``register_vectorizer`` in Python modules
    - configured: registered at instance create time in
                  ``configured_vectorizers``

    :see EncodableFeatureVectorizer:
    :see parse:

    """
    ATTR_EXP_META = ('torch_config', 'configured_vectorizers')
    VECTORIZERS = {}

    name: str
    config_factory: ConfigFactory
    torch_config: TorchConfig
    module_vectorizers: Set[str]
    configured_vectorizers: Set[str]

    def __post_init__(self):
        if self.module_vectorizers is None:
            self.module_vectorizers = set(self.VECTORIZERS.keys())

    @classmethod
    def register_vectorizer(self, cls: Type[EncodableFeatureVectorizer]):
        """Register static (class space) vectorizer, typically right after the
        definition of the class.

        """
        key = cls.FEATURE_TYPE
        if key in self.VECTORIZERS:
            s = f'{cls} is already registered under \'{key}\' in {self}'
            if 1:
                logger.warning(s)
            else:
                # this breaks ImportConfigFactory reloads
                raise ValueError(s)
        self.VECTORIZERS[key] = cls

    def transform(self, data: Any) -> \
            Tuple[torch.Tensor, EncodableFeatureVectorizer]:
        """Return a tuple of duples with the output tensor of a vectorizer and the
        vectorizer that created the output.  Every vectorizer listed in
        ``feature_types`` is used.

        """
        return tuple(map(lambda vec: (vec.transform(data), vec),
                         self.vectorizers.values()))

    @property
    @persisted('_vectorizers')
    def vectorizers(self) -> Dict[str, FeatureVectorizer]:
        """Return a dictionary of all registered vectorizers.  This includes both
        module and configured vectorizers.  The keys are the ``feature_type``s
        and values are the contained vectorizers.

        """
        vectorizers = collections.OrderedDict()
        ftypes = set(self.module_vectorizers)
        vec_classes = dict(self.VECTORIZERS)
        conf_instances = {}
        if self.configured_vectorizers is not None:
            for sec in self.configured_vectorizers:
                vec = self.config_factory(sec, manager=self)
                conf_instances[vec.feature_type] = vec
                ftypes.add(vec.feature_type)
        for feature_type in sorted(ftypes):
            inst = conf_instances.get(feature_type)
            if inst is None:
                inst = vec_classes[feature_type](self)
            vectorizers[feature_type] = inst
        return vectorizers

    def __getitem__(self, name: str) -> FeatureVectorizer:
        fv = self.vectorizers.get(name)
        if fv is None:
            raise KeyError(f"manager '{self}' has no vectorizer: '{name}'")
        return fv

    @property
    @persisted('_feature_types')
    def feature_types(self) -> Set[str]:
        """Get the feature types supported by this manager, which are the keys of the
        vectorizer.

        :see vectorizers:

        """
        return set(self.vectorizers.keys())


@dataclass
class FeatureVectorizerManagerSet(object):
    """A set of managers used collectively to encode and decode a series of
    features across many different kinds of data (i.e. labels, language
    features, numeric).

    """
    config_factory: ConfigFactory = field(repr=False)
    names: List[str]

    @property
    @persisted('_managers')
    def managers(self) -> Dict[str, FeatureVectorizerManager]:
        return {k: self.config_factory(k) for k in self.names}

    @property
    @persisted('_feature_types')
    def feature_types(self) -> Set[str]:
        return set(chain.from_iterable(
            map(lambda m: m.feature_types, self.values())))

    def __getitem__(self, name: str) -> FeatureVectorizerManager:
        return self.managers[name]

    def values(self) -> List[FeatureVectorizerManager]:
        return self.managers.values()

    def keys(self) -> Set[str]:
        return set(self.managers.keys())



# vectorizers
@dataclass
class CategoryEncodableFeatureVectorizer(EncodableFeatureVectorizer):
    """Vectorize from a list of nominals.  This is useful for encoding labels for
    the categorization machine learning task.

    """
    NAME = 'category label encoder'

    categories: Set[str]
    feature_type: str

    def __post_init__(self):
        super().__post_init__()
        le = LabelEncoder()
        le.fit(self.categories)
        llen = len(le.classes_)
        arr = self.manager.torch_config.zeros((llen, llen))
        for i in range(llen):
            arr[i][i] = 1
        self.label_encoder = le
        self.identity = arr

    def _get_shape(self):
        return -1, len(self.label_encoder.classes_)

    def _encode(self, category_instances: List[str]) -> FeatureContext:
        tc = self.manager.torch_config
        arr = tc.empty((len(category_instances), self.identity.shape[0]))
        indicies = self.label_encoder.transform(category_instances)
        for i, idx in enumerate(indicies):
            arr[i] = self.identity[idx]
        return SparseTensorFeatureContext.instance(
            self.feature_type, arr, self.manager.torch_config)

    def _decode(self, context: FeatureContext) -> torch.Tensor:
        ctx: SparseTensorFeatureContext = context
        return ctx.to_tensor(self.manager.torch_config)


@dataclass
class SeriesEncodableFeatureVectorizer(EncodableFeatureVectorizer):
    """Vectorize a Pandas series, such as a list of rows.  This vectorizer has an
    undefined shape since both the number of columns and rows are not specified at
    runtime.

    """
    NAME = 'pandas series'

    feature_type: str

    def _get_shape(self):
        return -1, -1,

    def _encode(self, rows: Iterable[pd.Series]) -> FeatureContext:
        narrs = []
        tc = self.manager.torch_config
        nptype = tc.numpy_data_type
        for row in rows:
            narrs.append(row.to_numpy(dtype=nptype))
        arr = np.stack(narrs)
        arr = tc.from_numpy(arr)
        return TensorFeatureContext(self.feature_type, arr)
