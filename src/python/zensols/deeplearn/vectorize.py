"""Vectorization base classes and basic functionality

"""
__author__ = 'Paul Landes'


import logging
from abc import abstractmethod, ABC, ABCMeta
from dataclasses import dataclass, field
from typing import Tuple, Any, Set, Type, Dict
import collections
import torch
from zensols.persist import PersistableContainer, persisted
from zensols.config import ConfigFactory
from zensols.deeplearn import TorchConfig

logger = logging.getLogger(__name__)


# vectorize
@dataclass
class FeatureVectorizer(ABC):
    def __post_init__(self):
        if not hasattr(self, '_feature_type') and \
           hasattr(self.__class__, 'FEATURE_TYPE'):
            self._feature_type = self.FEATURE_TYPE
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
        """A human readable name.

        """
        return self._name

    @property
    def feature_type(self) -> str:
        """The description of the feature.  At the token level, this is the feature
        type name found in an instance of ``TokenAttributes``.

        :see TokenAttributes:

        """
        return self._feature_type

    @feature_type.setter
    def feature_type(self, feature_type):
        self._feature_type = feature_type

    def __str__(self):
        return self._name

    def __repr__(self):
        return f'{self.__class__}: {self.__str__()}'


@dataclass
class FeatureContext(PersistableContainer):
    """Data created by coding and meant to be pickled on the file system.

    :attribute feature_type: the feature type of the ``FeatureVectorizer`` that
                             created this context.

    :see EncodableFeatureVectorizer.encode:

    """
    feature_type: int


@dataclass
class TensorFeatureContext(FeatureContext):
    """A context that encodes data directly to a tensor.  This tensor could be a
    sparse matrix becomes dense during the decoding process.

    :attribute tensor: the output tensor of the encoding phase

    """
    tensor: torch.Tensor


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

    :attributes manager: the manager used to create this vectorizer that has
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
        """
        """
        if isinstance(context, TensorFeatureContext):
            return context.tensor
        else:
            cstr = str(context) if context is None else context.__class__
            raise ValueError(f'unknown context: {cstr}')

    def _validate_context(self, context: FeatureContext):
        if context.feature_type != self.feature_type:
            raise ValueError(f'context meant for {context.feature_type} ' +
                             f'routed to {self.feature_type}')


@dataclass
class FeatureVectorizerManager(object):
    """Creates and manages instances of ``EncodableFeatureVectorizer`` and
    parses text in to feature based document.

    This handles encoding data into a context, which is data ready to be
    pickled on the file system with the idea this intermediate state is
    expensive to create.  At training time, the context is brought back in to
    memory and efficiently decoded in to a tensor.

    :see EncodableFeatureVectorizer:
    :see parse:

    """
    ATTR_EXP_META = ('torch_config', 'configured_vectorizers')
    VECTORIZERS = {}

    config_factory: ConfigFactory
    torch_config: TorchConfig
    configured_vectorizers: Set[str]

    @classmethod
    def register_vectorizer(self, cls: Type[EncodableFeatureVectorizer]):
        key = cls.FEATURE_TYPE
        if key in self.VECTORIZERS:
            raise ValueError(
                f'{cls} is already registered under {key} in {self.__class__}')
        self.VECTORIZERS[key] = cls

    def transform(self, data: Any) -> \
            Tuple[torch.Tensor, EncodableFeatureVectorizer]:
        """Return a tuple of duples with the output tensor of a vectorizer and the
        vectorizer that created the output.  Every vectorizer listed in
        ``token_container_feature_types`` is used.

        """
        return tuple(map(lambda vec: (vec.transform(data), vec),
                         self.vectorizers.values()))

    @property
    @persisted('_vectorizers')
    def vectorizers(self) -> Dict[str, FeatureVectorizer]:
        vectorizers = collections.OrderedDict()
        ftypes = self.token_container_feature_types
        vec_classes = dict(self.VECTORIZERS)
        conf_instances = {}
        if self.configured_vectorizers is not None:
            for sec in self.configured_vectorizers:
                vec = self.config_factory(sec, manager=self)
                conf_instances[vec.feature_type] = vec
                ftypes.add(vec.feature_type)
        ftypes = sorted(ftypes)
        for feature_type in ftypes:
            inst = conf_instances.get(feature_type)
            if inst is None:
                inst = vec_classes[feature_type](self)
            vectorizers[feature_type] = inst
        return vectorizers
