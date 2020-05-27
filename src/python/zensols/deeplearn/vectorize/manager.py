"""Vectorization base classes and basic functionality.

"""
__author__ = 'Paul Landes'


import logging
from abc import abstractmethod, ABCMeta
from dataclasses import dataclass, field
from typing import Tuple, Any, Set, Type, Dict, List
from itertools import chain
import collections
import torch
from zensols.persist import persisted
from zensols.config import ConfigFactory
from zensols.deeplearn import TorchConfig
from . import FeatureVectorizer, FeatureContext, TensorFeatureContext

logger = logging.getLogger(__name__)


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

    :type manager: FeatureVectorizerManager

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
            raise ValueError('module_vectorizers must be configured')
            #self.module_vectorizers = set(self.VECTORIZERS.keys())

    @classmethod
    def register_vectorizer(self, cls: Type[EncodableFeatureVectorizer]):
        """Register static (class space) vectorizer, typically right after the
        definition of the class.

        """
        key = cls.FEATURE_TYPE
        logger.debug(f'registering vectorizer: {key} -> {cls}')
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
        return self._create_vectorizers()

    def _create_vectorizers(self) -> Dict[str, FeatureVectorizer]:
        vectorizers = collections.OrderedDict()
        ftypes = set(self.module_vectorizers)
        vec_classes = dict(self.VECTORIZERS)
        conf_instances = {}
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'class registered vectorizers: {self.VECTORIZERS}')
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