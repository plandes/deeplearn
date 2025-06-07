"""Vectorization base classes and basic functionality.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import Tuple, Any, Set, Dict, List, Iterable
from dataclasses import dataclass, field
from abc import abstractmethod, ABCMeta
import logging
import sys
from itertools import chain
import collections
from io import TextIOBase
from torch import Tensor
from zensols.persist import persisted, PersistedWork
from zensols.deeplearn import TorchConfig
from . import (
    VectorizerError, ConfigurableVectorization, FeatureVectorizer,
    FeatureContext, TensorFeatureContext, SparseTensorFeatureContext,
    NullFeatureContext, MultiFeatureContext,
)

logger = logging.getLogger(__name__)


@dataclass
class EncodableFeatureVectorizer(FeatureVectorizer, metaclass=ABCMeta):
    """This vectorizer splits transformation up in to encoding and decoding.
    The encoded state as a ``FeatureContext``, in cases where encoding is
    prohibitively expensive, is computed once and pickled to the file system.
    It is then loaded and finally decoded into a tensor.

    Examples include computing an encoding as indexes of a word embedding
    during the encoding phase.  Then generating the full embedding layer during
    decoding.  Note that this decoding is done with a ``TorchConfig`` so the
    output tensor goes directly to the GPU.

    This abstract base class only needs the ``_encode`` method overridden.  The
    ``_decode`` must be overridden if the context is not of type
    ``TensorFeatureContext``.

    """
    manager: FeatureVectorizerManager = field()
    """The manager used to create this vectorizer that has resources needed to
    encode and decode.

    """
    def transform(self, data: Any) -> Tensor:
        """Use the output of the encoding as input to the decoding to directly
        produce the output tensor ready to be used in testing, training,
        validation etc.

        """
        context = self.encode(data)
        return self.decode(context)

    def encode(self, data: Any) -> FeatureContext:
        """Encode data to a context ready to (potentially) be pickled.

        """
        return self._encode(data)

    def decode(self, context: FeatureContext) -> Tensor:
        """Decode a (potentially) unpickled context and return a tensor using
        the manager's :obj:`torch_config`.

        """
        arr: Tensor = None
        self._validate_context(context)
        if isinstance(context, NullFeatureContext):
            pass
        elif isinstance(context, MultiFeatureContext) and context.is_empty:
            arr = NullFeatureContext(context.feature_id)
        else:
            arr = self._decode(context)
        return arr

    @property
    def torch_config(self) -> TorchConfig:
        """The torch configuration used to create encoded/decoded tensors.

        """
        return self.manager.torch_config

    @abstractmethod
    def _encode(self, data: Any) -> FeatureContext:
        pass

    def _decode(self, context: FeatureContext) -> Tensor:
        arr: Tensor
        if isinstance(context, NullFeatureContext):
            arr = None
        elif isinstance(context, TensorFeatureContext):
            arr = context.tensor
        elif isinstance(context, SparseTensorFeatureContext):
            arr = context.to_tensor(self.manager.torch_config)
        else:
            cstr = str(context) if context is None else context.__class__
            raise VectorizerError(f'Unknown context: {cstr}')
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'decoded {type(context)} to {arr.shape}')
        return arr

    def _validate_context(self, context: FeatureContext):
        if context.feature_id != self.feature_id:
            raise VectorizerError(f'Context meant for {context.feature_id} ' +
                                  f'routed to {self.feature_id}')


@dataclass
class TransformableFeatureVectorizer(EncodableFeatureVectorizer,
                                     metaclass=ABCMeta):
    """Instances of this class use the output of
    :meth:`.EncodableFeatureVectorizer.transform` (chain encode and decode) as
    the output of :meth:`EncodableFeatureVectorizer.encode`, then passes
    through the decode.

    This is useful if the decoding phase is very expensive and you'd rather
    take that hit when creating batches written to the file system.

    """
    encode_transformed: bool = field()
    """If ``True``, enable the transformed output of the encoding step as the
    decode step (see class docs).

    """
    def encode(self, data: Any) -> FeatureContext:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'encoding {type(data)}, also decode after encode' +
                         f'{self.encode_transformed}')
        if self.encode_transformed:
            ctx: FeatureContext = self._encode(data)
            arr: Tensor = self._decode(ctx)
            ctx = TensorFeatureContext(ctx.feature_id, arr)
        else:
            ctx = super().encode(data)
        return ctx

    def decode(self, context: FeatureContext) -> Tensor:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'decoding {type(context)}, already decoded: ' +
                         f'{self.encode_transformed}')
        if self.encode_transformed:
            ctx: TensorFeatureContext = context
            arr = ctx.tensor
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'already decoded: {arr.shape}')
        else:
            arr = super().decode(context)
        return arr


# manager
@dataclass
class FeatureVectorizerManager(ConfigurableVectorization):
    """Creates and manages instances of :class:`.EncodableFeatureVectorizer` and
    parses text in to feature based document.

    This handles encoding data into a context, which is data ready to be
    pickled on the file system with the idea this intermediate state is
    expensive to create.  At training time, the context is brought back in to
    memory and efficiently decoded in to a tensor.

    This class keeps track of two kinds of vectorizers:

        * module: registered with ``register_vectorizer`` in Python modules

        * configured: registered at instance create time in
                      ``configured_vectorizers``

    Instances of this class act like a :class:`dict` of all registered
    vectorizers.  This includes both module and configured vectorizers.  The
    keys are the ``feature_id``s and values are the contained vectorizers.

    :see: :class:`.EncodableFeatureVectorizer`

    """
    ATTR_EXP_META = ('torch_config', 'configured_vectorizers')
    MANAGER_SEP = '.'

    torch_config: TorchConfig = field()
    """The torch configuration used to encode and decode tensors."""

    configured_vectorizers: Set[str] = field()
    """Configuration names of vectorizors to use by this manager."""

    def __post_init__(self):
        super().__post_init__()
        self.manager_set = None
        self._vectorizers_pw = PersistedWork('_vectorizers_pw', self)

    def transform(self, data: Any) -> \
            Tuple[Tensor, EncodableFeatureVectorizer]:
        """Return a tuple of duples with the output tensor of a vectorizer and
        the vectorizer that created the output.  Every vectorizer listed in
        ``feature_ids`` is used.

        """
        return tuple(map(lambda vec: (vec.transform(data), vec),
                         self._vectorizers.values()))

    @property
    @persisted('_vectorizers_pw')
    def _vectorizers(self) -> Dict[str, FeatureVectorizer]:
        """Return a dictionary of all registered vectorizers.  This includes
        both module and configured vectorizers.  The keys are the
        ``feature_id``s and values are the contained vectorizers.

        """
        return self._create_vectorizers()

    def _create_vectorizers(self) -> Dict[str, FeatureVectorizer]:
        vectorizers: Dict[str, FeatureVectorizer] = collections.OrderedDict()
        conf_instances: Dict[str, FeatureVectorizer] = {}
        feature_ids: Set[str] = set()
        if self.configured_vectorizers is not None:
            sec: str
            for sec in self.configured_vectorizers:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'creating vectorizer {sec}')
                if sec.find(self.MANAGER_SEP) >= 0:
                    raise VectorizerError(
                        f'Separator {self.MANAGER_SEP} not ' +
                        f'allowed in names: {sec}')
                vec: FeatureVectorizer = self.config_factory(sec, manager=self)
                conf_instances[vec.feature_id] = vec
                feature_ids.add(vec.feature_id)
        for feature_id in sorted(feature_ids):
            inst: FeatureVectorizer = conf_instances.get(feature_id)
            vectorizers[feature_id] = inst
        return vectorizers

    @property
    @persisted('_feature_ids')
    def feature_ids(self) -> Set[str]:
        """Get the feature ids supported by this manager, which are the keys of
        the vectorizer.

        :see: :class:`.FeatureVectorizerManager`

        """
        return frozenset(self._vectorizers.keys())

    def get(self, name: str) -> FeatureVectorizer:
        """Return the feature vectorizer named ``name``."""
        fv = self._vectorizers.get(name)
        # if we can't find the vectorizer, try using dot syntax to find it in
        # the parent manager set
        if name is not None and fv is None:
            idx = name.find(self.MANAGER_SEP)
            if self.manager_set is not None and idx > 0:
                mng_name, vec = name[:idx], name[idx + 1:]
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'looking up {mng_name}:{vec}')
                mng = self.manager_set.get(mng_name)
                if mng is not None:
                    fv = mng._vectorizers.get(vec)
        return fv

    def keys(self) -> Iterable[str]:
        return self._vectorizers.keys()

    def values(self) -> Iterable[FeatureVectorizer]:
        return self._vectorizers.values()

    def items(self) -> Iterable[Tuple[str, FeatureVectorizer]]:
        return self._vectorizers.items()

    def __len__(self) -> int:
        return len(self._vectorizers)

    def __getitem__(self, name: str) -> FeatureVectorizer:
        fv = self.get(name)
        if fv is None:
            raise VectorizerError(
                f"Manager '{self.name}' has no vectorizer: '{name}'")
        return fv

    def deallocate(self):
        if self._vectorizers_pw.is_set():
            vecs = self._vectorizers
            for vec in vecs.values():
                vec.deallocate()
            vecs.clear()
        super().deallocate()

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(str(self), depth, writer)
        for vec in self._vectorizers.values():
            vec.write(depth + 1, writer)


@dataclass
class FeatureVectorizerManagerSet(ConfigurableVectorization):
    """A set of managers used collectively to encode and decode a series of
    features across many different kinds of data (i.e. labels, language
    features, numeric).

    In the same way a :class:`.FeatureVectorizerManager` acts like a
    :class:`dict`, this class is a ``dict`` for
    :class:`.FeatureVectorizerManager` instances.

    """
    ATTR_EXP_META = ('_managers',)

    names: List[str] = field()
    """The sections defining :class:`.FeatureVectorizerManager` instances."""

    def __post_init__(self):
        super().__post_init__()
        self._managers_pw = PersistedWork('_managers_pw', self)

    @property
    @persisted('_managers_pw')
    def _managers(self) -> Dict[str, FeatureVectorizerManager]:
        """All registered vectorizer managers of the manager."""
        mngs = {}
        for n in self.names:
            f: FeatureVectorizerManager = self.config_factory(n)
            if not isinstance(f, FeatureVectorizerManager):
                raise VectorizerError(
                    f"Config section '{n}' does not define a " +
                    f'FeatureVectoizerManager: {f}')
            f.manager_set = self
            mngs[n] = f
        return mngs

    def get_vectorizer_names(self) -> Iterable[str]:
        """Return the names of vectorizers across all vectorizer managers."""
        return map(lambda vec: vec.name,
                   chain.from_iterable(
                       map(lambda vm: vm.values(), self.values())))

    def get_vectorizer(self, name: str) -> FeatureVectorizer:
        """Find vectorizer with ``name`` in all vectorizer managers.

        """
        for vm in self.values():
            for vec in vm.values():
                if name == vec.name:
                    return vec

    @property
    @persisted('_feature_ids')
    def feature_ids(self) -> Set[str]:
        """Return all feature IDs supported across all manager registered with
        the manager set.

        """
        return set(chain.from_iterable(
            map(lambda m: m.feature_ids, self.values())))

    def __getitem__(self, name: str) -> FeatureVectorizerManager:
        mng = self._managers.get(name)
        if mng is None:
            raise VectorizerError(
                f"No such manager '{name}' in manager set '{self.name}'")
        return mng

    def get(self, name: str) -> FeatureVectorizerManager:
        return self._managers.get(name)

    def values(self) -> List[FeatureVectorizerManager]:
        return self._managers.values()

    def keys(self) -> Set[str]:
        return set(self._managers.keys())

    def deallocate(self):
        if self._managers_pw.is_set():
            mngs = self._managers
            for mng in mngs.values():
                mng.deallocate()
            mngs.clear()
        super().deallocate()

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(f'{self.name}', depth, writer)
        for mng in self._managers.values():
            mng.write(depth + 1, writer)
