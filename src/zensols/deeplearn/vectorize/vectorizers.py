"""Vectorizer implementations.

"""
__author__ = 'Paul Landes'

from typing import Set, List, Iterable, Union, Any, Tuple, Dict
from dataclasses import dataclass, field
import sys
import logging
import pandas as pd
import numpy as np
import itertools as it
from io import TextIOBase
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import torch
from torch import Tensor
from torch import nn
from zensols.persist import persisted
from zensols.deeplearn import TorchTypes, TorchConfig
from . import (
    VectorizerError,
    FeatureVectorizer,
    EncodableFeatureVectorizer,
    TensorFeatureContext,
    FeatureContext,
    MultiFeatureContext,
)

logger = logging.getLogger(__name__)


@dataclass
class IdentityEncodableFeatureVectorizer(EncodableFeatureVectorizer):
    """An identity vectorizer, which encodes tensors verbatim, or concatenates a
    list of tensors in to one tensor of the same dimension.

    """
    DESCRIPTION = 'identity function encoder'

    def _get_shape(self) -> Tuple[int]:
        return -1,

    def _encode(self, obj: Union[list, Tensor]) -> Tensor:
        if isinstance(obj, Tensor):
            arr = obj
        else:
            tc = self.torch_config
            if len(obj[0].shape) == 0:
                arr = tc.singleton(obj, dtype=obj[0].dtype)
            else:
                arr = torch.cat(obj)
        return TensorFeatureContext(self.feature_id, arr)


@dataclass
class CategoryEncodableFeatureVectorizer(EncodableFeatureVectorizer):
    """A base class that vectorizies nominal categories in to integer indexes.

    """
    categories: Set[str] = field()
    """A list of string enumerated values."""

    def __post_init__(self):
        super().__post_init__()
        if len(self.categories) == 0:
            raise VectorizerError(f'No categories given: <{self.categories}>')
        self.label_encoder = LabelEncoder()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'encoding categories: <{self.categories}>')
        self.label_encoder.fit(self.categories)

    @staticmethod
    def _str_to_dtype(data_type: str, torch_config: TorchConfig) -> torch.dtype:
        if data_type is None:
            data_type = torch.int64
        else:
            data_type = TorchTypes.type_from_string(data_type)
        return data_type

    @property
    @persisted('_by_label')
    def by_label(self) -> Dict[str, int]:
        le = self.label_encoder
        return dict(zip(le.classes_, le.transform(le.classes_)))

    def get_classes(self, nominals: Iterable[int]) -> List[str]:
        """Return the label string values for indexes ``nominals``.

        :param nominals: the integers that map to the respective string class

        """
        return self.label_encoder.inverse_transform(nominals)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        super().write(depth, writer)
        le: LabelEncoder = self.label_encoder
        self._write_line('labels:', depth, writer)
        for cat, ix in zip(le.classes_, le.transform(le.classes_)):
            self._write_line(f'{cat}: {ix}', depth + 1, writer)


@dataclass
class NominalEncodedEncodableFeatureVectorizer(
        CategoryEncodableFeatureVectorizer):
    """Map each label to a nominal, which is useful for class labels.

    :shape: (1, 1)

    """
    DESCRIPTION = 'nominal encoder'

    data_type: Union[str, None, torch.dtype] = field(default=None)
    """The type to use for encoding, which if a string, must be a key in of
    :obj:`.TorchTypes.NAME_TO_TYPE`.

    """
    decode_one_hot: bool = field(default=False)
    """If ``True``, during decoding create a one-hot encoded tensor of shape
    ``(N, |labels|)``.

    """
    def __post_init__(self):
        super().__post_init__()
        self.data_type = self._str_to_dtype(self.data_type, self.torch_config)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'init categories: {self.categories}')

    def _get_shape(self) -> Tuple[int]:
        return (1, 1)

    def _str_to_dtype(self, data_type: str,
                      torch_config: TorchConfig) -> torch.dtype:
        if data_type is None:
            data_type = torch.int64
        else:
            data_type = TorchTypes.type_from_string(data_type)
        return data_type

    def _encode(self, category_instances: List[str]) -> FeatureContext:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'encode categories: {category_instances} ' +
                         f'(one of {self.categories})')
        if not isinstance(category_instances, (tuple, list)):
            raise VectorizerError(
                f'expecting list but got: {type(category_instances)}')
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'instances: {category_instances}')
        indicies = self.label_encoder.transform(category_instances)
        singleton = self.torch_config.singleton
        arr = singleton(indicies, dtype=self.data_type)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'encoding cat arr: {arr.dtype}')
        return TensorFeatureContext(self.feature_id, arr)

    def _decode(self, context: FeatureContext) -> Tensor:
        arr = super()._decode(context)
        if self.decode_one_hot:
            batches = arr.shape[0]
            he = self.torch_config.zeros((batches, len(self.categories)),
                                         dtype=torch.long)
            for row in range(batches):
                idx = arr[row]
                he[row][idx] = 1
            del arr
            arr = he
        return arr


@dataclass
class NominalMultiLabelEncodedEncodableFeatureVectorizer(
        EncodableFeatureVectorizer):
    """Map each label to a nominal, which is useful for class labels.

    :shape: (1, |categories|)

    """
    DESCRIPTION = 'nominal encoder'

    categories: Set[str] = field()
    """A list of string enumerated values."""

    data_type: Union[str, None, torch.dtype] = field(default=None)
    """The type to use for encoding, which if a string, must be a key in of
    :obj:`.TorchTypes.NAME_TO_TYPE`.

    """
    def __post_init__(self):
        super().__post_init__()
        self.data_type = CategoryEncodableFeatureVectorizer._str_to_dtype(
            self.data_type, self.torch_config)
        self.label_binarizer = MultiLabelBinarizer(classes=self.categories)
        self.label_binarizer.fit([self.categories])
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'init categories: {self.categories}')

    def _get_shape(self) -> Tuple[int]:
        return (1, len(self.label_binarizer.classes_))

    def _encode(self, category_instances: List[Tuple[str, ...]]) -> \
            FeatureContext:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'encode categories: {category_instances} ' +
                         f'(one of {self.categories})')
        if not isinstance(category_instances, (tuple, list)):
            raise VectorizerError(
                f'expecting list but got: {type(category_instances)}')
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'instances: {category_instances}')
        indicies = self.label_binarizer.transform(category_instances)
        singleton = self.torch_config.singleton
        arr = singleton(indicies, dtype=self.data_type)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'encoding cat arr: {arr.dtype}')
        return TensorFeatureContext(self.feature_id, arr)


@dataclass
class OneHotEncodedEncodableFeatureVectorizer(
        CategoryEncodableFeatureVectorizer):
    """Vectorize from a list of nominals.  This is useful for encoding labels
    for the categorization machine learning task.

    :shape: (1,) when optimizing bools and classes = 2, else (1, |categories|)

    """
    DESCRIPTION = 'category encoder'

    optimize_bools: bool = field()
    """If ``True``, more efficiently represent boolean encodings."""

    def __post_init__(self):
        super().__post_init__()
        le = self.label_encoder
        llen = len(le.classes_)
        if not self.optimize_bools or llen != 2:
            arr = self.torch_config.zeros((llen, llen))
            for i in range(llen):
                arr[i][i] = 1
            self.identity = arr

    def _get_shape(self) -> Tuple[int]:
        n_classes = len(self.label_encoder.classes_)
        if self.optimize_bools and n_classes == 2:
            return (1,)
        else:
            return (-1, n_classes)

    def _encode_cats(self, category_instances: List[str], arr: Tensor) -> \
            Tuple[int, FeatureContext]:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'encoding: {self.category_instances}')
        tc = self.torch_config
        indicies = self.label_encoder.transform(category_instances)
        is_one_row = self.shape[0] == 1
        if is_one_row:
            if arr is None:
                arr = tc.singleton(indicies)
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'creating: {self.identity.shape}')
            if arr is None:
                arr = tc.empty(
                    (len(category_instances), self.identity.shape[0]))
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'created: {arr.dtype}')
            for i, idx in enumerate(it.islice(indicies, arr.size(0))):
                arr[i] = self.identity[idx]
        return is_one_row, arr

    def _encode(self, category_instances: List[str]) -> FeatureContext:
        is_one_row, arr = self._encode_cats(category_instances, None)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'encoding cat arr: {arr.dtype}')
        return TensorFeatureContext(self.feature_id, arr)


@dataclass
class AggregateEncodableFeatureVectorizer(EncodableFeatureVectorizer):
    """Use another vectorizer to vectorize each instance in an iterable.  Each
    iterable is then concatenated in to a single tensor on decode.

    **Important**: you must add the delegate vectorizer to the same vectorizer
    manager set as this instance since it uses the manager to find it.

    :shape: (-1, delegate.shape[1] * (2 ^ add_mask))

    """
    DESCRIPTION = 'aggregate vectorizer'

    DEFAULT_PAD_LABEL = nn.CrossEntropyLoss().ignore_index
    """The default value used for :obj:`pad_label`, which is used since this
    vectorizer is most often used to encode labels.

    """
    delegate_feature_id: str = field()
    """The feature ID of the delegate vectorizer to use (configured in same
    vectorizer manager).

    """
    size: int = field(default=-1)
    """The second dimension size of the tensor to create when decoding."""

    pad_label: int = field(default=DEFAULT_PAD_LABEL)
    """The numeric label to use for padded elements.  This defaults to
    :obj:`~torch.nn.CrossEntry.ignore_index`."""

    def _get_shape(self):
        return -1, *self.delegate.shape[1:]

    @property
    def delegate(self) -> EncodableFeatureVectorizer:
        return self.manager[self.delegate_feature_id]

    def _encode(self, datas: Iterable[Iterable[Any]]) -> MultiFeatureContext:
        vec = self.delegate
        ctxs = tuple(map(lambda d: vec.encode(d), datas))
        return MultiFeatureContext(self.feature_id, ctxs)

    @persisted('_pad_tensor_pw')
    def _pad_tensor(self, data_type: torch.dtype,
                    device: torch.device) -> Tensor:
        return torch.tensor([self.pad_label], device=device, dtype=data_type)

    def create_padded_tensor(self, size: torch.Size,
                             data_type: torch.dtype = None,
                             device: torch.device = None):
        """Create a tensor with all elements set to :obj:`pad_label`.

        :param size: the dimensions of the created tensor

        :param data_type: the data type of the new tensor

        """
        data_type = self.delegate.data_type if data_type is None else data_type
        device = self.torch_config.device if device is None else device
        pad = self._pad_tensor(data_type, device)
        if pad.dtype != data_type or pad.device != device:
            pad = torch.tensor(
                [self.pad_label], device=device, dtype=data_type)
        return pad.repeat(size)

    def _decode(self, context: MultiFeatureContext) -> Tensor:
        vec: FeatureVectorizer = self.delegate
        srcs: Tuple[Tensor] = tuple(
            map(lambda c: vec.decode(c), context.contexts))
        clen: int = len(srcs)
        first: Tensor = srcs[0]
        dtype: torch.dtype = first.dtype
        mid_dims: int = first.shape[1:]
        sz: int
        if self.size > 0:
            sz = self.size
        else:
            sz = max(map(lambda t: t.size(0), srcs))
        arr = self.create_padded_tensor((clen, sz, *mid_dims), dtype)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'num contexts: {clen}, dtype={dtype}, ' +
                         f'src={first.shape}, dst={arr.shape}, ' +
                         f'mid_dims={mid_dims}')
        rowix = 0
        for carr in srcs:
            lsz = min(carr.size(0), sz)
            if carr.dim() == 1:
                arr[rowix, :lsz] = carr[:lsz]
            elif carr.dim() == 2:
                arr[rowix, :lsz, :] = carr[:lsz, :]
            elif carr.dim() == 3:
                arr[rowix, :lsz, :, :] = carr[:lsz, :, :]
            rowix += 1
        return arr


@dataclass
class MaskFeatureContext(FeatureContext):
    """A feature context used for the :class:`.MaskFeatureVectorizer`
    vectorizer.

    :param sequence_lengths: the lengths of all each row to mask

    """
    sequence_lengths: Tuple[int]


@dataclass
class MaskFeatureVectorizer(EncodableFeatureVectorizer):
    """Creates masks where the first N elements of a vector are 1's with the
    rest 0's.

    :shape: (-1, size)

    """
    DESCRIPTION = 'mask'

    size: int = field(default=-1)
    """The length of all mask vectors or ``-1`` make the length the max size of
    the sequence in the batch.

    """
    data_type: Union[str, None, torch.dtype] = field(default='bool')
    """The mask tensor type.  To use the int type that matches the resolution of
    the manager's :obj:`torch_config`, use ``DEFAULT_INT``.

    """
    def __post_init__(self):
        super().__post_init__()
        self.data_type = self.str_to_dtype(self.data_type, self.torch_config)
        if self.size > 0:
            tc = self.torch_config
            self.ones = tc.ones((self.size,), dtype=self.data_type)
        else:
            self.ones = None

    @staticmethod
    def str_to_dtype(data_type: str, torch_config: TorchConfig) -> torch.dtype:
        if data_type == 'DEFAULT_INT':
            data_type = torch_config.int_type
        else:
            data_type = TorchTypes.type_from_string(data_type)
        return data_type

    def _get_shape(self):
        return -1, self.size,

    def _encode(self, datas: Iterable[Iterable[Any]]) -> FeatureContext:
        lens = tuple(map(lambda d: sum(1 for _ in d), datas))
        return MaskFeatureContext(self.feature_id, lens)

    def _decode(self, context: MaskFeatureContext) -> Tensor:
        tc = self.torch_config
        batch_size = len(context.sequence_lengths)
        lens = context.sequence_lengths
        if self.ones is None:
            # when no configured size is given, recreate for each batch
            sz = max(lens)
            ones = self.torch_config.ones((sz,), dtype=self.data_type)
        else:
            # otherwise, the mask was already created in the initializer
            sz = self.size
            ones = self.ones
        arr = tc.zeros((batch_size, sz), dtype=self.data_type)
        for bix, slen in enumerate(lens):
            arr[bix, :slen] = ones[:slen]
        return arr


@dataclass
class SeriesEncodableFeatureVectorizer(EncodableFeatureVectorizer):
    """Vectorize a Pandas series, such as a list of rows.  This vectorizer has
    an undefined shape since both the number of columns and rows are not
    specified at runtime.

    :shape: (-1, 1)

    """
    DESCRIPTION = 'pandas series'

    def _get_shape(self):
        return -1, -1

    def _encode(self, rows: Iterable[pd.Series]) -> FeatureContext:
        narrs = []
        tc = self.torch_config
        nptype = tc.numpy_data_type
        for row in rows:
            narrs.append(row.to_numpy(dtype=nptype))
        arr = np.stack(narrs)
        arr = tc.from_numpy(arr)
        return TensorFeatureContext(self.feature_id, arr)


@dataclass
class AttributeEncodableFeatureVectorizer(EncodableFeatureVectorizer):
    """Vectorize a iterable of floats.  This vectorizer has an undefined shape
    since both the number of columns and rows are not specified at runtime.

    :shape: (1,)

    """
    DESCRIPTION = 'single attribute'

    def _get_shape(self):
        return 1,

    def _encode(self, data: Iterable[float]) -> FeatureContext:
        arr = self.torch_config.from_iterable(data)
        return TensorFeatureContext(self.feature_id, arr)
