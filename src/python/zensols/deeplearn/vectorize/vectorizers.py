"""Vectorizer implementations.

"""
__author__ = 'Paul Landes'

from typing import Set, List, Iterable, Union, Any, Tuple
from dataclasses import dataclass, field
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from zensols.deeplearn import TorchTypes, TorchConfig
from . import (
    EncodableFeatureVectorizer,
    TensorFeatureContext,
    SparseTensorFeatureContext,
    FeatureContext,
    MultiFeatureContext,
)

logger = logging.getLogger(__name__)


def str_to_dtype(data_type: str, torch_config: TorchConfig) -> torch.dtype:
    if data_type is None:
        data_type = torch_config.int_type
    else:
        data_type = TorchTypes.type_from_string(data_type)
    return data_type


@dataclass
class IdentityEncodableFeatureVectorizer(EncodableFeatureVectorizer):
    DESCRIPTION = 'identity function encoder'

    def _get_shape(self):
        return -1,

    def _encode(self, obj: Union[list, torch.Tensor]) -> torch.Tensor:
        if isinstance(obj, torch.Tensor):
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
    categories: Set[str]

    def __post_init__(self):
        super().__post_init__()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.categories)

    def _get_shape(self):
        return 1, len(self.categories)

    def get_classes(self, nominals: Iterable[int]) -> List[str]:
        return self.label_encoder.inverse_transform(nominals)


@dataclass
class NominalEncodedEncodableFeatureVectorizer(CategoryEncodableFeatureVectorizer):
    """Map each label to a nominal, which is useful for class labels.

    """
    DESCRIPTION = 'nominal encoder'
    data_type: Union[str, None, torch.dtype] = field(default=None)
    decode_one_hot: bool = field(default=False)

    def __post_init__(self):
        super().__post_init__()
        self.data_type = str_to_dtype(self.data_type, self.torch_config)

    def _encode(self, category_instances: List[str]) -> FeatureContext:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'categories: {category_instances} ' +
                         f'(one of {self.categories})')
        if not isinstance(category_instances, (tuple, list)):
            raise ValueError(
                f'expecting list but got: {type(category_instances)}')
        indicies = self.label_encoder.transform(category_instances)
        singleton = self.torch_config.singleton
        arr = singleton(indicies, dtype=self.data_type)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'encoding cat arr: {arr.dtype}')
        return TensorFeatureContext(self.feature_id, arr)

    def _decode(self, context: FeatureContext) -> torch.Tensor:
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
class OneHotEncodedEncodableFeatureVectorizer(CategoryEncodableFeatureVectorizer):
    """Vectorize from a list of nominals.  This is useful for encoding labels for
    the categorization machine learning task.

    """
    DESCRIPTION = 'category encoder'

    optimize_bools: bool = field(default=True)

    def __post_init__(self):
        super().__post_init__()
        le = self.label_encoder
        llen = len(le.classes_)
        if not self.optimize_bools or llen != 2:
            arr = self.torch_config.zeros((llen, llen))
            for i in range(llen):
                arr[i][i] = 1
            self.identity = arr

    def _get_shape(self):
        n_classes = len(self.label_encoder.classes_)
        if self.optimize_bools and n_classes == 2:
            return 1,
        else:
            return -1, n_classes

    def _encode(self, category_instances: List[str]) -> FeatureContext:
        tc = self.torch_config
        indicies = self.label_encoder.transform(category_instances)
        is_one_row = self.shape[0] == 1
        if is_one_row:
            arr = tc.singleton(indicies)
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'creating: {self.identity.shape}')
            arr = tc.empty((len(category_instances), self.identity.shape[0]))
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'created: {arr.dtype}')
            for i, idx in enumerate(indicies):
                arr[i] = self.identity[idx]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'encoding cat arr: {arr.dtype}')
        if is_one_row or True:
            return TensorFeatureContext(self.feature_id, arr)
        else:
            return SparseTensorFeatureContext.instance(
                self.feature_id, arr, self.torch_config)


@dataclass
class AggregateEncodableFeatureVectorizer(EncodableFeatureVectorizer):
    DESCRIPTION = 'aggregate vectorizer'

    delegate_feature_id: str
    size: int
    add_mask: bool = field(default=False)

    def _get_shape(self):
        return -1, self.delegate.shape[1]

    @property
    def delegate(self) -> EncodableFeatureVectorizer:
        return self.manager[self.delegate_feature_id]

    def _encode(self, datas: Iterable[Iterable[Any]]) -> FeatureContext:
        vec = self.delegate
        ctxs = tuple(map(lambda d: vec.encode(d), datas))
        return MultiFeatureContext(self.feature_id, ctxs)

    def _decode(self, context: MultiFeatureContext) -> torch.Tensor:
        vec = self.delegate
        srcs = tuple(map(lambda c: vec.decode(c), context.contexts))
        clen = len(srcs) * (2 if self.add_mask else 1)
        tc = self.torch_config
        first = srcs[0]
        dtype = first.dtype
        mid_dims = first.shape[1:]
        arr = tc.zeros((clen, self.size, *mid_dims), dtype=dtype)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'num contexts: {clen}, dtype={dtype}, ' +
                         f'src={first.shape}, dst={arr.shape}, ' +
                         f'mid_dims={mid_dims}')
        sz = self.size
        rowix = 0
        if self.add_mask:
            ones = tc.ones((self.size, *mid_dims), dtype=dtype)
        ctx: TensorFeatureContext
        for carr in srcs:
            lsz = min(carr.size(0), sz)
            if carr.dim() == 1:
                arr[rowix, :lsz] = carr[:lsz]
            elif carr.dim() == 2:
                arr[rowix, :lsz, :] = carr[:lsz, :]
            elif carr.dim() == 3:
                arr[rowix, :lsz, :, :] = carr[:lsz, :, :]
            if self.add_mask:
                arr[rowix + 1, :lsz] = ones[:lsz]
            rowix += (2 if self.add_mask else 1)
        return arr


@dataclass
class MaskTokenFeatureContext(FeatureContext):
    sequence_lengths: Tuple[int]


@dataclass
class MaskTokenContainerFeatureVectorizer(EncodableFeatureVectorizer):
    DESCRIPTION = 'mask'

    size: int
    data_type: Union[str, None, torch.dtype] = field(default=None)

    def __post_init__(self):
        super().__post_init__()
        self.data_type = str_to_dtype(self.data_type, self.torch_config)
        self.ones = self.torch_config.ones((self.size,), dtype=self.data_type)

    def _get_shape(self):
        return -1, self.size,

    def _encode(self, datas: Iterable[Iterable[Any]]) -> FeatureContext:
        lens = tuple(map(lambda d: sum(1 for _ in d), datas))
        return MaskTokenFeatureContext(self.feature_id, lens)

    def _decode(self, context: MaskTokenFeatureContext) -> torch.Tensor:
        tc = self.torch_config
        batch_size = len(context.sequence_lengths)
        ones = self.ones
        arr = tc.zeros((batch_size, self.size), dtype=self.data_type)
        for bix, slen in enumerate(context.sequence_lengths):
            arr[bix, :slen] = ones[:slen]
        return arr


@dataclass
class SeriesEncodableFeatureVectorizer(EncodableFeatureVectorizer):
    """Vectorize a Pandas series, such as a list of rows.  This vectorizer has an
    undefined shape since both the number of columns and rows are not specified
    at runtime.

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

    """
    DESCRIPTION = 'single attribute'

    def _get_shape(self):
        return 1,

    def _encode(self, data: Iterable[float]) -> FeatureContext:
        arr = self.torch_config.from_iterable(data)
        return TensorFeatureContext(self.feature_id, arr)
