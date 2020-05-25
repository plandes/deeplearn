"""Vectorizer implementations.

"""
__author__ = 'Paul Landes'


import logging
from dataclasses import dataclass, field
from typing import Set, List, Iterable
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from zensols.persist import persisted
from zensols.vectorize import (
    EncodableFeatureVectorizer,
    TensorFeatureContext,
    SparseTensorFeatureContext,
    FeatureContext,
)

logger = logging.getLogger(__name__)


@dataclass
class CategoryEncodableFeatureVectorizer(EncodableFeatureVectorizer):
    """Vectorize from a list of nominals.  This is useful for encoding labels for
    the categorization machine learning task.

    """
    NAME = 'category label encoder'

    categories: Set[str]
    feature_type: str
    optimize_bools: bool = field(default=True)

    def __post_init__(self):
        super().__post_init__()
        le = LabelEncoder()
        le.fit(self.categories)
        llen = len(le.classes_)
        if not self.optimize_bools or llen != 2:
            arr = self.manager.torch_config.zeros((llen, llen))
            for i in range(llen):
                arr[i][i] = 1
            self.identity = arr
        self.label_encoder = le

    @persisted('_get_shape_pw')
    def _get_shape(self):
        n_classes = len(self.label_encoder.classes_)
        if self.optimize_bools and n_classes == 2:
            return 1,
        else:
            return -1, n_classes

    def _encode(self, category_instances: List[str]) -> FeatureContext:
        tc = self.manager.torch_config
        indicies = self.label_encoder.transform(category_instances)
        is_one_row = self.shape[0] == 1
        if is_one_row:
            arr = self.manager.torch_config.singleton(indicies)
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
            return TensorFeatureContext(self.feature_type, arr)
        else:
            return SparseTensorFeatureContext.instance(
                self.feature_type, arr, self.manager.torch_config)

    def _decode(self, context: FeatureContext) -> torch.Tensor:
        if isinstance(context, SparseTensorFeatureContext):
            return context.to_tensor(self.manager.torch_config)
        else:
            return super()._decode(context)


@dataclass
class SeriesEncodableFeatureVectorizer(EncodableFeatureVectorizer):
    """Vectorize a Pandas series, such as a list of rows.  This vectorizer has an
    undefined shape since both the number of columns and rows are not specified at
    runtime.

    """
    NAME = 'pandas series'

    feature_type: str

    def _get_shape(self):
        return -1, -1

    def _encode(self, rows: Iterable[pd.Series]) -> FeatureContext:
        narrs = []
        tc = self.manager.torch_config
        nptype = tc.numpy_data_type
        for row in rows:
            narrs.append(row.to_numpy(dtype=nptype))
        arr = np.stack(narrs)
        arr = tc.from_numpy(arr)
        return TensorFeatureContext(self.feature_type, arr)


@dataclass
class AttributeEncodableFeatureVectorizer(EncodableFeatureVectorizer):
    """Vectorize a Pandas series, such as a list of rows.  This vectorizer has an
    undefined shape since both the number of columns and rows are not specified
    at runtime.

    """
    NAME = 'single attribute'

    feature_type: str

    def _get_shape(self):
        return 1,

    def _encode(self, rows: Iterable[float]) -> FeatureContext:
        arr = self.manager.torch_config.from_iterable(rows)
        return TensorFeatureContext(self.feature_type, arr)
