import logging
from typing import Tuple
from dataclasses import dataclass, InitVar
import pandas as pd
import torch
from zensols.deeplearn.batch import (
    BatchFeatureMapping,
    BatchStash,
    DataPoint,
    Batch,
)
from zensols.deeplearn.dataframe import DataframeFeatureVectorizerManager

logger = logging.getLogger(__name__)


@dataclass
class DataframeBatchStash(BatchStash):
    @property
    def feature_vectorizer_manager(self):
        managers = tuple(self.vectorizer_manager_set.values())
        if len(managers) != 1:
            raise ValueError('exected only one vector manager but got: ' +
                             tuple(self.vectorizer_manager_set.keys()))
        vec_mng = managers[0]
        if not isinstance(vec_mng, DataframeFeatureVectorizerManager):
            raise ValueError(
                'expected class of type DataframeFeatureVectorizerManager ' +
                f'but got {vec_mng.__class__}')
        return vec_mng

    @property
    def label_shape(self) -> Tuple[int]:
        return self.feature_vectorizer_manager.label_shape

    @property
    def flattened_features_shape(self) -> Tuple[int]:
        vec_mng = self.feature_vectorizer_manager
        return vec_mng.get_flattened_features_shape(self.decoded_attributes)


@dataclass
class DataframeDataPoint(DataPoint):
    row: InitVar[pd.Series]

    def __post_init__(self, row: pd.Series):
        for name, val in row.iteritems():
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'setting attrib: {name}={val}')
            setattr(self, name, val)


@dataclass
class DataframeBatch(Batch):
    def _get_batch_feature_mappings(self) -> BatchFeatureMapping:
        df_vec_mng = self.batch_stash.feature_vectorizer_manager
        return df_vec_mng.batch_feature_mapping

    def get_features(self) -> torch.Tensor:
        def magic_shape(name: str) -> torch.Tensor:
            arr = attrs[name]
            if len(arr.shape) == 1:
                arr = arr.unsqueeze(dim=1)
            return arr

        attrs = self.attributes
        label_attr = self._get_batch_feature_mappings().label_attribute_name
        attr_names = filter(lambda k: k != label_attr, attrs.keys())
        feats = tuple(map(magic_shape, attr_names))
        return torch.cat(feats, dim=1)
