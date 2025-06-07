"""An implementation of batch level API for Pandas dataframe based data.

"""
__author__ = 'Paul Landes'

import logging
from typing import Tuple
from dataclasses import dataclass, InitVar
import pandas as pd
import torch
from zensols.deeplearn.batch import (
    BatchError,
    BatchFeatureMapping,
    BatchStash,
    DataPoint,
    Batch,
)
from zensols.deeplearn.dataframe import DataframeFeatureVectorizerManager

logger = logging.getLogger(__name__)


@dataclass
class DataframeBatchStash(BatchStash):
    """A stash used for batches of data using :class:`.DataframeBatch`
    instances.  This stash uses an instance of
    :class:`.DataframeFeatureVectorizerManager` to vectorize the data in the
    batches.

    """
    @property
    def feature_vectorizer_manager(self) -> DataframeFeatureVectorizerManager:
        managers = tuple(self.vectorizer_manager_set.values())
        if len(managers) != 1:
            raise BatchError('Exected only one vector manager but got: ' +
                             tuple(self.vectorizer_manager_set.keys()))
        vec_mng = managers[0]
        if not isinstance(vec_mng, DataframeFeatureVectorizerManager):
            raise BatchError(
                'Expected class of type DataframeFeatureVectorizerManager ' +
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
    """A data point used in a batch, which contains a single row of data in the
    Pandas dataframe.  When created, column is saved as an attribute in the
    instance.

    """
    row: InitVar[pd.Series]

    def __post_init__(self, row: pd.Series):
        for name, val in row.items():
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'setting attrib: {name}={val}')
            setattr(self, name, val)


@dataclass
class DataframeBatch(Batch):
    """A batch of data that contains instances of :class:`.DataframeDataPoint`,
    each of which has the row data from the dataframe.

    """
    def _get_batch_feature_mappings(self) -> BatchFeatureMapping:
        """Use the dataframe based vectorizer manager.

        """
        df_vec_mng: DataframeFeatureVectorizerManager = \
            self.batch_stash.feature_vectorizer_manager
        return df_vec_mng.batch_feature_mapping

    def get_features(self) -> torch.Tensor:
        """A utility method to a tensor of all features of all columns in the
        datapoints.

        :return: a tensor of shape (batch size, feature size), where the
                 *feaure size* is the number of all features vectorized; that
                 is, a data instance for each row in the batch, is a flattened
                 set of features that represent the respective row from the
                 dataframe

        """
        def magic_shape(name: str) -> torch.Tensor:
            """Return a tensor that has two dimenions of the data (the first
            always with size 1 since it is a row of data).

            """
            arr = attrs[name]
            if len(arr.shape) == 1:
                arr = arr.unsqueeze(dim=1)
            return arr

        attrs = self.attributes
        label_attr = self._get_batch_feature_mappings().label_attribute_name
        attr_names = filter(lambda k: k != label_attr, attrs.keys())
        feats = tuple(map(magic_shape, attr_names))
        return torch.cat(feats, dim=1)
