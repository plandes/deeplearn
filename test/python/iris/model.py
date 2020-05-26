import logging
from dataclasses import dataclass, field
from typing import Any, List
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from zensols.persist import persisted
from zensols.deeplearn import NetworkSettings
#from zensols.deeplearn.layer import DeepLinearLayer
from zensols.deeplearn.model import BaseNetworkModule
from zensols.deeplearn.batch import (
    DataPoint,
    Batch,
    BatchFeatureMapping,
    ManagerFeatureMapping,
    FieldFeatureMapping,
)

logger = logging.getLogger(__name__)


@dataclass
class IrisDataPoint(DataPoint):
    LABEL_COL = 'species'
    FLOWER_DIMS = 'sepal_length sepal_width petal_length petal_width'.split()

    row: pd.Series

    @property
    def label(self) -> str:
        return self.row[self.LABEL_COL]

    @property
    def flower_dims(self) -> pd.Series:
        return [self.row[self.FLOWER_DIMS]]


@dataclass
class IrisBatch(Batch):
    MAPPINGS = BatchFeatureMapping(
        'label',
        [ManagerFeatureMapping(
            'iris_vectorizer_manager',
            (FieldFeatureMapping('label', 'ilabel', True),
             FieldFeatureMapping('flower_dims', 'iseries')))])

    def _get_batch_feature_mappings(self) -> BatchFeatureMapping:
        return self.MAPPINGS

    def get_flower_dimensions(self) -> torch.Tensor:
        return self.attributes['flower_dims']


@dataclass
class IrisNetworkSettings(NetworkSettings):
    """A utility container settings class for convulsion network models.

    """
    in_features: int
    out_features: int

    def get_module_class_name(self) -> str:
        return __name__ + '.IrisNetwork'


class IrisNetwork(BaseNetworkModule):
    """A recurrent neural network model that is used to classify sentiment.

    """
    def __init__(self, net_settings: IrisNetworkSettings):
        super().__init__(net_settings)
        ns = net_settings
        self.fc = nn.Linear(ns.in_features, ns.out_features)
        self.dropout = None if ns.dropout is None else nn.Dropout(ns.dropout)

    def _forward(self, batch):
        logger.debug(f'batch: {batch}')

        x = batch.get_flower_dimensions()
        self._shape_debug('input', x)

        x = self.fc(x)
        self._shape_debug('linear', x)

        if self.dropout is not None:
            x = self.dropout.fc(x)

        if self.net_settings.activation_function is not None:
            x = self.net_settings.activation_function(x)
            x = F.relu(x)

        return x
