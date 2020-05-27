from typing import Tuple
from dataclasses import dataclass
import logging
import torch
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
class MnistDataPoint(DataPoint):
    data_label: Tuple[torch.Tensor, torch.Tensor]

    @property
    def data(self):
        return self.data_label[0]

    @property
    def label(self):
        return self.data_label[1]


@dataclass
class MnistBatch(Batch):
    MAPPINGS = BatchFeatureMapping(
        'label',
        [ManagerFeatureMapping(
            'mnist_vectorizer_manager',
            (FieldFeatureMapping('label', 'identity', True),
             FieldFeatureMapping('data', 'identity')))])

    def _get_batch_feature_mappings(self) -> BatchFeatureMapping:
        return self.MAPPINGS

    def get_data(self) -> torch.Tensor:
        return self.attributes['data']
