from dataclasses import dataclass
import logging
import pandas as pd
import torch
from zensols.deeplearn import (
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
