import logging
from typing import Tuple, Dict, List, Iterable
from dataclasses import dataclass, field, InitVar
import sys
from io import TextIOWrapper
from itertools import chain
import numpy as np
import pandas as pd
import torch
from zensols.persist import persisted
from zensols.config import Writable
from zensols.deeplearn import (
    FeatureVectorizerManager,
    BatchFeatureMapping,
    ManagerFeatureMapping,
    FieldFeatureMapping,
    CategoryEncodableFeatureVectorizer,
    AttributeEncodableFeatureVectorizer,
    DataframeStash,
    FeatureVectorizer,
    BatchStash,
    DataPoint,
    Batch,
)

logger = logging.getLogger(__name__)


@dataclass
class DatasetMetadata(Writable):
    prefix: str
    label_name: str
    label_values: Tuple[str]
    continuous: Tuple[str]
    descrete: Dict[str, Tuple[str]]

    def write(self, depth: int = 0, writer: TextIOWrapper = sys.stdout):
        sp = self._sp(depth)
        sp2 = self._sp(depth + 1)
        sp3 = self._sp(depth + 2)
        writer.write(f'{sp}label: {self.label_name} => ' +
                     f'{", ".join(self.label_values)}\n')
        writer.write(f'{sp}continuous:\n')
        for c in self.continuous:
            writer.write(f'{sp2}{c}\n')
        writer.write(f'{sp}discrete:\n')
        for name, labels in self.descrete.items():
            writer.write(f'{sp2}{name}:\n')
            for label in labels:
                writer.write(f'{sp3}{label}\n')


@dataclass
class DataframeFeatureVectorizerManager(FeatureVectorizerManager, Writable):
    prefix: str
    label_col: str
    stash: DataframeStash
    include_columns: Tuple[str] = field(default=None)
    exclude_columns: Tuple[str] = field(default=None)

    @property
    @persisted('_dataset_metadata')
    def dataset_metadata(self) -> DatasetMetadata:
        logger.debug('constructing metadata')
        df = self.stash.dataframe
        skip = set([self.stash.split_col, self.label_col])
        labels = tuple(df[self.label_col].unique())
        cont = set()
        desc = {}
        for name, dtype in df.dtypes.iteritems():
            if name in skip:
                continue
            if dtype == np.object:
                desc[name] = tuple(df[name].unique())
            else:
                cont.add(name)
        return DatasetMetadata(self.prefix, self.label_col, labels, cont, desc)

    @property
    def label_feature_type(self) -> str:
        return f'{self.prefix}label'

    def column_to_feature_type(self, col: str) -> str:
        return f'{self.prefix}{col}'

    def _filter_columns(self, cols: Tuple[str]) -> Iterable[str]:
        def inc_vec(col: str):
            inc = incs is None or col in incs
            exc = excs is not None and col in excs
            return inc and not exc

        incs = self.include_columns
        excs = self.exclude_columns
        return filter(inc_vec, cols)

    def _create_label_vectorizer(self) -> FeatureVectorizer:
        label_col = self.label_feature_type#self.label_col
        label_values = self.dataset_metadata.label_values
        logger.debug(f'creating label {label_col} => {label_values}')
        return CategoryEncodableFeatureVectorizer(
            manager=self,
            feature_type=label_col,
            categories=label_values)

    def _create_feature_vectorizers(self) -> List[FeatureVectorizer]:
        vecs = []
        meta = self.dataset_metadata
        for col in meta.continuous:
            vec = AttributeEncodableFeatureVectorizer(
                manager=self,
                feature_type=self.column_to_feature_type(col))
            vecs.append(vec)
        for col in meta.descrete.keys():
            vec = CategoryEncodableFeatureVectorizer(
                manager=self,
                feature_type=self.column_to_feature_type(col),
                categories=meta.descrete[col])
            vecs.append(vec)
        return vecs

    def _create_vectorizers(self) -> Dict[str, FeatureVectorizer]:
        logger.debug('create vectorizers')
        vectorizers = super()._create_vectorizers()
        vecs = [self._create_label_vectorizer()]
        vecs.extend(self._create_feature_vectorizers())
        for vec in vecs:
            logger.debug(f'adding vectorizer: {vec.feature_type}')
            vectorizers[vec.feature_type] = vec
        return vectorizers

    @property
    @persisted('_batch_feature_mapping')
    def batch_feature_mapping(self) -> BatchFeatureMapping:
        def create_fileld_mapping(col: str) -> FieldFeatureMapping:
            feature_type = self.column_to_feature_type(col)
            return FieldFeatureMapping(col, feature_type, True)

        meta = self.dataset_metadata
        cols = (meta.continuous, meta.descrete.keys())
        fields = list(map(create_fileld_mapping,
                          chain.from_iterable(
                              map(self._filter_columns, cols))))
        fields.append(FieldFeatureMapping(
            self.label_col, self.label_feature_type, True))
        return BatchFeatureMapping(
            self.label_col,
            [ManagerFeatureMapping(self.name, fields)])

    def write(self, depth: int = 0, writer: TextIOWrapper = sys.stdout):
        sp = self._sp(depth)
        sp2 = self._sp(depth + 1)
        writer.write(f'{sp}{self.name}:\n')
        writer.write(f'{sp2}included: {self.include_columns}\n')
        writer.write(f'{sp2}excluded: {self.exclude_columns}\n')
        writer.write(f'{sp2}batch feature metadata:\n')
        self.batch_feature_mapping.write(depth + 2, writer)


@dataclass
class DataframeBatchStash(BatchStash):
    feature_vectorizer_manager: DataframeFeatureVectorizerManager


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
        label_attr = self._get_batch_feature_mappings().label_feature_type
        attr_names = filter(lambda k: k != label_attr, attrs.keys())
        feats = tuple(map(magic_shape, attr_names))
        return torch.cat(feats, dim=1)