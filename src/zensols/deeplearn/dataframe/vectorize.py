"""Contains classes used to vectorize dataframe data.

"""
__author__ = 'Paul Landes'

import logging
from typing import Tuple, Dict, List, Iterable, Set
from dataclasses import dataclass, field
import sys
from io import TextIOBase
from itertools import chain
from functools import reduce
import operator
import numpy as np
from zensols.persist import persisted
from zensols.config import Writable
from zensols.deeplearn.vectorize import (
    VectorizerError,
    OneHotEncodedEncodableFeatureVectorizer,
    AttributeEncodableFeatureVectorizer,
    FeatureVectorizer,
    FeatureVectorizerManager,
)
from zensols.deeplearn.batch import (
    BatchFeatureMapping,
    ManagerFeatureMapping,
    FieldFeatureMapping,
)
from zensols.dataframe import DataframeStash

logger = logging.getLogger(__name__)


@dataclass
class DataframeMetadata(Writable):
    """Metadata for a Pandas dataframe.

    """
    prefix: str = field()
    """The prefix to use for all vectorizers in the dataframe (i.e. ``adl_``
    for the Adult dataset test case example).

    """

    label_col: str = field()
    """The column that contains the label/class."""

    label_values: Tuple[str] = field()
    """All classes (unique across ``label_col``)."""

    continuous: Tuple[str] = field()
    """The list of data columns that are continuous."""

    descrete: Dict[str, Tuple[str]] = field()
    """A mapping of label to nominals the column takes for descrete mappings.

    """

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        sp = self._sp(depth)
        sp2 = self._sp(depth + 1)
        sp3 = self._sp(depth + 2)
        writer.write(f'{sp}label: {self.label_col} => ' +
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
    """A pure instance based feature vectorizer manager for a Pandas dataframe.
    All vectorizers used in this vectorizer manager are dynamically allocated
    and attached.

    This class not only acts as the feature manager itself to be used in a
    :class:`~zensols.deeplearn.vectorize.FeatureVectorizerManager`, but also
    provides a batch mapping to be used in a
    :class:`~zensols.deeplearn.batch.BatchStash`.

    """
    prefix: str = field()
    """The prefix to use for all vectorizers in the dataframe (i.e. ``adl_``
    for the Adult dataset test case example).

    """

    label_col: str = field()
    """The column that contains the label/class."""

    stash: DataframeStash = field()
    """The stash that contains the dataframe."""

    include_columns: Tuple[str] = field(default=None)
    """The columns to be included, or if ``None`` (the default), all columns
    are used as features.

    """

    exclude_columns: Tuple[str] = field(default=None)
    """The columns to be excluded, or if ``None`` (the default), no columns are
    excluded as features.

    """

    @property
    @persisted('_dataset_metadata')
    def dataset_metadata(self) -> DataframeMetadata:
        """Create a metadata from the data in the dataframe.

        """
        logger.debug('constructing metadata')
        df = self.stash.dataframe
        skip = set([self.stash.split_col, self.label_col])
        labels = tuple(df[self.label_col].unique())
        cont = set()
        desc = {}
        for name, dtype in df.dtypes.items():
            if name in skip:
                continue
            if dtype == object:
                desc[name] = tuple(df[name].unique())
            else:
                cont.add(name)
        return DataframeMetadata(
            self.prefix, self.label_col, labels, cont, desc)

    @property
    def label_attribute_name(self) -> str:
        """Return the label attribute.

        """
        return f'{self.prefix}label'

    def column_to_feature_id(self, col: str) -> str:
        """Generate a feature id from the column name.  This just attaches the
        prefix to the column name.

        """
        return f'{self.prefix}{col}'

    def _filter_columns(self, cols: Tuple[str]) -> Iterable[str]:
        """Return an interable of the columns to use as features based on
        ``include_columns`` and ``exclude_columns``.

        """
        def inc_vec(col: str):
            inc = incs is None or col in incs
            exc = excs is not None and col in excs
            return inc and not exc

        incs = self.include_columns
        excs = self.exclude_columns
        return filter(inc_vec, cols)

    def _create_label_vectorizer(self) -> FeatureVectorizer:
        """Create a vectorizer for the label/class of the dataframe.

        """
        label_col = self.label_attribute_name
        label_values = self.dataset_metadata.label_values
        logger.debug(f'creating label {label_col} => {label_values}')
        return OneHotEncodedEncodableFeatureVectorizer(
            name=str(self.__class__),
            config_factory=self.config_factory,
            manager=self,
            feature_id=label_col,
            categories=label_values,
            optimize_bools=False)

    def _create_feature_vectorizers(self) -> List[FeatureVectorizer]:
        """Create a vectorizer, one for each column/feature, included as a
        feature type based on :meth:`_filter_columns`.

        """
        vecs = []
        meta = self.dataset_metadata
        for col in meta.continuous:
            vec = AttributeEncodableFeatureVectorizer(
                manager=self,
                name=str(self.__class__),
                config_factory=self.config_factory,
                feature_id=self.column_to_feature_id(col))
            vecs.append(vec)
        for col in meta.descrete.keys():
            vec = OneHotEncodedEncodableFeatureVectorizer(
                manager=self,
                name=str(self.__class__),
                config_factory=self.config_factory,
                feature_id=self.column_to_feature_id(col),
                categories=meta.descrete[col],
                optimize_bools=True)
            vecs.append(vec)
        return vecs

    def _create_vectorizers(self) -> Dict[str, FeatureVectorizer]:
        """Create a mapping of feature id to vectorizer used across all dataframe
        columsn.

        """
        logger.debug('create vectorizers')
        vectorizers = super()._create_vectorizers()
        vecs = [self._create_label_vectorizer()]
        vecs.extend(self._create_feature_vectorizers())
        for vec in vecs:
            logger.debug(f'adding vectorizer: {vec.feature_id}')
            vectorizers[vec.feature_id] = vec
        return vectorizers

    @property
    @persisted('_batch_feature_mapping')
    def batch_feature_mapping(self) -> BatchFeatureMapping:
        """Return the mapping for :class:`zensols.deeplearn.batch.Batch`
        instances.

        """
        def create_fileld_mapping(col: str) -> FieldFeatureMapping:
            feature_id = self.column_to_feature_id(col)
            return FieldFeatureMapping(col, feature_id, True)

        meta = self.dataset_metadata
        cols = (meta.continuous, meta.descrete.keys())
        fields = list(map(create_fileld_mapping,
                          chain.from_iterable(
                              map(self._filter_columns, cols))))
        fields.append(FieldFeatureMapping(
            self.label_col, self.label_attribute_name, True))
        return BatchFeatureMapping(
            self.label_col,
            [ManagerFeatureMapping(self.name, fields)])

    @property
    def label_shape(self) -> Tuple[int]:
        """Return the shape if all vectorizers were used.

        """
        label_attr = self.batch_feature_mapping.label_feature_id
        for k, v in self.items():
            if k == label_attr:
                return (sum(filter(lambda n: n > 0, v.shape)),)

    def get_flattened_features_shape(self, attribs: Set[str]) -> Tuple[int]:
        """Return the shape if all vectorizers were used.

        """
        bmapping = self.batch_feature_mapping
        label_feature_id = bmapping.label_feature_id
        n_flat_neurons = 0
        for feature_id, v in self.items():
            _, field_map = bmapping.get_field_map_by_feature_id(feature_id)
            if field_map is None:
                s = f'no feature: {feature_id} in vectorizer {self.name}'
                raise VectorizerError(s)
            attr = field_map.attr
            if feature_id != label_feature_id and \
               (attribs is None or attr in attribs):
                n = reduce(operator.mul, filter(lambda n: n > 0, v.shape))
                n_flat_neurons += n
        return (n_flat_neurons,)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        sp = self._sp(depth)
        sp2 = self._sp(depth + 1)
        writer.write(f'{sp}{self.name}:\n')
        writer.write(f'{sp2}included: {self.include_columns}\n')
        writer.write(f'{sp2}excluded: {self.exclude_columns}\n')
        writer.write(f'{sp2}batch feature metadata:\n')
        self.batch_feature_mapping.write(depth + 2, writer)
