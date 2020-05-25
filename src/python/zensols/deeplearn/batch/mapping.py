"""Mapping metadata for batch domain specific instances.

"""
__author__ = 'Paul Landes'

from typing import Tuple, List, Union
from dataclasses import dataclass, field
import sys
import logging
from io import TextIOWrapper
from zensols.config import Writable

logger = logging.getLogger(__name__)


@dataclass
class FieldFeatureMapping(Writable):
    """Meta data describing an attribute of the data point.

    :params attr: the attribute name, which is used to identify the
                  feature that is vectorized

    :param feature_type: indicates which vectorizer to use

    :param is_agg: if ``True``, tuplize across all data points and encode as
                   one tuple of data to create the batched tensor on decode;
                   otherwise, each data point feature is encoded and
                   concatenated on decode

    """
    attr: str
    feature_type: str
    is_agg: bool = field(default=False)

    def write(self, depth: int = 0, writer: TextIOWrapper = sys.stdout):
        self._write_line(str(self), depth, writer)


@dataclass
class ManagerFeatureMapping(Writable):
    """Meta data for a vectorizer manager with fields describing attributes to be
    vectorized from features in to feature contests.

    :param vectorizer_manager_name: the configuration name that identifiees
                                    an instance of ``FeatureVectorizerManager``
    :param field: the fields of the data point to be vectorized
    """
    vectorizer_manager_name: str
    fields: Tuple[FieldFeatureMapping]

    def write(self, depth: int = 0, writer: TextIOWrapper = sys.stdout):
        self._write_line(self.vectorizer_manager_name, depth, writer)
        for f in self.fields:
            f.write(depth + 1, writer)


@dataclass
class BatchFeatureMapping(Writable):
    """The meta data used to encode and decode each feature in to tensors.  It is
    best to define a class level instance of this in the ``Batch`` class and
    return it with ``_get_batch_feature_mappings``.

    An example from the iris data set test:

        MAPPINGS = BatchFeatureMapping(
            'label',
            [ManagerFeatureMapping(
                'iris_vectorizer_manager',
                (FieldFeatureMapping('label', 'ilabel', True),
                 FieldFeatureMapping('flower_dims', 'iseries')))])

    :param label_attribute_name: the name of the attribute used for labels
    :param manager_mappings: the manager level attribute mapping meta data
    """
    label_attribute_name: str
    manager_mappings: List[ManagerFeatureMapping]

    @property
    def label_feature_type(self) -> Union[None, str]:
        """Return the feature type of the label.  This is the vectorizer used to
        transform the label data.

        """
        mng, f = self.get_field_map_by_attribute(self.label_attribute_name)
        if f is not None:
            return f.feature_type

    def get_field_map_by_feature_type(self, feature_type: str) -> \
            Union[None, Tuple[ManagerFeatureMapping, FieldFeatureMapping]]:
        for mng in self.manager_mappings:
            for f in mng.fields:
                if feature_type == f.feature_type:
                    return mng, f

    def get_field_map_by_attribute(self, attribute_name: str) -> \
            Union[None, Tuple[ManagerFeatureMapping, FieldFeatureMapping]]:
        for mng in self.manager_mappings:
            for f in mng.fields:
                if attribute_name == f.attr:
                    return mng, f

    def write(self, depth: int = 0, writer: TextIOWrapper = sys.stdout):
        self._write_line(f'label: {self.label_attribute_name}', depth, writer)
        for m in self.manager_mappings:
            m.write(depth + 1, writer)
