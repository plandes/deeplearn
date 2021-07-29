"""Mapping metadata for batch domain specific instances.

"""
__author__ = 'Paul Landes'

from typing import Tuple, List, Union, Iterable
from dataclasses import dataclass, field
import sys
import logging
from itertools import chain
from io import TextIOBase
from zensols.config import Dictable
from zensols.deeplearn.vectorize import FeatureVectorizerManager

logger = logging.getLogger(__name__)


@dataclass
class FieldFeatureMapping(Dictable):
    """Meta data describing an attribute of the data point.

    """
    attr: str = field()
    """The (human readable/used) name for the mapping."""

    feature_id: str = field()
    """Indicates which vectorizer to use."""

    is_agg: bool = field(default=False)
    """If ``True``, tuplize across all data points and encode as one tuple of
    data to create the batched tensor on decode; otherwise, each data point
    feature is encoded and concatenated on decode.

    """

    attr_access: str = field(default=None)
    """The attribute on the source :class:`DataPoint` instance (see
    :obj:`~attribute_accessor`).

    """

    is_label: bool = field(default=False)
    """Whether or not this field is a label.  The is ``True`` in cases where there
    is more than one label.  In these cases, usually which label to use changes
    based on the model (i.e. word embedding vs. BERT word piece token IDs).

    This is used in :class:`.Batch` to skip label vectorization while encoding
    of prediction based batches.

    """

    @property
    def attribute_accessor(self):
        """Return the attribute name on the :class:`DataPoint` instance.  This uses
        :obj:`~attr_access` if it is not ``None``, otherwise, use
        :obj:`~attr`.

        """
        return self.attr if self.attr_access is None else self.attr_access

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(str(self), depth, writer)


@dataclass
class ManagerFeatureMapping(Dictable):
    """Meta data for a vectorizer manager with fields describing attributes to be
    vectorized from features in to feature contests.

    """
    vectorizer_manager_name: str = field()
    """The configuration name that identifiees an instance of
    ``FeatureVectorizerManager``.

    """

    fields: Tuple[FieldFeatureMapping] = field()
    """The fields of the data point to be vectorized."""

    def remove_field(self, attr: str) -> bool:
        """Remove a field by attribute if it exists.

        :param attr: the name of the field's attribute to remove

        :return: ``True`` if the field was removed, ``False`` otherwise

        """
        plen = len(self.fields)
        self.fields = tuple(filter(lambda f: f.attr != attr, self.fields))
        return plen != len(self.fields)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(self.vectorizer_manager_name, depth, writer)
        for f in self.fields:
            f.write(depth + 1, writer)


@dataclass
class BatchFeatureMapping(Dictable):
    """The meta data used to encode and decode each feature in to tensors.  It is
    best to define a class level instance of this in the ``Batch`` class and
    return it with ``_get_batch_feature_mappings``.

    An example from the iris data set test::

        MAPPINGS = BatchFeatureMapping(
            'label',
            [ManagerFeatureMapping(
                'iris_vectorizer_manager',
                (FieldFeatureMapping('label', 'ilabel', True),
                 FieldFeatureMapping('flower_dims', 'iseries')))])

    """
    label_attribute_name: str = field()
    """The name of the attribute used for labels."""

    manager_mappings: List[ManagerFeatureMapping] = field()
    """The manager level attribute mapping meta data."""

    def __post_init__(self):
        attrs = tuple(map(lambda f: f.attr, self.get_attributes()))
        attr_set = set(attrs)
        if len(attrs) != len(attr_set):
            raise ValueError(f'attribute names must be unique: {attrs}')

    def get_attributes(self) -> Iterable[FieldFeatureMapping]:
        return chain.from_iterable(
            map(lambda m: m.fields, self.manager_mappings))

    @property
    def label_feature_id(self) -> Union[None, str]:
        """Return the feature id of the label.  This is the vectorizer used to
        transform the label data.

        """
        mng, f = self.get_field_map_by_attribute(self.label_attribute_name)
        if f is not None:
            return f.feature_id

    @property
    def label_vectorizer_manager(self) -> \
            Union[FeatureVectorizerManager, None]:
        """Return the feature id of the label.  This is the vectorizer used to
        transform the label data.

        """
        mng, f = self.get_field_map_by_attribute(self.label_attribute_name)
        if mng is not None:
            return mng

    def get_field_map_by_feature_id(self, feature_id: str) -> \
            Union[None, Tuple[ManagerFeatureMapping, FieldFeatureMapping]]:
        for mng in self.manager_mappings:
            for f in mng.fields:
                if feature_id == f.feature_id:
                    return mng, f

    def get_field_map_by_attribute(self, attribute_name: str) -> \
            Union[None, Tuple[ManagerFeatureMapping, FieldFeatureMapping]]:
        for mng in self.manager_mappings:
            for f in mng.fields:
                if attribute_name == f.attr:
                    return mng, f

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(f'label: {self.label_attribute_name}', depth, writer)
        for m in self.manager_mappings:
            m.write(depth + 1, writer)
