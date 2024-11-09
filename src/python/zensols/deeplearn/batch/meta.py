"""Contains container classes for batch data.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import Dict, Type, TYPE_CHECKING
if TYPE_CHECKING:
    from .stash import BatchStash
from dataclasses import dataclass
from dataclasses import field as dc_field
import sys
from io import TextIOBase
from zensols.config import Dictable
from zensols.deeplearn import NetworkSettings
from zensols.deeplearn.vectorize import FeatureVectorizer
from . import DataPoint, Batch, BatchFeatureMapping, FieldFeatureMapping


@dataclass
class BatchFieldMetadata(Dictable):
    """Data that describes a field mapping in a batch object.

    """
    field: FieldFeatureMapping = dc_field()
    """The field mapping."""

    vectorizer: FeatureVectorizer = dc_field(repr=False)
    """The vectorizer used to map the field."""

    @property
    def shape(self):
        return self.vectorizer.shape

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(self.field.attr, depth, writer)
        self._write_line('field:', depth + 1, writer)
        self.field.write(depth + 2, writer)
        self._write_line('vectorizer:', depth + 1, writer)
        self.vectorizer.write(depth + 2, writer)


@dataclass
class BatchMetadata(Dictable):
    """Describes metadata about a :class:`.Batch` instance.

    """
    data_point_class: Type[DataPoint] = dc_field()
    """The :class:`.DataPoint` class, which are created at encoding time."""

    batch_class: Type[Batch] = dc_field()
    """The :class:`.Batch` class, which are created at encoding time."""

    mapping: BatchFeatureMapping = dc_field()
    """The mapping used for encoding and decoding the batch."""

    fields_by_attribute: Dict[str, BatchFieldMetadata] = dc_field(repr=False)
    """Mapping by field name to attribute."""

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(f'data point: {self.data_point_class}', depth, writer)
        self._write_line(f'batch: {self.batch_class}', depth, writer)
        self._write_line('mapping:', depth, writer)
        self.mapping.write(depth + 1, writer)
        self._write_line('attributes:', depth, writer)
        for attr, field in self.fields_by_attribute.items():
            field.write(depth + 1, writer)


@dataclass
class MetadataNetworkSettings(NetworkSettings):
    """A network settings container that has metadata about batches it recieves
    for its model.

    """
    _PERSITABLE_TRANSIENT_ATTRIBUTES = {'batch_stash'}

    batch_stash: BatchStash = dc_field(repr=False)
    """The batch stash that created the batches and has the batch metdata.

    """
    @property
    def batch_metadata(self) -> BatchMetadata:
        """Return the batch metadata used by this model.

        """
        return self.batch_stash.batch_metadata

    def _from_dictable(self, *args, **kwargs):
        dct = super()._from_dictable(*args, **kwargs)
        del dct['batch_stash']
        return dct
