"""Contains container classes for batch data.

"""
__author__ = 'Paul Landes'

from typing import Dict, Type
from dataclasses import dataclass
from dataclasses import field as dc_field
import sys
from io import TextIOBase
from zensols.config import Dictable
from zensols.persist import persisted, PersistableContainer
from zensols.deeplearn import NetworkSettings
from zensols.deeplearn.vectorize import (
    FeatureVectorizerManagerSet,
    FeatureVectorizerManager,
    FeatureVectorizer,
)
from . import (
    DataPoint,
    Batch,
    BatchStash,
    BatchFeatureMapping,
    ManagerFeatureMapping,
    FieldFeatureMapping,
)


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
class BatchMetadataFactory(PersistableContainer):
    """Creates instances of :class:`.BatchMetadata`.

    """
    stash: BatchStash = dc_field()
    """The stash used to create the batches."""

    @persisted('_metadata')
    def __call__(self) -> BatchMetadata:
        stash: BatchStash = self.stash
        batch: Batch = stash.batch_type(None, None, None, None)
        batch.batch_stash = stash
        mapping: BatchFeatureMapping = batch._get_batch_feature_mappings()
        batch.deallocate()
        vec_mng_set: FeatureVectorizerManagerSet = stash.vectorizer_manager_set
        attrib_keeps = stash.decoded_attributes
        vec_mng_names = set(vec_mng_set.keys())
        by_attrib = {}
        mmng: ManagerFeatureMapping
        for mmng in mapping.manager_mappings:
            vec_mng_name: str = mmng.vectorizer_manager_name
            if vec_mng_name in vec_mng_names:
                vec_mng: FeatureVectorizerManager = vec_mng_set[vec_mng_name]
                field: FieldFeatureMapping
                for field in mmng.fields:
                    if field.attr in attrib_keeps:
                        vec = vec_mng[field.feature_id]
                        by_attrib[field.attr] = BatchFieldMetadata(field, vec)
        return BatchMetadata(stash.data_point_type, stash.batch_type,
                             mapping, by_attrib)


@dataclass
class MetadataNetworkSettings(NetworkSettings):
    """A network settings container that has metadata about batches it recieves for
    its model.

    """
    batch_metadata_factory: BatchMetadataFactory = dc_field()
    """The factory that produces the metadata that describe the batch data
    during the calls to :py:meth:`_forward`.

    """

    @property
    def batch_metadata(self) -> BatchMetadata:
        """Return the batch metadata used by this model.

        """
        # it's not necessary to persist here since the call in the factory
        # already does; also cleaned up by the metadata factory as it extends
        # from PersistableContainer
        return self.batch_metadata_factory()
