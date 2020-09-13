from typing import Tuple, Dict, Type
from dataclasses import dataclass
import sys
from io import TextIOBase
from zensols.config import Writable
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
class BatchFieldMetadata(Writable):
    """Data that describes a field mapping in a batch object.

    :param field: the field mapping

    :param vectorizer: the vectorizer used to map the field

    """
    field: FieldFeatureMapping
    vectorizer: FeatureVectorizer

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
class BatchMetadata(Writable):
    """Describes metadata about a :class:`.Batch` instance.

    :param data_point_class: the :class:`.DataPoint` class, which are created
                             at encoding time

    :param batch_class: the :class:`.Batch` class, which are created at
                        encoding time

    :param mapping: the mapping used for encoding and decoding the batch

    :param fields_by_attribute: a dict of name to a batch field mapping

    """
    data_point_class: Type[DataPoint]
    batch_class: Type[Batch]
    mapping: BatchFeatureMapping
    fields_by_attribute: Dict[str, BatchFieldMetadata]

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

    :param stash: the stash used to create the batches

    """
    stash: BatchStash

    @persisted('_metadata')
    def __call__(self) -> BatchMetadata:
        stash = self.stash
        batch: Batch = stash.batch_type(None, None, None, None)
        mapping: BatchFeatureMapping = batch._get_batch_feature_mappings()
        batch.deallocate()
        vec_mng_set: FeatureVectorizerManagerSet = stash.vectorizer_manager_set
        attrib_keeps = stash.decoded_attributes
        vec_mngs: Tuple[ManagerFeatureMapping] = vec_mng_set.managers
        by_attrib = {}
        mmng: ManagerFeatureMapping
        for mmng in mapping.manager_mappings:
            vec_mng_name: str = mmng.vectorizer_manager_name
            if vec_mng_name in vec_mngs:
                vec_mng: FeatureVectorizerManager = vec_mngs[vec_mng_name]
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

    :param batch_metadata_factory:

        the factory that produces the metadata that describe the batch data
        during the calls to :py:meth:`_forward`

    """
    batch_metadata_factory: BatchMetadataFactory

    @property
    def batch_metadata(self) -> BatchMetadata:
        """Return the batch metadata used by this model.

        """
        # it's not necessary to persist here since the call in the factory
        # already does; also cleaned up by the metadata factory as it extends
        # from PersistableContainer
        return self.batch_metadata_factory()
