from typing import Tuple, Dict
from dataclasses import dataclass
import sys
from io import TextIOWrapper
from zensols.config import Writable
from zensols.persist import persisted
from zensols.deeplearn.vectorize import (
    FeatureVectorizerManagerSet,
    FeatureVectorizerManager,
    FeatureVectorizer,
)
from . import (
    Batch,
    BatchStash,
    BatchFeatureMapping,
    ManagerFeatureMapping,
    FieldFeatureMapping,
)


@dataclass
class FieldVectorizerMetadata(Writable):
    field: FieldFeatureMapping
    vectorizer: FeatureVectorizer

    @property
    def shape(self):
        return self.vectorizer.shape

    def write(self, depth: int = 0, writer: TextIOWrapper = sys.stdout):
        self._write_line(self.field.attr)
        self._write_line('field:', depth + 1, writer)
        self.field.write(depth + 2, writer)
        self._write_line('vectorizer:', depth + 1, writer)
        self.vectorizer.write(depth + 2, writer)


@dataclass
class FieldVectorizerMetadataFactory(object):
    stash: BatchStash

    @persisted('_metadata')
    def __call__(self) -> Dict[str, FieldVectorizerMetadata]:
        stash = self.stash
        batch: Batch = stash.batch_type(None, None, None, None)
        mapping: BatchFeatureMapping = batch._get_batch_feature_mappings()
        vec_mng_set: FeatureVectorizerManagerSet = stash.vectorizer_manager_set
        attrib_keeps = stash.decoded_attributes
        vec_mngs: Tuple[ManagerFeatureMapping] = vec_mng_set.managers
        fields = {}
        mmng: ManagerFeatureMapping
        for mmng in mapping.manager_mappings:
            vec_mng_name: str = mmng.vectorizer_manager_name
            if vec_mng_name in vec_mngs:
                vec_mng: FeatureVectorizerManager = vec_mngs[vec_mng_name]
                field: FieldFeatureMapping
                for field in mmng.fields:
                    if field.attr in attrib_keeps:
                        vec = vec_mng[field.feature_type]
                        fields[field.attr] = FieldVectorizerMetadata(field, vec)
        return fields
