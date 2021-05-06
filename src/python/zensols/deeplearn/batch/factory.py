"""Utility class to create batches from configuration.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass
import logging
from . import Batch, BatchFeatureMapping

logger = logging.getLogger(__name__)


@dataclass
class ConfigBatch(Batch):
    """A batch class that gets its feature mappings from a configuration file.

    """
    def _get_batch_feature_mappings(self) -> BatchFeatureMapping:
        return self.MAPPINGS


@dataclass
class BatchFactory(object):
    pass
