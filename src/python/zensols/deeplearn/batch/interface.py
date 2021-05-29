"""Interface and simple domain classes.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Set, Dict, Any, Iterable, List
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod
from zensols.persist import DirectoryCompositeStash
from zensols.deeplearn import DeepLearnError


class BatchError(DeepLearnError):
    """Thrown for any batch related error."""
    pass


class BatchDirectoryCompositeStash(DirectoryCompositeStash):
    """A composite stash used for instances of :class:`BatchStash`.

    """
    def __init__(self, path: Path, groups: Tuple[Set[str]]):
        super().__init__(path, groups, '_feature_contexts')


@dataclass
class DataPointIDSet(object):
    """Set of subordinate stash IDs with feature values to be vectorized with
    :class:`.BatchStash`.  Groups of these are sent to subprocesses for
    processing in to :class:`.Batch` instances.

    """
    batch_id: str = field()
    """The ID of the batch."""

    data_point_ids: Tuple[str] = field()
    """The IDs each data point in the setLevel."""

    split_name: str = field()
    """The split (i.e. ``train``, ``test``, ``validation``)."""

    torch_seed_context: Dict[str, Any] = field()
    """The seed context given by :class:`.TorchConfig`."""

    def __post_init__(self):
        if not isinstance(self.batch_id, str):
            raise ValueError(f'wrong id type: {type(self.batch_id)}')

    def __str__(self):
        return (f'{self.batch_id}: s={self.split_name} ' +
                f'({len(self.data_point_ids)})')

    def __repr__(self):
        return self.__str__()


@dataclass
class PredictionMapper(ABC):
    """Used by a top level client to create features used to create instances of
    :class:`.DataPoint` and map label classes from nominal IDs to their string
    representations.

    """

    @abstractmethod
    def create_features(self, data: Any) -> Tuple[Any]:
        """Create an instance of a feature from ``data``.

        :param data: data used to create data points

        :return: the data used in the initializer of the respective (in list)
                 :class:`.DataPoint`

        """
        pass

    @abstractmethod
    def get_classes(self, nominals: Iterable[int]) -> List[str]:
        """Return the label string values for indexes ``nominals``.

        :param nominals: the integers that map to the respective string class

        """
        pass
