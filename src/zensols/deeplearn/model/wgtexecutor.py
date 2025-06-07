"""A class that weighs labels non-uniformly.

"""
__author__ = 'Paul Landes'

from typing import Dict, Any
from dataclasses import dataclass, field, InitVar
import logging
import collections
from pathlib import Path
import torch
from zensols.util import time
from zensols.persist import persisted, PersistedWork
from . import ModelExecutor

logger = logging.getLogger(__name__)


@dataclass
class WeightedModelExecutor(ModelExecutor):
    """A class that weighs labels non-uniformly.  This class uses invert class
    sampling counts to help the minority label.

    """
    weighted_split_name: str = field(default='train')
    """The split name used to re-weight labels."""

    weighted_split_path: InitVar[Path] = field(default=None)
    """The path to the cached weithed labels."""

    use_weighted_criterion: bool = field(default=True)
    """If ``True``, use the class weights in the initializer of the criterion.
    Setting this to ``False`` effectively disables this class.

    """

    def __post_init__(self, weighted_split_path: Path):
        super().__post_init__()
        if weighted_split_path is None:
            path = '_label_counts'
        else:
            file_name = f'weighted-labels-{self.weighted_split_name}.dat'
            path = weighted_split_path / file_name
        self._label_counts = PersistedWork(path, self)

    def clear(self):
        super().clear()
        self._label_counts.clear()

    @persisted('_label_counts')
    def get_label_counts(self) -> Dict[int, int]:
        stash = self.dataset_stash.splits[self.weighted_split_name]
        label_counts = collections.defaultdict(lambda: 0)
        batches = tuple(stash.values())
        for batch in batches:
            for label in batch.get_labels():
                label_counts[label.item()] += 1
        for batch in batches:
            batch.deallocate()
        return dict(label_counts)

    @persisted('_class_weighs')
    def get_class_weights(self) -> torch.Tensor:
        """Compute invert class sampling counts to return the weighted class.

        """
        counts = self.get_label_counts().items()
        counts = map(lambda x: x[1], sorted(counts, key=lambda x: x[0]))
        counts = self.torch_config.from_iterable(counts)
        return counts.mean() / counts

    def get_label_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Return a dictionary whose keys are the labels and values are
        dictionaries containing statistics on that label.

        """
        counts = self.get_label_counts()
        weights = self.get_class_weights().cpu().numpy()
        batch = next(iter(self.dataset_stash.values()))
        vec = batch.batch_stash.get_label_feature_vectorizer(batch)
        classes = vec.get_classes(range(weights.shape[0]))
        return {c[0]: {'index': c[1],
                       'count': counts[c[1]],
                       'weight': weights[c[1]]}
                for c in zip(classes, range(weights.shape[0]))}

    def _create_criterion(self) -> torch.optim.Optimizer:
        resolver = self.config_factory.class_resolver
        criterion_class_name = self.model_settings.criterion_class_name
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'criterion: {criterion_class_name}')
        criterion_class = resolver.find_class(criterion_class_name)
        with time('weighted classes'):
            class_weights = self.get_class_weights()
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'using class weights: {class_weights}')
        if self.use_weighted_criterion:
            inst = criterion_class(weight=class_weights)
        else:
            inst = criterion_class()
        return inst
