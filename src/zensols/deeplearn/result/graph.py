"""Provides a class to graph the results.

"""
__author__ = 'Paul Landes'

from typing import List, Tuple
from dataclasses import dataclass, field
import logging
from pathlib import Path
from zensols.util import APIError
from zensols.deeplearn import DatasetSplitType
from . import ModelResult, DatasetResult

logger = logging.getLogger(__name__)


@dataclass
class ModelResultGrapher(object):
    """Graphs the an instance of :class:`.ModelResult`.  This creates
    subfigures, one for each of the results given as input to :meth:`plot`.

    """
    name: str = field(default=None)
    """The name that goes in the title of the graph."""

    figsize: Tuple[int, int] = field(default=(15, 5))
    """the size of the top level figure (not the panes)"""

    split_types: List[DatasetSplitType] = field(default=None)
    """The splits to graph (list of size 2); defaults to
    ``[DatasetSplitType.train, DatasetSplitType.validation]``.

    """
    title: str = field(default=None)
    """The title format used to create each sub pane graph."""

    save_path: Path = field(default=None)
    """Where the plot is saved."""

    def __post_init__(self):
        if self.split_types is None:
            self.split_types = [DatasetSplitType.train,
                                DatasetSplitType.validation]
        if self.title is None:
            self.title = ('Result {r.name} ' +
                          '(lr={learning_rate:e}, ' +
                          '{r.last_test.converged_epoch.metrics})')

    def _render_title(self, cont: ModelResult) -> str:
        lr = cont.model_settings['learning_rate']
        return self.title.format(**{'r': cont, 'learning_rate': lr})

    def plot_loss(self, containers: List[ModelResult]):
        from zensols.datdesc.figure import Figure
        from .plots import LossPlot

        name: str = containers[0].name if self.name is None else self.name
        title: str = f'Training and Validation Learning Rates: {name}'
        fig = Figure(
            name=title,
            title_font_size=14,
            width=self.figsize[0],
            height=self.figsize[1] * len(containers))
        fig.path = self.save_path
        row: int
        cont: ModelResult
        for row, cont in enumerate(containers):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'plotting {cont}')
            results: Tuple[Tuple[str, DatasetResult], ...] = tuple(map(
                lambda n: (n.name.capitalize(), cont.dataset_result[n]),
                self.split_types))
            plot = LossPlot(title=self._render_title(cont), row=row)
            name: str
            result: DatasetResult
            for name, result in results:
                plot.add(name, result.losses)
            fig.add_plot(plot)
            self._fig = fig

    def show(self):
        """Render and display the plot."""
        self._fig.show()

    def save(self):
        """Save the plot to disk."""
        if not hasattr(self, '_fig'):
            raise APIError('Must first call :meth:`plot` before saving')
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'saving results graph to {self.save_path}')
        self._fig.save()
        self._fig.clear()
