"""Provides a class to graph the results.

"""
__author__ = 'Paul Landes'


from dataclasses import dataclass, field
from typing import List
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
from . import ResultsContainer, ModelResult

logger = logging.getLogger(__name__)


@dataclass
class ModelResultGrapher(object):
    """Graphs the an instance of ``ModelResult``.  This creates subfigures,
    one for each of the results given as input to ``plot``.

    :param name: the name that goes in the title of the graph
    :param figsize: the size of the top level figure (not the panes)
    :param split_types: the splits to graph (list of size 2); defaults to
                        ['train', 'validation']
    :param title: the title format used to create each sub pane graph.

    :see: plot

    """
    name: str = field(default=None)
    figsize: List[int] = (15, 10)
    split_types: List[str] = None
    title: str = None

    def __post_init__(self):
        if self.split_types is None:
            self.split_types = 'train validation'.split()
        else:
            self.split_types = self.split_types
        if self.title is None:
            self.title = ('Figure {r.index} ' +
                          '(lr={r.model_settings.learning_rate:.5f}, ' +
                          'F1={r.micro_metrics[f1]:.3f})')

    def _render_title(self, cont: ResultsContainer) -> str:
        return self.title.format(**{'r': cont})

    def plot(self, containers: List[ModelResult], show: bool = False):
        name = containers[0].name if self.name is None else self.name
        ncols = min(2, len(containers))
        nrows = math.ceil(len(containers) / ncols)
        logger.debug(f'plot grid: {nrows} X {ncols}')
        fig, axs = plt.subplots(
            ncols=ncols, nrows=nrows, sharex=True, figsize=self.figsize)
        logger.debug(f'ax type: {type(axs)}')
        if not isinstance(axs, np.ndarray):
            logger.debug(f'adding dim')
            axs = np.array([[axs]])
        if axs.shape == (ncols,):
            axs = np.expand_dims(axs, axis=0)
        logger.debug(f'ax shape: {axs.shape}')
        fig.suptitle(f'Training and Validation Learning Rates: {name}')
        handles = []
        row = 0
        col = 0
        for i, cont in enumerate(containers):
            logger.debug(f'plotting {cont}')
            es = tuple(map(lambda n: (n.capitalize(), cont.dataset_result[n]),
                           self.split_types))
            x = range(len(es[0][1].losses))
            ax = axs[row][col]
            ax.plot(x, es[0][1].losses, color='r', label=es[0][0])
            ax.plot(x, es[1][1].losses, color='b', label=es[1][0])
            ax.set_title(self._render_title(cont))
            handles.append(ax)
            ax.set(xlabel='Epochs', ylabel='Loss')
            col += 1
            if col == ncols:
                col = 0
                row += 1
        plt.legend(tuple(map(lambda e: e[0], es)))
        if show:
            plt.show()
