"""Provides a class to graph the results.

"""
__author__ = 'Paul Landes'


from typing import List, Tuple
from dataclasses import dataclass, field
import logging
from pathlib import Path
import math
import matplotlib.pyplot as plt
import numpy as np
from zensols.deeplearn import DatasetSplitType
from . import ModelResult

logger = logging.getLogger(__name__)


@dataclass
class ModelResultGrapher(object):
    """Graphs the an instance of ``ModelResult``.  This creates subfigures,
    one for each of the results given as input to ``plot``.

    :param name: the name that goes in the title of the graph
    :param figsize: the size of the top level figure (not the panes)
    :param split_types: the splits to graph (list of size 2); defaults to
                        ``[DatasetSplitType.train, DatasetSplitType.validation]``
    :param title: the title format used to create each sub pane graph.

    :see: plot

    """
    name: str = field(default=None)
    figsize: Tuple[int, int] = (15, 5)
    split_types: List[DatasetSplitType] = None
    title: str = None
    save_path: Path = field(default=None)

    def __post_init__(self):
        if self.split_types is None:
            self.split_types = [DatasetSplitType.train,
                                DatasetSplitType.validation]
        if self.title is None:
            self.title = ('Figure {r.name} ' +
                          '(lr={learning_rate:.5f}, ' +
                          '{r.last_test.converged_epoch.metrics})')

    def _render_title(self, cont: ModelResult) -> str:
        lr = cont.model_settings['learning_rate']
        return self.title.format(**{'r': cont, 'learning_rate': lr})

    def plot(self, containers: List[ModelResult]):
        name = containers[0].name if self.name is None else self.name
        ncols = min(2, len(containers))
        nrows = math.ceil(len(containers) / ncols)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'plot grid: {nrows} X {ncols}')
        fig, axs = plt.subplots(
            ncols=ncols, nrows=nrows, sharex=True, figsize=self.figsize)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'ax type: {type(axs)}')
        if not isinstance(axs, np.ndarray):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('adding axs dim')
            axs = np.array([[axs]])
        if axs.shape == (ncols,):
            axs = np.expand_dims(axs, axis=0)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'ax shape: {axs.shape}')
        fig.suptitle(f'Training and Validation Learning Rates: {name}')
        handles = []
        row = 0
        col = 0
        for i, cont in enumerate(containers):
            logger.debug(f'plotting {cont}')
            es = tuple(
                map(lambda n: (n.name.capitalize(), cont.dataset_result[n]),
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

    def show(self):
        plt.show()

    def save(self):
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'saving results graph to {self.save_path}')
        plt.savefig(self.save_path)
        plt.close()
