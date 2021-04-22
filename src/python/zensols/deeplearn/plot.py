"""Graphcal plotting convenience utilities.

"""
__author__ = 'Paul Landes'

import logging
import pylab
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
from . import DeepLearnError

logger = logging.getLogger(__name__)


class PlotManager(object):
    """A Convenience class to give window geomtry placement and blocking.

    """
    def __init__(self, geometry=(50, 0), size=(5, 5), block=False):
        self.geometry = '+{}+{}'.format(*geometry)
        logger.debug('using geometry: {} -> {}'.format(
            geometry, self.geometry))
        self.size = size
        self.block = block

    @staticmethod
    def clear():
        global _plot_mng_fig
        if '_plot_mng_fig' in globals():
            del globals()['_plot_mng_fig']

    @property
    def fig(self):
        return self.get_fig()

    def get_fig(self, *args, **kwargs):
        if not hasattr(self, '_fig'):
            global _plot_mng_fig
            if '_plot_mng_fig' in globals():
                plt.close(_plot_mng_fig)
            _plot_mng_fig = self._fig = plt.figure(
                *args, figsize=self.size, **kwargs)
        return self._fig

    @property
    def ax(self):
        return self.subplots()

    def subplots(self, *args, **kwargs):
        return self.fig.subplots(*args, **kwargs)

    def subplot(self, *args, **kwargs):
        # 1, 1, 1
        return self.fig.add_subplot(*args, **kwargs)

    def show(self):
        mng = pylab.get_current_fig_manager()
        mng.window.wm_geometry(self.geometry)
        self.fig.tight_layout()
        plt.show(block=self.block)

    def save(self, fig_path=None, *args, **kwargs):
        if fig_path is None:
            fig_path = 'fig.png'
        logger.info(f'saving output figure to {fig_path}')
        plt.savefig(fig_path, *args, **kwargs)


class DensityPlotManager(PlotManager):
    """Create density plots.

    """

    def __init__(self, data, covariance_factor=0.5, interval=None, margin=None,
                 *args, **kwargs):
        """Initailize.

        :param covariance_factor: smooth factor for visualization only

        """
        super(DensityPlotManager, self).__init__(*args, **kwargs)
        self.interval = interval
        self.margin = margin
        self.covariance_factor = covariance_factor
        self.data = data

    def plot(self):
        data = self.data
        ax = self.ax
        density = gaussian_kde(data)
        if ax is None:
            ax = self.ax
        if self.interval is None:
            self.interval = (min(data), max(data))
        if self.margin is None:
            self.margin = 0.2 * abs(self.interval[0] - self.interval[1])
        # create evenly spaced numbers over the probably range
        xs = np.linspace(
            self.interval[0] - self.margin, self.interval[1] + self.margin)
        logger.debug(f'data size: {len(data)}, X graph points: {len(xs)}')
        # smooth factor for visualization
        density.covariance_factor = lambda: self.covariance_factor
        # compute probably density and plot
        density._compute_covariance()
        logger.debug(f'plotting with ax: {ax}')
        ax.plot(xs, density(xs))


class GraphPlotManager(PlotManager):
    """Convenience class for plotting ``networkx`` graphs.

    """
    def __init__(self, graph, style='spring', pos=None, *args, **kwargs):
        super(GraphPlotManager, self).__init__(*args, **kwargs)
        self.graph = graph
        self.style = style
        self.pos = pos
        self.set_draw_arguments()

    def set_draw_arguments(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def _get_layout_function(self):
        import networkx as nx
        style = self.style
        if style == 'spectral':
            layoutfn = nx.spectral_layout
        elif style == 'circular':
            layoutfn = nx.circular_layout
        elif style == 'spring':
            layoutfn = nx.spring_layout
        elif style == 'shell':
            layoutfn = nx.shell_layout
        elif style == 'kamada':
            layoutfn = nx.kamada_kawai_layout
        elif style == 'planar':
            layoutfn = nx.layout.planar_layout
        elif style == 'random':
            layoutfn = nx.layout.random_layout
        else:
            raise DeepLearnError(f'No such layout: {style}')
        return layoutfn

    def _get_pos(self):
        if self.pos is None:
            layoutfn = self._get_layout_function()
            pos = layoutfn(self.graph)
        else:
            pos = self.pos
        return pos

    def show(self):
        import networkx as nx
        nxg = self.graph
        ax = self.ax
        pos = self._get_pos()
        nx.draw_networkx(nxg, pos=pos, ax=ax, *self.args, **self.kwargs)
        super(GraphPlotManager, self).show()
