"""Graphcal plotting convenience utilities.

"""
__author__ = 'Paul Landes'

import logging
import pylab
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class PlotManager(object):
    """A Convenience class to give window geomtry placement and blocking.

    """
    def __init__(self, geometry=(50, 0), block=False):
        self.geometry = '+{}+{}'.format(*geometry)
        logger.debug('using geometry: {} -> {}'.format(
            geometry, self.geometry))
        self.block = block

    @staticmethod
    def clear():
        global _plot_mng_fig
        if '_plot_mng_fig' in globals():
            del globals()['_plot_mng_fig']

    @property
    def fig(self):
        if not hasattr(self, '_fig'):
            global _plot_mng_fig
            if '_plot_mng_fig' in globals():
                plt.close(_plot_mng_fig)
            _plot_mng_fig = self._fig = plt.figure()
        return self._fig

    @property
    def ax(self, *args, **kwargs):
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

    def __init__(self, covariance_factor=0.5, interval=None, margin=None,
                 *args, **kwargs):
        """
        :param covariance_factor: smooth factor for visualization only
        """
        super(DensityPlotManager, self).__init__(*args, **kwargs)
        self.interval = interval
        self.margin = margin
        self.covariance_factor = covariance_factor

    def plot_density(self, data, ax=None):
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

    def plot(self, data):
        logger.debug(f'plotting with {len(data)} data points')
        self.plot_density(data, None)
