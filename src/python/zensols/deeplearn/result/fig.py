"""A simple object oriented plotting API.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import Tuple, List, Dict, Set, Any, Union, Type, Callable, ClassVar
from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
from matplotlib.figure import Figure as MatplotFigure
from zensols.persist import (
    persisted, PersistedWork, FileTextUtil, Deallocatable
)
from zensols.config import Dictable, ConfigFactory

logger = logging.getLogger(__name__)


@dataclass
class Plot(Dictable, metaclass=ABCMeta):
    """An abstract base class for plots.  The subclass overrides :meth:`plot` to
    generate the plot.  Then the client can use :meth:`save` or :meth:`render`
    it.  The plot is created as a subplot providing attributes for space to be
    taken in rows, columns, height and width.

    """
    title: str = field(default=None)
    """The title to render in the plot."""

    row: int = field(default=0)
    """The row grid position of the plot."""

    column: int = field(default=0)
    """The column grid position of the plot."""

    post_hooks: List[Callable] = field(default_factory=list)
    """Methods to invoke after rendering."""

    def __post_init__(self):
        pass

    @abstractmethod
    def _render(self, axes: Axes):
        pass

    def render(self, axes: Axes):
        if self.title is not None:
            axes.set_title(self.title)
        self._render(axes)
        for hook in self.post_hooks:
            hook(self, axes)

    def _set_defaults(self, **attrs: Dict[str, Any]):
        """Unset member attributes are set to ``attribs``."""
        attr: str
        for attr, val in attrs.items():
            if getattr(self, attr) is None:
                setattr(self, attr, val)

    def _set_legend_title(self, axes: Axes, title: str = None):
        if title is None:
            axes.legend_.set_title(None)
        else:
            axes.legend(title=title)

    def _set_axis_labels(self, axes: Axes, x_label: str = None,
                         y_label: str = None):
        if x_label is not None:
            axes.set_xlabel(x_label)
        if y_label is not None:
            axes.set_ylabel(y_label)

    def __str__(self) -> str:
        cls: str = self.__class__.__name__
        return f'{self.title}({cls}): row={self.row}, col={self.column}'


@dataclass
class Figure(Deallocatable, Dictable):
    """An object oriented class to manage :class:`matplit.figure.Figure` and
    subplots (:class:`matplit.pyplot.Axes`).

    """
    _DICTABLE_ATTRIBUTES: ClassVar[Set[str]] = {'path'}

    name: str = field(default='Untitled')
    """Used for file naming and the title."""

    config_factory: ConfigFactory = field(default=None, repr=False)
    """The configuration factory used to create plots."""

    title_font_size: int = field(default=0)
    """The font size :obj:`name`.  A size of 0 means do not render the title.
    Typically a font size of 16 is appropriate.

    """
    height: int = field(default=5)
    """The height in inches of the entire figure."""

    width: int = field(default=5)
    """The width in inches of the entire figure."""

    padding: float = field(default=5.)
    """Tight layout padding."""

    metadata: Dict[str, str] = field(default_factory=dict)
    """Metadata added to the image when saved."""

    plots: Tuple[Plot, ...] = field(default=())
    """The plots managed by this object instance.  Use :meth:`add_plot` to add
    new plots.

    """
    image_dir: Path = field(default=Path('.'))
    """The default image save directory."""

    image_format: str = field(default='svg')
    """The image format to use when saving plots."""

    def __post_init__(self):
        super().__init__()
        self._subplots = PersistedWork('_subplots', self)
        self._rendered = False
        self._file_name = None

    def add_plot(self, plot: Plot):
        """Add to the collection of managed plots.  This is needed for the plot
        to work if not created from this manager instance.

        :param plot: the plot to be managed

        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'adding plot: {plot}')
        self.plots = (*self.plots, plot)
        self._reset()

    def create(self, name: Union[str, Type[Plot]], **kwargs) -> Plot:
        """Create a plot using the arguments of :class:`.Plot`.

        :param name: the configuration section name of the plot

        :param kwargs: the initializer keyword arguments when creating the plot

        """
        plot: Plot
        if isinstance(name, Type):
            plot = name(**kwargs)
        else:
            plot = self.config_factory.new_instance(name, **kwargs)
        self.add_plot(plot)
        return plot

    @persisted('_subplots')
    def _get_subplots(self) -> Axes:
        """The subplot matplotlib axes.  A new subplot is create on the first
        time this is accessed.

        """
        params: Dict[str, Any] = dict(
            ncols=max(map(lambda p: p.column, self.plots)) + 1,
            nrows=max(map(lambda p: p.row, self.plots)) + 1,
            figsize=(self.width, self.height))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'creating subplots: {params}')
        fig, axs = plt.subplots(**params)
        fig.tight_layout(pad=self.padding)
        if self.title_font_size > 0:
            fig.suptitle(self.name, fontsize=self.title_font_size)
        return fig, axs

    def _get_axes(self) -> Union[Axes, np.ndarray]:
        return self._get_subplots()[1]

    def _get_figure(self) -> MatplotFigure:
        """The matplotlib figure."""
        return self._get_subplots()[0]

    @property
    def path(self) -> Path:
        """The path of the image figure to save.  This is constructed from
        :obj:`image_dir`, :obj:`name` and :obj`image_format`.  Conversely,
        when set, it updates these fields.

        """
        file_name: str = None
        if hasattr(self, '_file_name') and self._file_name is not None:
            file_name = self._file_name
        else:
            file_name = FileTextUtil.normalize_text(self.name)
            file_name = f'{file_name}.{self.image_format}'
        return self.image_dir / file_name

    @path.setter
    def path(self, path: Path):
        """The path of the image figure to save.  This is constructed from
        :obj:`image_dir`, :obj:`name` and :obj`image_format`.  Conversely,
        when set, it updates these fields.

        """
        if path is None:
            if hasattr(self, '_file_name'):
                del self._file_name
        else:
            self._file_name = path.name
            self.image_dir = path.parent
            self.image_format = path.suffix[1:]

    @persisted('_matplotlib_offline', cache_global=True)
    def _set_matplotlib_offline(self):
        """Invoke the API to create images offline so headless Python
        interpreters don't raise exceptions for long running tasks such as
        training/testing a large model.  The method uses a ``@persisted`` with a
        global caching so it's only called once per interpreter life cycle.

        """
        import matplotlib
        matplotlib.use('agg')

    def _get_image_metadata(self) -> Dict[str, Any]:
        """Factory method to add metadata to the file.  By default,
        :obj:`metadata` is added and ``Title`` with the contents of
        :obj:`name`.

        """
        metadata: Dict[str, str] = {'Title': self.name}
        metadata.update(self.metadata)
        return metadata

    def _render(self):
        """Render the image using :meth:`.Plot.render`."""
        if not self._rendered:
            self._set_matplotlib_offline()
            axes: Union[Axes, np.ndarray] = self._get_axes()
            plot: Plot
            for plot in self.plots:
                ax: Axes = axes
                if isinstance(ax, np.ndarray):
                    if len(ax.shape) == 1:
                        ix = plot.row if plot.row != 0 else plot.column
                        ax = axes[ix]
                    else:
                        ax = axes[plot.row, plot.column]
                plot.render(ax)
            self._rendered = True

    def save(self) -> Path:
        """Save the figure of subplot(s) to at location :obj:`path`.

        :param: if provided, overrides the save location :obj:`path`

        :return: the value of :obj:`path`

        """
        path: Path = self.path
        self._render()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._get_figure().savefig(
            fname=path,
            format=self.image_format,
            bbox_inches='tight',
            metadata=self._get_image_metadata())
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'wrote: {path}')
        return path

    def show(self):
        """Render and display the plot."""
        plt.show()

    def _reset(self):
        """Reset the :mod:`matplotlib` module and any data structures."""
        if self._subplots.is_set():
            fig: MatplotFigure = self._get_figure()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'deallocating fig: {fig}')
            fig.clear()
        self._subplots.clear()
        self._rendered = False

    def clear(self):
        """Remove all plots and reset the :mod:`matplotlib` module."""
        self._reset()
        self.plots = ()

    def deallocate(self):
        self.clear()
