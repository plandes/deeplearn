"""Common used plots for ML.

"""
__author__ = 'Paul Landes'

from typing import Tuple, List, Dict, Sequence, Iterable, Any, Union, Callable
from dataclasses import dataclass, field
import itertools as it
import math
from matplotlib.pyplot import Axes
import pandas as pd
from zensols.util import APIError
from .fig import Plot


@dataclass
class PaletteContainerPlot(Plot):
    """A base class that supports creating a color palette for subclasses.

    """
    palette: Union[str, Callable] = field(default=None)
    """Either the a list of color characters or a callable that takes the number
    of colors as input.  For example, the Seaborn color palette (such as
    ``sns.color_palette('tab10', n_colors=n)``).  This is used as the
    ``palette`` parameter in the ``sns.pointplot`` call.

    """
    def _get_palette(self, hue_names: Sequence[str]) -> \
            Dict[str, Tuple[int, int, int]]:
        palette: Union[str, Callable] = self.palette
        n_colors = len(hue_names)
        colors: Tuple[int, int, int]
        if isinstance(palette, Callable):
            colors = palette(n_colors)
        elif isinstance(palette, str):
            p_len: int = len(palette)
            n_iters: int = math.ceil(n_colors / p_len)
            colors = tuple(it.chain.from_iterable(
                it.repeat(list(palette), n_iters)))[:n_colors]
        else:
            raise APIError(f'Unknown palette type: {type(palette)}')
        return dict(zip(hue_names, colors))


@dataclass
class PointPlot(PaletteContainerPlot):
    """An abstract base class that renders overlapping lines that uses a
    :mod:`seaborn` ``pointplot``.

    """
    data: List[Tuple[str, pd.DataFrame]] = field(
        default_factory=list, repr=False)
    """The data to plot.  Each element is tuple first components with the plot
    name.  The second component is a dataframe with columns:

        * :obj:`x_column_name`: the X values of the graph, usually an
          incrementing number

        * :obj:`y_column_name`: a list loss float values

    Optionally use :meth:`add_line` to populate this list.

    """
    x_axis_name: str = field(default=None)
    """The axis name with the X label."""

    y_axis_name: str = field(default=None)
    """The axis name with the Y label."""

    x_column_name: str = field(default=None)
    """The :obj:`data` column with the X values."""

    y_column_name: str = field(default=None)
    """The :obj:`data` column with the Y values."""

    key_title: str = field(default=None)
    """The title that goes in the key."""

    sample_rate: int = field(default=0)
    """Every $n$ data point in the list of losses is added to the plot."""

    plot_params: Dict[str, Any] = field(
        default_factory=lambda: dict(markersize=0, linewidth=1.5))
    """Parameters given to :func:`seaborn.plotpoint`.  The default are
    decorative parameters for the marker size and line width.

    """
    def add(self, name: str, line: Iterable[float]):
        """Add the losses of a dataset by adding X values as incrementing
        integers the size of ``line``.

        :param name: the line name

        :param line: the Y values for the line

        """
        line = tuple(line)
        n: int = len(line)
        df = pd.DataFrame(
            data=tuple(range(1, n + 1)),
            columns=[self.x_column_name])
        df[self.y_column_name] = line
        self.data.append((name, df))

    def _render(self, axes: Axes):
        import seaborn as sns
        data: Sequence[Tuple[str, pd.DataFrame]] = self.data
        hue_name: str = self.title
        x_axis_name: str = self.x_axis_name
        y_axis_name: str = self.y_axis_name
        x_column_name: str = self.x_column_name
        y_column_name: str = self.y_column_name
        df: pd.DataFrame = None
        assert len(data) > 0
        desc: str
        dfl: pd.DataFrame
        for desc, dfl in data:
            dfl = dfl[[y_column_name, x_column_name]]
            dfl = dfl.rename(columns={y_column_name: desc})
            if df is None:
                df = dfl
            else:
                df = df.merge(
                    dfl, left_on=x_column_name, right_on=x_column_name,
                    suffixes=(None, None))
        if self.sample_rate > 0:
            df = df[(df.index % self.sample_rate) == 0]
        df = df.rename(columns={x_column_name: x_axis_name})
        df = df.melt(x_axis_name, var_name=hue_name, value_name=y_axis_name)
        hue_names: List[str] = df[hue_name].drop_duplicates().to_list()
        params: Dict[str, Any] = dict(
            ax=axes, data=df, x=x_axis_name, y=y_axis_name, hue=hue_name,
            palette=self._get_palette(hue_names))
        params.update(self.plot_params)
        sns.pointplot(**params)
        self._set_legend_title(axes, self.key_title)


@dataclass
class LossPlot(PointPlot):
    """A training and validation loss plot.  The :obj:`data` dataframe columns
    are:

        * ``epoch``: for the X values
        * ``loss``: for the Y values

    The :meth:`add_line` method adds a dataset split curves where the ``name``
    is the split name and ``line`` are the loss values.

    """
    def __post_init__(self):
        super().__post_init__()
        self._set_defaults(
            title='Model Loss',
            key_title='Dataset Split',
            palette='rbgm',
            x_axis_name='Epocs',
            y_axis_name='Loss',
            x_column_name='epocs',
            y_column_name='loss')


@dataclass
class HistPlot(PaletteContainerPlot):
    """Create a histogram plot using :meth:`seaborn.histplot`.

    """
    data: List[Tuple[str, pd.DataFrame]] = field(
        default_factory=list, repr=False)
    """The data to plot.  Each element is tuple first components with the plot
    name.

    """
    x_axis_label: str = field(default=None)
    """The axis name with the X label."""

    y_axis_label: str = field(default=None)
    """The axis name with the Y label."""

    key_title: str = field(default=None)
    """The title that goes in the key."""

    log_scale: float = field(default=None)
    """See the :meth:`seaborn.histplot` ``log_scale`` parameter.  This is also
    used to update the ticks if provided.

    """
    plot_params: Dict[str, Any] = field(default_factory=dict)
    """Parameters given to :func:`seaborn.histplot`."""

    def __post_init__(self):
        import seaborn as sns
        self._set_defaults(
            palette=lambda n: sns.color_palette('hls', n_colors=n))

    def add(self, name: str, data: Iterable[float]):
        """Add occurances to use in the histogram.

        :param name: the variable name

        :param data: the data to render

        """
        self.data.append((name, pd.DataFrame(data, columns=[name])))

    def _render(self, axes: Axes):
        import math
        import matplotlib.ticker as ticker
        import seaborn as sns
        dfs: pd.DataFrame = []
        hue_col: str = 'name'
        value_col: str = 'occur'

        # create column-singleton dataframes with the occurance data with a name
        # column for the hue
        name: str
        df: pd.DataFrame
        for name, df in self.data:
            dfg: pd.DataFrame = df[name].to_frame().\
                rename(columns={name: value_col})
            dfg[hue_col] = name
            dfs.append(dfg)
        params: Dict[str, Any] = dict(
            # dataframe of occurancesand hue name
            data=pd.concat(dfs),
            # subplot
            ax=axes,
            # occurances column in the agg dataframe
            x=value_col,
            # hue identifier
            hue=hue_col,
            # palette's colors are the hues of variables
            palette=self._get_palette(tuple(map(lambda t: t[0], self.data))),
            **self.plot_params)
        # log_scale is treated separately to recreate ticks
        if self.log_scale is not None:
            params['log_scale'] = self.log_scale
        sns.histplot(**params)
        # create human readable ticks
        # https://stackoverflow.com/questions/53747298/how-to-format-axis-tick-labels-from-number-to-thousands-or-millions-125-436-to
        if self.log_scale is not None:
            axes.xaxis.set_major_formatter(ticker.FuncFormatter(
                lambda x, pos: f'{math.log(x, self.log_scale) * 10:.0f}'))
        # set x/y axis text
        self._set_axis_labels(axes, self.x_axis_label, self.y_axis_label)
        # set the legend title or hide the hue_col text if not set
        self._set_legend_title(axes, self.key_title)


@dataclass
class HeatMapPlot(Plot):
    """Create heat map plot and optionally normalize.  This uses
    :mod:`seaborn`'s ``heatmap``.

    """
    dataframe: pd.DataFrame = field(default=None)
    """The data to render."""

    format: str = field(default='.2f')
    """The format of the plots's cell numerical values."""

    x_label_rotation: float = field(default=0)
    """The degree of label rotation."""

    params: Dict[str, Any] = field(default_factory=dict)
    """Additional parameters to give to :func:`seaborn.heatmap`."""

    def _render(self, axes: Axes):
        import seaborn as sns
        chart = sns.heatmap(ax=axes, data=self.dataframe,
                            annot=True, fmt=self.format, **self.params)
        if self.x_label_rotation != 0:
            axes.set_xticklabels(
                chart.get_xticklabels(),
                rotation=self.x_label_rotation)
