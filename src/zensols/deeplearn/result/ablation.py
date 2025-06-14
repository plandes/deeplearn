"""Feature ablation graph construction using summary results.

"""
__author__ = 'Paul Landes'

from typing import Any, Dict, Tuple, ClassVar, List
from dataclasses import dataclass, field
import pandas as pd
from matplotlib.pyplot import Axes
from zensols.persist import persisted
from zensols.config import Dictable
from zensols.datdesc.figure import Figure, Plot
from zensols.datdesc.plots import PointPlot
from zensols.deeplearn.result import (
    ModelResultManager, ModelResultReporter, Metrics, ScoreMetrics,
    ModelResult, DatasetResult, EpochResult, PredictionsDataFrameFactory
)
from . import ModelResultError


@dataclass
class FeatureAblationPlot(PointPlot):
    """Plot ablation as feature sets with peformance scores.  The :obj:`lines`
    dataframe columns are:

        * ``epoch``: for the X values
        * ``performance``: for the Y values

    The :meth:`add_line` method adds a dataset split curves where the ``name``
    is the split name and ``line`` are the loss values.

    """
    test_performance: pd.DataFrame = field(default=None)
    """The performance for each feature set on the test set.  The dataframe has
    the following columns:

        * :obj:`test_performance_column`: the feature sets
        * ``epoch``: see class docs
        * ``performance``: see class docs

    """
    test_performance_column: str = field(default='features')
    """The name of the column in :obj:`test_performance` with the feature sets.

    """
    test_performance_plot_params: Dict[str, Any] = field(
        default_factory=lambda: dict(markers=r'^', markersize=14, alpha=0.8))
    """Parameters given to the test performance plot."""

    def __post_init__(self):
        def resolve_palette(n_colors: int) -> Tuple[int, int, int]:
            import seaborn as sns
            return sns.color_palette('husl', n_colors=n_colors)

        super().__post_init__()
        if self.sample_rate > 0:
            raise ModelResultError('Sample rate > 0 is not supported (train perf sync)')
        self._set_defaults(
            title='Model Ablation by Feature',
            key_title='Feature Set',
            x_axis_name='Epoch',
            y_axis_name='Performance',
            x_column_name='epoch',
            y_column_name='performance',
            palette=resolve_palette)
        self.plot_params = dict(markersize=4, linewidth=1.5)

    def _render_test_perf(self, axes: Axes):
        import seaborn as sns
        df: pd.DataFrame = self.test_performance
        hue_name: str = self.title
        x_axis_name: str = self.x_axis_name
        y_axis_name: str = self.y_axis_name
        x_column_name: str = self.x_column_name
        y_column_name: str = self.y_column_name
        df = df.rename(columns={x_column_name: x_axis_name,
                                y_column_name: y_axis_name,
                                self.test_performance_column: hue_name})
        hue_names: List[str] = df[hue_name].drop_duplicates().to_list()
        params: Dict[str, Any] = dict(
            ax=axes, data=df, x=x_axis_name, y=y_axis_name, hue=hue_name,
            legend=False,
            palette=self._get_palette(hue_names))
        params.update(self.test_performance_plot_params)
        sns.pointplot(**params)

    def _render(self, axes: Axes):
        super()._render(axes)
        self._render_test_perf(axes)


@dataclass
class FeatureAblationResults(Dictable):
    """Create feature ablation results.

    :see: :class:`.plots.FeatureAblationPlot`

    """
    _FEAT_COL: ClassVar[str] = 'features'
    _PERF_COL: ClassVar[str] = 'performance'
    _EPOC_COL: ClassVar[str] = 'epoch'

    reporter: ModelResultReporter = field()
    """The reporter used to obtain the results to create the graph."""

    average: str = field(default='weighted')
    """The type of metric average to use for the performance metric."""

    metric: str = field(default='f1')
    """The name of the metric to use for the performance values."""

    append_metric_to_legend: bool = field(default=True)
    """Whether or not to append the metric on the test dataset to the legend."""

    conv_flatten: bool = field(default=False)
    """Whether to max performance metrics to the last highest for epochs after
    the converged epoch.

    """
    epoch_start: int = field(default=0)
    """The left boundary used to clamp the epochs reported."""

    epoch_end: int = field(default=None)
    """The right boundary used to clamp the epochs reported."""

    plot_params: Dict[str, Any] = field(default_factory=dict)
    """Additional args in the "meth:`.Figure.create` call."""

    def _to_performance_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Morph the dataframe into one usable by the plot"""
        abbrev: str = PredictionsDataFrameFactory.\
            METRIC_AVERAGE_TO_COLUMN[self.average]
        col = f'{abbrev}{self.metric.upper()}t'
        df = df.rename(columns={col: self._PERF_COL})
        df = df.loc[df[self._FEAT_COL].drop_duplicates().index]
        df = df.sort_values(self._PERF_COL, ascending=False)
        return df

    def _append_metric_desc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add performance metrics to each feature set description so they show
        up in in the graph's legend.

        """
        def map_row(r: pd.Series) -> str:
            return f"{r[self._FEAT_COL]} ({r[self._PERF_COL]:.2f})"

        df[self._FEAT_COL] = df.apply(map_row, axis=1)
        return df

    @property
    @persisted('_summary_dataframe')
    def summary_dataframe(self) -> pd.DataFrame:
        """The multi-run summary used to create the graph.  It is modified to be
        used in the ablation graph.

        """
        df: pd.DataFrame = self.reporter.dataframe_describer.df
        df = self._to_performance_frame(df)
        if self.append_metric_to_legend:
            df = self._append_metric_desc(df)
        return df

    def _get_epoch_metrics(self) -> pd.DataFrame:
        """Return the performance metric for epochs of each result in the
        summary sets.

        """
        df: pd.DataFrame = self.summary_dataframe
        result_manager: ModelResultManager = self.reporter.result_manager
        met_name: str = self.metric
        rows: List[Tuple[Any, ...]] = []
        resid: str
        for resid in df['resid'].values:
            res: ModelResult = result_manager.results_stash[resid].model_result
            val: DatasetResult = res.validation
            epochs: List[EpochResult] = val.results
            epoch: EpochResult
            for epoch in epochs:
                mets: Metrics = epoch.metrics
                ave: ScoreMetrics = getattr(mets, self.average)
                row: Dict[str, Any] = {
                    'resid': resid,
                    self._EPOC_COL: epoch.index,
                    self._PERF_COL: getattr(ave, met_name)}
                rows.append(row)
        return pd.DataFrame(rows)

    def add_plot(self, fig: Figure) -> Plot:
        """Add a new ablation plot to ``fig``."""
        mn: str = PredictionsDataFrameFactory.METRIC_NAME_TO_COLUMN[self.metric]
        epoch_start: int = self.epoch_start
        epoch_end: int = self.epoch_end
        dfs: pd.DataFrame = self.summary_dataframe
        dfe: pd.DataFrame = self._get_epoch_metrics()
        dft: pd.DataFrame = dfs.rename(columns={'converged': self._EPOC_COL})
        feat_perfs: List[Tuple[str, pd.DataFrame]] = []
        if epoch_end is None:
            epoch_end = dfs['converged'].max()
        srow: pd.Series
        for _, srow in dfs.iterrows():
            resid: str = srow['resid']
            features: str = srow[self._FEAT_COL]
            line: pd.DataFrame = dfe[dfe['resid'] == resid]
            if self.conv_flatten:
                conv: int = srow['converged']
                line.loc[line.iloc[conv:].index, self._PERF_COL] = \
                    line[self._PERF_COL].max()
            line = line[(line[self._EPOC_COL] >= epoch_start) &
                        (line[self._EPOC_COL] <= epoch_end)]
            feat_perfs.append((features, line))
        dft = dft[(dft[self._EPOC_COL] >= epoch_start) &
                  (dft[self._EPOC_COL] <= epoch_end)]
        return fig.create(
            name=FeatureAblationPlot,
            x_column_name=self._EPOC_COL,
            y_column_name=self._PERF_COL,
            y_axis_name=f'{self.average.capitalize()} {mn.capitalize()}',
            data=feat_perfs,
            test_performance=dft,
            **self.plot_params)
