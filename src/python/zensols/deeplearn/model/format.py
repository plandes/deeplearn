"""Contains a class to write performance metrics.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Dict, Iterable, Any, Union
from dataclasses import dataclass, field
import logging
import sys
import re
from io import TextIOBase
from pathlib import Path
import yaml
import pandas as pd
from zensols.config import Writable
from zensols.deeplearn.model import ModelFacade
from zensols.deeplearn.result import (
    ClassificationMetrics, PredictionsDataFrameFactory,
    ModelResultError, ModelResultManager, ModelResultReporter,
)

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetricsDumper(Writable):
    """Formats performance metrics, which can be used in papers.

    :see: :class:`.LatexPerformanceMetricsDumper`

    """
    facade: ModelFacade = field()
    """The facade used to fetch previously written results."""

    summary_columns: Tuple[str] = field(
        default=tuple('mF1t mPt mRt MF1t MPt MRt'.split()))
    """The columns used in the summary report."""

    by_label_columns: Tuple[str] = field(
        default=tuple('mF1 mP mR MF1 MP MR acc count'.split()))
    """The columns used in the by-label report."""

    name_replace: Tuple[str, str] = field(default=None)
    """If provided, a tuple of ``(regular expression, replacement)`` string
    given to :func:`re.sub` in the name column of generated tables.

    """
    sort_column: str = field(default='mF1')
    """The column to sort, with the exception of the majority label, which is
    always first.

    """
    majority_label_res_id: Union[str, bool] = field(default=True)
    """Indicates how to create (if any) the majority label performance metrics.
    If a string, use as the result id (``res_id``) of previous result set used
    to compute the majority label statitics to include in the summary.  If
    ``True`` use the results from the last tested model.  If ``None`` the
    majority label is not added.

    """
    precision: int = field(default=3)
    """The number of signification digits to format results."""

    @staticmethod
    def format_thousand(x: int, apply_k: bool = True,
                        add_comma: bool = True) -> str:
        add_k = False
        if x > 10000:
            if apply_k:
                x = round(x / 1000)
                add_k = True
        if add_comma:
            x = f'{x:,}'
        else:
            x = str(x)
        if add_k:
            x += 'K'
        return x

    @staticmethod
    def capitalize(name: str) -> str:
        return ' '.join(map(lambda s: s.capitalize(),
                            re.split(r'[ _-]', name)))

    @staticmethod
    def _map_col(col: str) -> str:
        desc = ModelResultReporter.METRIC_DESCRIPTIONS.get(col)
        if desc is not None:
            return f'{col} is the {desc}'

    def _map_name(self, name: str) -> str:
        m: re.Match = re.match(r'^(.+): (\d+)$', name)
        if m is None:
            raise ModelResultError(f'Unknown model name format: {name}')
        run_idx = int(m.group(2))
        if run_idx != 1:
            raise ModelResultError(
                f'Multiple runs not supported: {name} ({run_idx})')
        name = m.group(1)
        if self.name_replace is not None:
            name = re.sub(*self.name_replace, name)
        return name

    @property
    def summary_dataframe(self) -> pd.DataFrame:
        pcols = list(self.summary_columns)
        rcols = list(map(lambda x: x[:-1], pcols))
        rm: ModelResultManager = self.facade.result_manager
        reporter = ModelResultReporter(rm)
        reporter.include_validation = False
        df: pd.DataFrame = reporter.dataframe
        df = df[['name'] + pcols]
        df = df.rename(columns=dict(zip(pcols, rcols)))
        if self.sort_column is not None:
            df = df.sort_values(self.sort_column)
        df['name'] = df['name'].apply(self._map_name)
        if self.majority_label_res_id is not None:
            params = {}
            if isinstance(self.majority_label_res_id, str):
                params['name'] = self.majority_label_res_id
            pred_factory: PredictionsDataFrameFactory = \
                self.facade.get_predictions_factory(**params)
            mets: ClassificationMetrics = pred_factory.majority_label_metrics
            majlab = pred_factory.metrics_to_series('Majority Label', mets)
            majlab = majlab.rename({
                PredictionsDataFrameFactory.LABEL_COL: 'name'})
            dfm = pd.DataFrame([majlab[['name'] + rcols]])
            df = pd.concat((dfm, df), ignore_index=True)
        fmt = '{x:.%sf}' % self.precision
        for c in rcols:
            df[c] = df[c].apply(lambda x: fmt.format(x=x))
        df = df.rename(columns={'name': 'Name'})
        return df

    def _get_best_results(self) -> pd.DataFrame:
        rm: ModelResultManager = self.facade.result_manager
        reporter = ModelResultReporter(rm)
        reporter.include_validation = False
        df: pd.DataFrame = reporter.dataframe
        ix = df['wF1t'].idxmax()
        name, file_name = df.loc[ix, ['name', 'file']]
        df = self.facade.get_predictions_factory(
            name=file_name).metrics_dataframe
        return df

    @property
    def by_label_dataframe(self) -> pd.DataFrame:
        cols = list(self.by_label_columns)
        df: pd.DataFrame = self._get_best_results().copy()
        df = df[['label'] + cols]
        fmt = '{x:.%sf}' % self.precision
        for c in cols:
            if c == 'count':
                continue
            df[c] = df[c].apply(lambda x: fmt.format(x=x))
        crenames = dict(map(lambda c: (c, self.capitalize(c)),
                            'label correct acc count'.split()))
        df = df.rename(columns=crenames)
        if self.sort_column is not None:
            col = self.sort_column
            if self.sort_column == 'name':
                col = 'label'
            df = df.sort_values(col)
        return df

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              indent: int = 0):
        from tabulate import tabulate
        self._write_line('summary:', depth, writer)
        df = self.summary_dataframe
        content = tabulate(df, headers=df.columns, disable_numparse=True)
        self._write_block(content, depth + indent, writer)
        self._write_empty(writer)

        self._write_line('label:', depth, writer)
        df = self.by_label_dataframe
        content = tabulate(df, headers=df.columns, disable_numparse=True)
        self._write_block(content, depth + indent, writer)

    def __call__(self):
        self.write()


@dataclass
class LatexPerformanceMetricsDumper(PerformanceMetricsDumper):
    """Writes model performance metrics in data formats then used to import to
    the LaTeX typesetting system used by the Zensols build framework.  The class
    writes a YAML configuration used by `mklatextbl.py` script in the Zensols
    Build repo, which generates a LaTeX table.  The output is a ``.sty` style
    file with the table, which is included with ``usepackage`` and then added
    with a command.

    :see: `Zensols Build <https://github.com/plandes/zenbuild>`_

    :see: `mklatextbl.py <https://github.com/plandes/zenbuild/blob/master/bin/mklatextbl.py>`_

    """
    results_dir: Path = field(default=Path('results/perf'))
    """The path to the output CSV files with performance metrics."""

    config_dir: Path = field(default=Path('../config'))
    """The path to the YAML configuration files used by the ``mklatextbl.py``
    Zensols LaTeX table generator.

    """
    def _create_table(self, name: str, output_csv: Path, caption: str,
                      cols: Iterable[str]) -> Dict[str, Any]:
        desc = ', '.join(filter(lambda x: x is not None,
                                map(self._map_col, cols)))
        return {
            f'metrics{name}tab':
            {'path': f'../model/{output_csv}',
             # 'type': 'slack',
             # 'slack_col': 0,
             'caption': caption.format(**dict(desc=desc)),
             'placement': 'VAR',
             'size': 'small',
             'single_column': False,
             'uses': 'zentable'}}

    def dump_summary(self) -> Tuple[Path, Path]:
        """Dump summary of metrics to a LaTeX mktable YAML and CSV files.

        :return: a tuple of the output CSV and YAML files

        """
        output_csv: Path = self.results_dir / 'metrics-summary.csv'
        output_yml: Path = self.config_dir / 'metrics-summary-table.yml'
        df = self.summary_dataframe
        caption = 'Summarization of performance metrics where {desc}.'
        rcols = df.columns.to_list()[1:]
        table_def = self._create_table('summary', output_csv, caption, rcols)
        for path in (output_csv, output_yml):
            path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_yml, 'w') as f:
            yaml.dump(table_def, f)
        logger.info(f'wrote: {output_yml}')
        df.to_csv(output_csv, index=False)
        logger.info(f'wrote: {output_csv}')
        return (output_csv, output_yml)

    def dump_by_label(self) -> Tuple[Path, Path]:
        """Dump per label of metrics of the highest performing model to a LaTeX
        mktable YAML and CSV files.

        """
        output_csv: Path = self.results_dir / 'metrics-by-label.csv'
        output_yml: Path = self.config_dir / 'metrics-by-label-table.yml'
        df = self.by_label_dataframe
        caption = 'By label performance metrics where {desc}.'
        cols = self.by_label_columns
        table_def = self._create_table('label', output_csv, caption, cols)
        for path in (output_csv, output_yml):
            path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_yml, 'w') as f:
            yaml.dump(table_def, f)
        logger.info(f'wrote: {output_yml}')
        df.to_csv(output_csv, index=False)
        logger.info(f'wrote: {output_csv}')
        return (output_csv, output_yml)

    def __call__(self):
        self.dump_summary()
        self.dump_by_label()
