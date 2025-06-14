"""Tools to generate and visualize contingency matrices.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import List, Dict, Set, ClassVar
from dataclasses import dataclass, field
import pandas as pd
from sklearn import metrics as mt
from zensols.config import Dictable
from zensols.datdesc import DataFrameDescriber
from zensols.datdesc.fig import Figure, Plot
from zenosls.datdesc.plots import HeatMapPlot


@dataclass
class ConfusionMatrix(Dictable):
    """Create confusion matrix and optionally heat map representation.

    """
    _DICTABLE_ATTRIBUTES: ClassVar[Set[str]] = {'matrix'}
    _DICTABLE_WRITABLE_DESCENDANTS = True

    preds: DataFrameDescriber = field(repr=False)
    """A dataframe of gold labels (column ``label``) and predictions (column
    ``pred``).

    """
    @property
    def matrix(self) -> DataFrameDescriber:
        """The confusion matrix of the predictions in :obj:`preds`."""
        df: pd.DataFrame = self.preds.df
        name: str = self.preds.desc
        labs: List[str] = df['label'].drop_duplicates().sort_values().to_list()
        cm = mt.confusion_matrix(df['label'], df['pred'], labels=labs)
        dfc = pd.DataFrame(cm, index=labs, columns=labs)
        dfc.index.name = 'label \\ prediction'
        idesc: str = 'Labels are rows, predictions are columns.'
        imeta: Dict[str, str] = {i: i for i in labs}
        imeta[dfc.index.name] = idesc
        return DataFrameDescriber(
            name=f'{name}-confusion',
            desc=f'Confusion matrix of dataset: {name}',
            df=dfc,
            index_meta=imeta)

    def add_plot(self, fig: Figure) -> Plot:
        """Ad a heat map plot of :obj:`matrix` to figure ``fig``."""
        return fig.create(
            name=HeatMapPlot,
            format='d',
            title=self.preds.desc,
            dataframe=self.matrix.df,
            x_label_rotation=50,
            params={'cbar': False})
