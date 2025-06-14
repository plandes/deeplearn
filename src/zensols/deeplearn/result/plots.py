"""Deep learning plots

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass
from zensols.datdesc.plots import PointPlot


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
        default_palette_set: bool = self.palette is None
        self._set_defaults(
            title='Model Loss',
            key_title='Dataset Split',
            x_axis_name='Epocs',
            y_axis_name='Loss',
            x_column_name='epocs',
            y_column_name='loss')
        if default_palette_set or True:
            # overwrite superclass pallet's ``set_defaults`` if not set before
            self.palette = 'rbgm'
