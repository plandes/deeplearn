"""Configuration classes using dataframes as sources.

"""
__author__ = 'Paul Landes'

from typing import Dict, Any
from pathlib import Path
import pandas as pd
from zensols.config import DictionaryConfig


class DataframeConfig(DictionaryConfig):
    """A :class:`~zensols.config.Configurable` that dataframes as sources.  This
    is useful for providing labels to nominial label vectorizers.

    """
    def __init__(self, csv_path: Path, default_section: str,
                 columns: Dict[str, str] = None, column_eval: str = None,
                 counts: Dict[str, str] = None):
        """Initialize the configuration from a dataframe (see parameters).

        :param csv_path: the path to the CSV file to create the dataframe

        :param default_section: the singleton section name, which has as options
                                a list of the columns of the dataframe

        :param columns: the columns to add to the configuration from the
                        dataframe with ``key, values`` as ``column names, option
                        names``

        :param column_eval: Python code to evaluate and apply to each column if
                            provided

        :param counts: additional option entries in the section to add as counts
                      of respective columns with ``key, values`` as ``column
                      option names, new entry option names; where the ``column
                      option names`` are those given as values from the
                      ``columns`` :class:`dict`

        """
        df: pd.DataFrame = pd.read_csv(csv_path)
        sec: Dict[str, Any] = {}
        if columns is None:
            columns = dict(map(lambda x: (x, x), df.columns))
        col_name: str
        for df_col, sec_name in columns.items():
            col: pd.Series = df[df_col]
            if column_eval is not None:
                col = eval(column_eval)
            if isinstance(col, pd.Series):
                col = col.tolist()
            sec[sec_name] = col
        if counts is not None:
            for col_name, sec_name in counts.items():
                sec[sec_name] = len(sec[col_name])
        super().__init__(config={default_section: sec},
                         default_section=default_section)
