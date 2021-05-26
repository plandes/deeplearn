"""Utility functionality for dataframe related containers.

"""

from typing import Any
from dataclasses import dataclass
import sys
from io import TextIOBase
import json
import pandas as pd
from zensols.config import Dictable


@dataclass
class DataFrameDictable(Dictable):
    """A container with utility methods that JSON and write Pandas dataframes.

    """

    NONE_REPR = ''
    """String used for NaNs."""

    DEFAULT_COLS = 40
    """Default width when writing the dataframe."""

    WRITABLE__DESCENDANTS = True

    def _from_object(self, obj: Any, recurse: bool, readable: bool) -> Any:
        if isinstance(obj, pd.DataFrame):
            return json.loads(obj.to_json())
        else:
            return super()._from_object(obj, recurse, readable)

    def _write_dataframe(self, df: pd.DataFrame, depth: int = 0,
                         writer: TextIOBase = sys.stdout,
                         index: bool = True):
        df_str = df.to_string(index=index, na_rep=self.NONE_REPR,
                              max_colwidth=self.DEFAULT_COLS)
        self._write_block(df_str, depth, writer)
