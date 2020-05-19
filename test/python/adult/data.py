from dataclasses import dataclass
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from zensols.deeplearn import (
    DataframeStash,
    Batch,
    BatchFeatureMapping,
)

logger = logging.getLogger(__name__)


@dataclass
class AdultDatasetStash(DataframeStash):
    """Load data from the adult dataset.

    :see https://archive.ics.uci.edu/ml/machine-learning-databases/adult:
    :see https://www.kaggle.com/kashnitsky/a1-demo-pandas-and-uci-adult-dataset:
    """
    CONTINUOUS = set("""age fnlwgt education_num capital_gain
                        capital_loss hours_per_week""".split())
    DESCRETE = set("""race sex country education martial_status relationship
                      workclass target occupation""".split())
    LABEL = 'target'
    NONE = '<none>'

    train_path: Path
    test_path: Path
    metadata_path: Path
    validation_portion: float

    def _load_file(self, path: Path) -> pd.DataFrame:
        logger.debug(f'loading training from {path}')
        df = pd.read_csv(path)
        df = df.rename(columns={c: c.lower() for c in df.columns})
        df = df[df[self.LABEL].isnull() == False]
        for c in self.DESCRETE:
            df[c] = df[c].str.strip().replace(np.nan, self.NONE, regex=True)
        df = df.astype({c: 'float64' for c in self.CONTINUOUS})
        return df

    def clear(self):
        super().clear()
        self._metadata.clear()

    def _get_dataframe(self) -> pd.DataFrame:
        df = self._load_file(self.train_path)
        val_size = int(self.validation_portion * df.shape[0])
        train_size = df.shape[0] - val_size
        df_val = df.iloc[:val_size]
        df_train = df.iloc[val_size:]
        logger.debug(f'val: {val_size} ({df_val.shape}), ' +
                     f'train: {train_size} ({df_train.shape})')
        assert val_size == df_val.shape[0]
        assert train_size == df_train.shape[0]
        df_val[self.split_col] = 'val'
        df_train[self.split_col] = 'train'
        df_test = self._load_file(self.test_path)
        df_test[self.split_col] = 'test'
        df = pd.concat([df_val, df_train, df_test], ignore_index=True)
        df[self.LABEL] = df[self.LABEL].replace(r'\.', '', regex=True)
        df.index = df.index.map(str)
        logger.debug(f'dataframe: {df.shape}')
        return df

    def write_metadata(self, writer=sys.stdout):
        from pprint import pprint
        pprint(self.metadata, stream=writer, width=150, indent=2)
