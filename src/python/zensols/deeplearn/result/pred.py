"""This creates Pandas dataframes containing predictions.

"""
__author__ = 'Paul Landes'

from typing import Callable, List, Iterable
from dataclasses import dataclass, field
import logging
import sys
import itertools as it
from pathlib import Path
import numpy as np
import pandas as pd
from zensols.persist import persisted
from zensols.deeplearn.vectorize import (
    CategoryEncodableFeatureVectorizer,
    AggregateEncodableFeatureVectorizer,
)
from zensols.deeplearn.batch import Batch, BatchStash, DataPoint
from . import ModelResultError, ModelResult, EpochResult

logger = logging.getLogger(__name__)


@dataclass
class PredictionsDataFrameFactory(object):
    """Create a Pandas data frame containing results from a result as output from a
    ``ModelExecutor``.  The data frame contains the feature IDs, labels,
    predictions mapped back to their original value from the feature data item.

    Currently only classification models are supported.

    """
    source: Path = field()
    """The source file from where the results were unpickled."""

    result: ModelResult = field()
    """The epoch containing the results."""

    stash: BatchStash = field()
    """The batch stash used to generate the results from the
    :class:`~zensols.deeplearn.model.ModelExecutor`.  This is used to get the
    vectorizer to reverse map the labels.

    """
    column_names: List[str] = field(default=None)
    """The list of string column names for each data item the list returned from
    ``data_point_transform`` to be added to the results for each
    label/prediction

    """
    data_point_transform: Callable[[DataPoint], tuple] = field(default=None)
    """A function that returns a tuple, each with an element respective of
    ``column_names`` to be added to the results for each label/prediction; if
    ``None`` (the default), ``str`` used (see the `Iris Jupyter Notebook
    <https://github.com/plandes/deeplearn/blob/master/notebook/iris.ipynb>`_
    example)

    """
    batch_limit: int = sys.maxsize
    """The max number of batche of results to output."""

    epoch_result: EpochResult = field(default=None)
    """The epoch containing the results.  If none given, take it from the test
    results..

    """
    def __post_init__(self):
        if self.column_names is None:
            self.column_names = ('data',)
        if self.data_point_transform is None:
            self.data_point_transform = lambda dp: (str(dp),)
        if self.epoch_result is None:
            self.epoch_result = self.result.test.results[0]

    @property
    def name(self) -> str:
        """The name of the results taken from :class:`.ModelResult`."""
        return self.result.name

    def _transform_dataframe(self, batch: Batch, labs: List[str],
                             preds: List[str]):
        transform: Callable = self.data_point_transform
        rows = []
        for dp, lab, pred in zip(batch.data_points, labs, preds):
            assert dp.label == lab
            row = [dp.id, lab, pred, lab == pred]
            row.extend(transform(dp))
            rows.append(row)
        cols = 'id label pred correct'.split() + list(self.column_names)
        return pd.DataFrame(rows, columns=cols)

    def _calc_len(self, batch: Batch) -> int:
        return len(batch)

    def _batch_dataframe(self) -> Iterable[pd.DataFrame]:
        """Return a data from for each batch.

        """
        epoch_labs: List[np.ndarray] = self.epoch_result.labels
        epoch_preds: List[np.ndarray] = self.epoch_result.predictions
        start = 0
        for bid in it.islice(self.epoch_result.batch_ids, self.batch_limit):
            batch: Batch = self.stash[bid]
            #end = start + len(batch)
            end = start + self._calc_len(batch)
            vec: CategoryEncodableFeatureVectorizer = \
                batch.get_label_feature_vectorizer()
            if isinstance(vec, AggregateEncodableFeatureVectorizer):
                vec = vec.delegate
            if not isinstance(vec, CategoryEncodableFeatureVectorizer):
                raise ModelResultError(
                    'Expecting a category feature vectorizer but got: ' +
                    f'{vec} ({vec.name})')
            inv_trans: Callable = vec.label_encoder.inverse_transform
            preds: List[str] = inv_trans(epoch_preds[start:end])
            labs: List[str] = inv_trans(epoch_labs[start:end])
            df = self._transform_dataframe(batch, labs, preds)
            df['batch_id'] = bid
            assert len(df) == len(labs)
            start = end
            yield df

    @property
    @persisted('_dataframe')
    def dataframe(self) -> pd.DataFrame:
        """Return the dataframe of results.  The first columns are generated from
        ``data_point_tranform``, and the remaining columns are:

        - id: the ID of the feature (not batch) data item
        - label: the label given by the feature data item
        - pred: the prediction
        - correct: whether or not the prediction was correct

        """
        return pd.concat(self._batch_dataframe(), ignore_index=True)


@dataclass
class SequencePredictionsDataFrameFactory(PredictionsDataFrameFactory):
    """Like the super class but create predictions for sequence based models.

    :see: :class:`~zensols.deeplearn.model.sequence.SequenceNetworkModule`

    """
    def _calc_len(self, batch: Batch) -> int:
        return sum(map(len, batch.data_points))

    def _transform_dataframe(self, batch: Batch, labs: List[str],
                             preds: List[str]):
        dfs: List[pd.DataFrame] = []
        start: int = 0
        transform: Callable = self.data_point_transform
        for dp, lab, pred in zip(batch.data_points, labs, preds):
            end = start + len(dp)
            df = pd.DataFrame({
                'id': dp.id,
                'label': labs[start:end],
                'pred': preds[start:end]})
            df[list(self.column_names)] = transform(dp)
            dfs.append(df)
            start = end
        return pd.concat(dfs)
