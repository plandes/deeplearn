"""This creates Pandas dataframes containing predictions.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field
from typing import Callable, List, Iterable
import logging
import sys
import itertools as it
from pathlib import Path
import numpy as np
import pandas as pd
from zensols.persist import persisted
from zensols.deeplearn.vectorize import CategoryEncodableFeatureVectorizer
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

    def __post_init__(self):
        if self.column_names is None:
            self.column_names = ('data',)
        if self.data_point_transform is None:
            self.data_point_transform = lambda dp: (str(dp),)

    @property
    def name(self) -> str:
        """The name of the results taken from :class:`.ModelResult`."""
        return self.result.name

    @property
    def epoch_result(self) -> EpochResult:
        """The epoch containing the results."""
        return self.result.test.results[0]

    def _batch_data_frame(self) -> Iterable[pd.DataFrame]:
        """Return a data from for each batch.

        """
        transform = self.data_point_transform
        batches = zip(self.epoch_result.batch_ids,
                      self.epoch_result.batch_predictions,
                      self.epoch_result.batch_labels)
        i: int
        preds: List[np.ndarray]
        labs: List[np.ndarray]
        for i, preds, labs in it.islice(batches, self.batch_limit):
            batch: Batch = self.stash[i]
            vec: CategoryEncodableFeatureVectorizer = \
                batch.get_label_feature_vectorizer()
            if not isinstance(vec, CategoryEncodableFeatureVectorizer):
                raise ModelResultError(
                    f'expecting a category feature vectorizer but got: {vec}')
            inv_trans = vec.label_encoder.inverse_transform
            preds = inv_trans(preds)
            labs = inv_trans(labs)
            # preds = inv_trans(pred_lab_batch[0])
            # labs = inv_trans(pred_lab_batch[1])
            rows = []
            for dp, lab, pred in zip(batch.get_data_points(), labs, preds):
                assert dp.label == lab
                row = [dp.id, lab, pred, lab == pred]
                row.extend(transform(dp))
                rows.append(row)
            cols = 'id label pred correct'.split() + list(self.column_names)
            yield pd.DataFrame(rows, columns=cols)

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
        return pd.concat(self._batch_data_frame(), ignore_index=True)
