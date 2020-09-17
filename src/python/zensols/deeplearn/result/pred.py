"""This creates Pandas dataframes containing predictions.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field
from typing import Callable, List, Iterable
import logging
import sys
import itertools as it
import pandas as pd
from zensols.persist import persisted
from zensols.deeplearn.vectorize import FeatureVectorizerManagerSet
from zensols.deeplearn.batch import Batch, BatchStash, DataPoint
from zensols.deeplearn.result import EpochResult

logger = logging.getLogger(__name__)


@dataclass
class PredictionsDataFrameFactory(object):
    """Create a Pandas data frame containing results from a result as output from a
    ``ModelExecutor``.  The data frame contains the feature IDs, labels,
    predictions mapped back to their original value from the feature data item.

    :param epoch_result: the test instance of an epoch result to populate the
                         dataframe

    :param stash: the batch stash used in the ``ModelExecutor`` that was used
                  to generate the results (not the ``DatasetSplitStash``)

    :param column_names: the list of string column names for each data item the
                         list returned from ``data_point_transform`` to be
                         added to the results for each label/prediction

    :param data_point_transform:

        a function that returns a tuple, each with an element respective of
        ``column_names`` to be added to the results for each label/prediction;
        if ``None`` (the default), ``str`` used (see the `Iris Jupyter Notebook
        <https://github.com/plandes/deeplearn/blob/master/notebook/iris.ipynb>`_
        example)

    :param batch_limit: the max number of batche of results to output

    """
    epoch_result: EpochResult
    stash: BatchStash
    column_names: List[str] = field(default=None)
    data_point_transform: Callable[[DataPoint], tuple] = field(default=None)
    batch_limit: int = sys.maxsize

    def __post_init__(self):
        if self.column_names is None:
            self.column_names = ('data',)
        if self.data_point_transform is None:
            self.data_point_transform = lambda dp: (str(dp),)

    @property
    def vectorizer_manager_set(self) -> FeatureVectorizerManagerSet:
        return self.stash.vectorizer_manager_set

    def _batch_data_frame(self) -> Iterable[pd.DataFrame]:
        """Return a data from for each batch.

        """
        pred_labs = self.epoch_result.prediction_updates
        transform = self.data_point_transform
        batches = zip(self.epoch_result.batch_ids, pred_labs)
        for i, pred_lab_batch in it.islice(batches, self.batch_limit):
            batch: Batch = self.stash[i]
            batch = self.stash.reconstitute_batch(batch)
            vec = batch.batch_stash.get_label_feature_vectorizer(batch)
            inv_trans = vec.label_encoder.inverse_transform
            preds = inv_trans(pred_lab_batch[0])
            labs = inv_trans(pred_lab_batch[1])
            rows = []
            for dp, lab, pred in zip(batch.data_points, labs, preds):
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
