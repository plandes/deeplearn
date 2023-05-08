"""This creates Pandas dataframes containing predictions.

"""
__author__ = 'Paul Landes'

from typing import Callable, List, Iterable, Any, ClassVar, Dict, Tuple
from dataclasses import dataclass, field
import logging
import sys
import itertools as it
from pathlib import Path
from frozendict import frozendict
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from zensols.persist import persisted
from zensols.datdesc import DataFrameDescriber
from zensols.deeplearn.vectorize import (
    CategoryEncodableFeatureVectorizer,
    FeatureVectorizerManagerSet,
)
from zensols.deeplearn.batch import Batch, BatchStash, DataPoint
from . import (
    ModelResultError, ModelResult, EpochResult, ClassificationMetrics
)

logger = logging.getLogger(__name__)


@dataclass
class PredictionsDataFrameFactory(object):
    """Create a Pandas data frame containing results from a result as output from a
    ``ModelExecutor``.  The data frame contains the feature IDs, labels,
    predictions mapped back to their original value from the feature data item.

    Currently only classification models are supported.

    """
    METRIC_DESCRIPTIONS: ClassVar[Dict[str, str]] = frozendict({
        'wF1': 'weighted F1',
        'wP': 'weighted precision',
        'wR': 'weighted recall',
        'mF1': 'micro F1',
        'mP': 'micro precision',
        'mR': 'micro recall',
        'MF1': 'macro F1',
        'MP': 'macro precision',
        'MR': 'macro recall',
        'correct': 'the number of correct classifications',
        'count': 'the number of data points in the test set',
        'acc': 'accuracy',

        'wF1t': 'weighted F1 on the test set',
        'wPt': 'weighted precision on the test set',
        'wRt': 'weighted recall on the test set',
        'mF1t': 'micro F1 on the test set',
        'mPt': 'micro precision on the test set',
        'mRt': 'micro recall on the test set',
        'MF1t': 'macro F1 on the test set',
        'MPt': 'macro precision on the test set',
        'MRt': 'macro recall on the test set',
        'acct': 'accuracy on the test set',

        'wF1v': 'weighted F1 on the validation set',
        'wPv': 'weighted precision on the validation set',
        'wRv': 'weighted recall on the validation set',
        'mF1v': 'micro F1 on the validation set',
        'mPv': 'micro precision on the validation set',
        'mRv': 'micro recall on the validation set',
        'MF1v': 'macro F1 on the validation set',
        'MPv': 'macro precision on the validation set',
        'MRv': 'macro recall on the validation set',
        'accv': 'accuracy on the validation set',

        'train_occurs': 'the number of data points used to train the model',
        'test_occurs': 'the number of data points used to test the model',
        'validation_occurs': 'the number of data points used to validate the model',

        'label': 'the model class',
        'name': 'the model or result set name',
        'file': 'the directory name of the results',
        'start': 'when the test started',
        'train_duration': 'the time it took to train the model in HH:MM:SS',
        'converged': 'the last epoch with the lowest loss',
        'features': 'the features used in the model'})
    """Dictionary of performance metrics column names to human readable
    descriptions.

    """
    ID_COL: ClassVar[str] = 'id'
    """The data point ID in the generated dataframe in :obj:`dataframe` and
    :obj:`metrics_dataframe`.

    """
    LABEL_COL: ClassVar[str] = 'label'
    """The gold label column in the generated dataframe in :obj:`dataframe` and
    :obj:`metrics_dataframe`.

    """
    PREDICTION_COL: ClassVar[str] = 'pred'
    """The prediction column in the generated dataframe in :obj:`dataframe` and
    :obj:`metrics_dataframe`.

    """
    CORRECT_COL: ClassVar[str] = 'correct'
    """The correct/incorrect indication column in the generated dataframe in
    :obj:`dataframe` and :obj:`metrics_dataframe`.

    """
    METRICS_DF_WEIGHTED_COLUMNS: ClassVar[Tuple[str, ...]] = tuple(
        'wF1 wP wR'.split())
    """Weighed performance metrics columns."""

    METRICS_DF_MICRO_COLUMNS: ClassVar[Tuple[str, ...]] = tuple(
        'mF1 mP mR'.split())
    """Micro performance metrics columns."""

    METRICS_DF_MACRO_COLUMNS: ClassVar[Tuple[str, ...]] = tuple(
        'MF1 MP MR'.split())
    """Macro performance metrics columns."""

    METRICS_DF_COLUMNS: ClassVar[Tuple[str, ...]] = tuple(
        'label wF1 wP wR mF1 mP mR MF1 MP MR correct acc count'.split())
    """
    :see: :obj:`metrics_dataframe`
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
    label_vectorizer_name: str = field(default=None)
    """The name of the vectorizer that encodes the labels, which is used to reverse
    map from integers to their original string nominal values.

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
            row = [dp.id, lab, pred, lab == pred]
            row.extend(transform(dp))
            rows.append(row)
        cols = [self.ID_COL, self.LABEL_COL, self.PREDICTION_COL,
                self.CORRECT_COL]
        cols = cols + list(self.column_names)
        return pd.DataFrame(rows, columns=cols)

    def _calc_len(self, batch: Batch) -> int:
        return len(batch)

    def _narrow_encoder(self, batch: Batch) -> LabelEncoder:
        vec: CategoryEncodableFeatureVectorizer = None
        if self.label_vectorizer_name is None:
            vec = batch.get_label_feature_vectorizer()
            while True:
                if not isinstance(vec, CategoryEncodableFeatureVectorizer) \
                   and hasattr(vec, 'delegate'):
                    vec = vec.delegate
                else:
                    break
        else:
            vms: FeatureVectorizerManagerSet = \
                batch.batch_stash.vectorizer_manager_set
            vec = vms.get_vectorizer(self.label_vectorizer_name)
        if not isinstance(vec, CategoryEncodableFeatureVectorizer):
            raise ModelResultError(
                'Expecting a category feature vectorizer but got: ' +
                f'{vec} ({vec.name if vec else "none"})')
        return vec.label_encoder

    def _batch_dataframe(self, inv_trans: bool) -> Iterable[pd.DataFrame]:
        """Return a data from for each batch.

        """
        epoch_labs: List[np.ndarray] = self.epoch_result.labels
        epoch_preds: List[np.ndarray] = self.epoch_result.predictions
        start = 0
        for bid in it.islice(self.epoch_result.batch_ids, self.batch_limit):
            batch: Batch = self.stash[bid]
            end = start + self._calc_len(batch)
            preds: List[int] = epoch_preds[start:end]
            labs: List[int] = epoch_labs[start:end]
            if inv_trans:
                le: LabelEncoder = self._narrow_encoder(batch)
                inv_trans: Callable = le.inverse_transform
                preds: List[str] = inv_trans(preds)
                labs: List[str] = inv_trans(labs)
            df = self._transform_dataframe(batch, labs, preds)
            df['batch_id'] = bid
            assert len(df) == len(labs)
            start = end
            yield df

    def _create_dataframe(self, inv_trans: bool) -> pd.DataFrame:
        return pd.concat(self._batch_dataframe(inv_trans), ignore_index=True)

    @property
    @persisted('_dataframe')
    def dataframe(self) -> pd.DataFrame:
        """The predictions and labels as a dataframe.  The first columns are generated
        from ``data_point_tranform``, and the remaining columns are:

        - id: the ID of the feature (not batch) data item
        - label: the label given by the feature data item
        - pred: the prediction
        - correct: whether or not the prediction was correct

        """
        return self._create_dataframe(True)

    def _to_metric_row(self, lab: str, mets: ClassificationMetrics) -> \
            List[Any]:
        return [lab, mets.weighted.f1, mets.weighted.precision,
                mets.weighted.recall,
                mets.micro.f1, mets.micro.precision, mets.micro.recall,
                mets.macro.f1, mets.macro.precision, mets.macro.recall,
                mets.n_correct, mets.accuracy, mets.n_outcomes]

    def _add_metric_row(self, le: LabelEncoder, df: pd.DataFrame, ann_id: str,
                        rows: List[Any]):
        lab: str = le.inverse_transform([ann_id])[0]
        data = df[self.LABEL_COL], df[self.PREDICTION_COL]
        mets = ClassificationMetrics(*data, len(data[0]))
        row = self._to_metric_row(lab, mets)
        rows.append(row)

    def metrics_to_series(self, lab: str, mets: ClassificationMetrics) -> \
            pd.Series:
        """Create a single row dataframe from classification metrics."""
        row = self._to_metric_row(lab, mets)
        return pd.Series(row, index=self.METRICS_DF_COLUMNS)

    @property
    def metrics_dataframe(self) -> pd.DataFrame:
        """Performance metrics by comparing the gold label to the predictions.

        """
        rows: List[Any] = []
        df = self._create_dataframe(False)
        dfg = df.groupby(self.LABEL_COL).agg({self.LABEL_COL: 'count'}).\
            rename(columns={self.LABEL_COL: 'count'})
        bids = self.epoch_result.batch_ids
        batch: Batch = self.stash[bids[0]]
        le: LabelEncoder = self._narrow_encoder(batch)
        for ann_id, dfg in df.groupby(self.LABEL_COL):
            try:
                self._add_metric_row(le, dfg, ann_id, rows)
            except ValueError as e:
                logger.error(f'Could not create metrics for {ann_id}: {e}')
        dfr = pd.DataFrame(rows, columns=self.METRICS_DF_COLUMNS)
        dfr = dfr.sort_values(self.LABEL_COL).reset_index(drop=True)
        return dfr

    @property
    def majority_label_metrics(self) -> ClassificationMetrics:
        """Compute metrics of the majority label of the test dataset.

        """
        df: pd.DataFrame = self.dataframe
        le = LabelEncoder()
        gold: np.ndarray = le.fit_transform(df[self.ID_COL].to_list())
        max_id: str = df.groupby(self.ID_COL)[self.ID_COL].agg('count').idxmax()
        majlab: np.ndarray = np.repeat(le.transform([max_id])[0], gold.shape[0])
        return ClassificationMetrics(gold, majlab, gold.shape[0])

    @property
    def metrics_dataframe_describer(self) -> DataFrameDescriber:
        """Get a dataframe describer of metrics (see :obj:`metrics_dataframe`).

        """
        df: pd.DataFrame = self.metrics_dataframe
        meta: Tuple[Tuple[str, str], ...] = \
            tuple(map(lambda c: (c, self.METRIC_DESCRIPTIONS[c]), df.columns))
        return DataFrameDescriber(
            name=self.name,
            df=df,
            desc=f'{self.name.capitalize()} Model Results',
            meta=meta)


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
                self.ID_COL: dp.id,
                self.LABEL_COL: labs[start:end],
                self.PREDICTION_COL: preds[start:end]})
            df[list(self.column_names)] = transform(dp)
            dfs.append(df)
            start = end
        return pd.concat(dfs)
