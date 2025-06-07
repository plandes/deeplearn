"""This creates Pandas dataframes containing predictions.

"""
__author__ = 'Paul Landes'

from typing import (
    Dict, Tuple, List, Iterable, Any, Type, Union, ClassVar, Callable
)
from dataclasses import dataclass, field
import logging
import sys
import itertools as it
from pathlib import Path
from frozendict import frozendict
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from zensols.persist import persisted, FileTextUtil
from zensols.datdesc import DataFrameDescriber
from zensols.deeplearn.vectorize import (
    FeatureVectorizer,
    CategoryEncodableFeatureVectorizer,
    FeatureVectorizerManagerSet,
)
from zensols.deeplearn.batch import Batch, BatchStash, DataPoint
from . import (
    ModelResultError, ModelResult, EpochResult,
    Metrics, ClassificationMetrics, MultiLabelClassificationMetrics
)

logger = logging.getLogger(__name__)


@dataclass
class PredictionsDataFrameFactory(object):
    """Create a Pandas :class:`pandas.DataFrame` containing the labels and
    predictions from the :class:`..model.ModelExecutor` test data set output .
    The data frame contains the feature IDs, labels, predictions mapped back to
    their original value from the feature data item.

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
        'correct': 'number of correct classifications',
        'count': 'number of data points in the test set',
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

        'train_occurs': 'the number of data points used to train',
        'test_occurs': 'the number of data points used to test',
        'validation_occurs': 'the number of data points used to validate',

        'label': 'model class',
        'name': 'model or result set name',
        'resid': 'result ID and file name prefix',
        'train_start': 'when the training started',
        'train_end': 'when the trainingstarted',
        'test_start': 'when the testing started',
        'test_end': 'when the testingstarted',
        'converged': 'last epoch with the lowest loss',
        'features': 'features used in the model'})
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

    METRIC_COLUMNS: ClassVar[Tuple[str, ...]] = (
        *METRICS_DF_WEIGHTED_COLUMNS,
        *METRICS_DF_MICRO_COLUMNS,
        *METRICS_DF_MACRO_COLUMNS,
        'acc')
    """Weighted, micro, macro and accuracy metrics columns."""

    METRIC_AVERAGE_TO_COLUMN: ClassVar[Dict[str, str]] = frozendict(
        {'weighted': 'w', 'micro': 'm', 'macro': 'M'})
    """Name to abbreviation average mapping."""

    METRIC_NAME_TO_COLUMN: ClassVar[Dict[str, str]] = frozendict(
        {'f1': 'f1', 'p': 'precision', 'r': 'recall'})
    """Name to abbreviation metric mapping."""

    METRICS_DF_COLUMNS: ClassVar[Tuple[str, ...]] = (
        'label', *METRIC_COLUMNS, 'correct', 'count')
    """
    :see: :obj:`metrics_dataframe`
    """
    TEST_METRIC_COLUMNS: ClassVar[Tuple[str, ...]] = tuple(map(
        lambda c: f'{c}t', METRIC_COLUMNS))
    """Test set performance metric columns."""

    VALIDATION_METRIC_COLUMNS: ClassVar[Tuple[str, ...]] = tuple(map(
        lambda c: f'{c}v', METRIC_COLUMNS))
    """Validation set performance metric columns."""

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
    label/prediction.

    """
    data_point_transform: Callable[[DataPoint], tuple] = field(default=None)
    """A function that returns a tuple, each with an element respective of
    ``column_names`` to be added to the results for each label/prediction; if
    ``None`` (the default), ``str`` used (see the `Iris Jupyter Notebook
    <https://github.com/plandes/deeplearn/blob/master/notebook/iris.ipynb>`_
    example)

    """
    batch_limit: int = sys.maxsize
    """The max number of batches of results to output."""

    epoch_result: EpochResult = field(default=None)
    """The epoch containing the results.  If none given, take it from the test
    results.

    """
    label_vectorizer_name: str = field(default=None)
    """The name of the vectorizer that encodes the labels, which is used to
    reverse map from integers to their original string nominal values.

    """
    metric_metadata: Dict[str, str] = field(default=None)
    """Additional metadata when creating instances of
    :class:`~zensols.datdesc.desc.DataFrameDescriber` in addition to
    :obj:`METRIC_DESCRIPTIONS`.

    """
    name: str = field(default=None)
    """The name of the results or the result ID (``res_id``).  If not provided,
    it is taken from :class:`.ModelResult`.

    """
    def __post_init__(self):
        if self.column_names is None:
            self.column_names = ('data',)
        if self.data_point_transform is None:
            self.data_point_transform = lambda dp: (str(dp),)
        if self.epoch_result is None:
            self.epoch_result = self.result.test.results[0]
        if self.name is None:
            self.name = self.result.name

    @classmethod
    def metrics_to_describer(cls: Type, metrics: Metrics) -> DataFrameDescriber:
        """Create a dataframe describer using a metrics instance with standard
        column naming and metadata.  Use :obj:`METRIC_AVERAGE_TO_COLUMN` and
        :obj:`METRIC_NAME_TO_COLUMN` to create a single dataframe row of
        performance metrics.

        """
        mets: Dict[str, Any] = metrics.asdict()
        count: int = metrics.n_outcomes \
            if hasattr(metrics, 'n_outcomes') else len(metrics)
        res: List[Tuple[str, Any]] = [('count', count)]
        ave_name: str
        ave_col: str
        for ave_name, ave_col in cls.METRIC_AVERAGE_TO_COLUMN.items():
            ave: Dict[str, float] = mets.pop(ave_name, None)
            if ave is not None:
                for mcol, mname in cls.METRIC_NAME_TO_COLUMN.items():
                    val: Any = ave.get(mname)
                    col: str = f'{ave_col}{mcol.upper()}'
                    res.append((col, val))
        return DataFrameDescriber(
            name='metrics',
            desc='performance metrics',
            df=pd.DataFrame(data=[tuple(map(lambda t: t[1], res))],
                            columns=tuple(map(lambda t: t[0], res))),
            meta=tuple(map(
                lambda t: (t[0], cls.METRIC_DESCRIPTIONS.get(t[0])), res)))

    def _assert_label_pred_batch_size(self, batch: Batch, labs: List[str],
                                      preds: List[str], compare_batch: bool):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'data points: {len(batch.data_points)}, ' +
                         f'labels: {len(labs)}, predictions: {len(preds)}')
        if len(labs) != len(preds):
            raise ModelResultError(f'label ({len(labs)}) and prediciton ' +
                                   f'({len(preds)}) counts do not match')
        if compare_batch and (len(labs) != len(batch.data_points)):
            msg: str = (f'label ({len(labs)}) and batch size ' +
                        f'({len(batch.data_points)}) counts do not match')
            if logger.isEnabledFor(logging.DEBUG):
                for i, dp in enumerate(batch.data_points):
                    lab = labs[i] if len(labs) < i else None
                    pred = labs[preds] if len(preds) < i else None
                    logger.debug(f'{dp}: lab={lab}, pred={pred}')
            logger.error(msg)
            batch.write_to_log(logger, logging.ERROR)
            raise ModelResultError(msg)

    def _transform_dataframe(self, batch: Batch, labs: List[str],
                             preds: List[str]):
        transform: Callable = self.data_point_transform
        rows: List[Any] = []
        self._assert_label_pred_batch_size(batch, labs, preds, True)
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

    def _narrow_vectorizer(self, batch: Batch) -> FeatureVectorizer:
        vec: FeatureVectorizer = None
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
        return vec

    def _narrow_encoder(self, batch: Batch) -> \
            Union[LabelEncoder, MultiLabelBinarizer]:
        vec: CategoryEncodableFeatureVectorizer = self._narrow_vectorizer(batch)
        if not isinstance(vec, CategoryEncodableFeatureVectorizer):
            raise ModelResultError(
                'Expecting a category feature vectorizer but got: ' +
                f'{vec} ({vec.name if vec else "none"})')
        return vec.label_encoder

    def _narrow_epoch(self, labs: np.ndarray, preds: np.ndarray,
                      start: int, end: int):
        labs = labs[start:end]
        preds = preds[start:end]
        return labs, preds

    def _batch_dataframe(self, inv_trans: bool) -> Iterable[pd.DataFrame]:
        """Return a dataframe from for each batch."""
        epoch_labs: np.ndarray = self.epoch_result.labels
        epoch_preds: np.ndarray = self.epoch_result.predictions
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('create batch data frames for ' +
                         f'{len(epoch_labs)} labels, ' +
                         f'{len(epoch_preds)} predictions' +
                         f'limit: {self.batch_limit}')
        assert len(epoch_labs) == len(epoch_preds)
        start: int = 0
        for bid in it.islice(self.epoch_result.batch_ids, self.batch_limit):
            batch: Batch = self.stash[bid]
            end: int = start + self._calc_len(batch)
            labs, preds = self._narrow_epoch(epoch_labs, epoch_preds, start, end)
            if inv_trans:
                le: Union[LabelEncoder, MultiLabelBinarizer] = \
                    self._narrow_encoder(batch)
                inv_trans: Callable = le.inverse_transform
                labs: List[str] = inv_trans(labs)
                preds: List[str] = inv_trans(preds)
            df = self._transform_dataframe(batch, labs, preds)
            df['batch_id'] = bid
            assert len(df) == len(labs)
            start = end
            yield df

    def _create_dataframe(self, inv_trans: bool) -> pd.DataFrame:
        return pd.concat(self._batch_dataframe(inv_trans), ignore_index=True)

    def _create_data_frame_describer(self, df: pd.DataFrame,
                                     desc: str = 'Run Model Results',
                                     metric_metadata: Dict[str, str] = None) \
            -> DataFrameDescriber:
        mdesc: Dict[str, str] = dict(self.METRIC_DESCRIPTIONS)
        if self.metric_metadata is not None:
            mdesc.update(self.metric_metadata)
        if metric_metadata is not None:
            mdesc.update(metric_metadata)
        meta: Tuple[Tuple[str, str], ...] = tuple(map(
            lambda c: (c, mdesc[c]), df.columns))
        return DataFrameDescriber(
            name=FileTextUtil.normalize_text(self.name),
            df=df,
            desc=f'{self.name.capitalize()} {desc}',
            meta=meta)

    @property
    @persisted('_dataframe')
    def dataframe(self) -> pd.DataFrame:
        """The predictions and labels as a dataframe.  The first columns are
        generated from ``data_point_tranform``, and the remaining columns are:

        - id: the ID of the feature (not batch) data item
        - label: the label given by the feature data item
        - pred: the prediction
        - correct: whether or not the prediction was correct

        """
        return self._create_dataframe(True)

    @property
    def dataframe_describer(self) -> DataFrameDescriber:
        """Same as :obj:`dataframe`, but return the data with metadata."""
        metric_metadata: Dict[str, str] = {
            'id': 'unique data point identifier',
            'label': 'gold label',
            'pred': 'predicted label',
            'correct': 'whether the prediction was correct',
            'data': 'data used for prediction',
            'batch_id': 'batch unique identifier',
        }
        return self._create_data_frame_describer(
            df=self.dataframe,
            metric_metadata=metric_metadata)

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

    def _get_metrics_dataframe(self) -> pd.DataFrame:
        """Performance metrics by comparing the gold label to the predictions.

        """
        rows: List[Any] = []
        df = self._create_dataframe(False)
        dfg = df.groupby(self.LABEL_COL).agg({self.LABEL_COL: 'count'}).\
            rename(columns={self.LABEL_COL: 'count'})
        bids = self.epoch_result.batch_ids
        batch: Batch = self.stash[bids[0]]
        le: Union[LabelEncoder, MultiLabelBinarizer] = \
            self._narrow_encoder(batch)
        for ann_id, dfg in df.groupby(self.LABEL_COL):
            try:
                self._add_metric_row(le, dfg, ann_id, rows)
            except ValueError as e:
                logger.error(f'Could not create metrics for {ann_id}: {e}')
        dfr = pd.DataFrame(rows, columns=self.METRICS_DF_COLUMNS)
        dfr = dfr.sort_values(self.LABEL_COL).reset_index(drop=True)
        return dfr

    @property
    def metrics_dataframe_describer(self) -> DataFrameDescriber:
        """Get a dataframe describer of metrics (see :obj:`metrics_dataframe`).

        """
        df: pd.DataFrame = self._get_metrics_dataframe()
        return self._create_data_frame_describer(df)

    @property
    def majority_label_metrics_describer(self) -> DataFrameDescriber:
        """Compute metrics of the majority label of the test dataset.

        """
        df: pd.DataFrame = self.dataframe
        le = LabelEncoder()
        gold: np.ndarray = le.fit_transform(df[self.ID_COL].to_list())
        max_id: str = df.groupby(self.ID_COL)[self.ID_COL].agg('count').idxmax()
        majlab: np.ndarray = np.repeat(le.transform([max_id])[0], gold.shape[0])
        mets = ClassificationMetrics(gold, majlab, gold.shape[0])
        return self._create_data_frame_describer(
            df=(self.metrics_to_series(None, mets).to_frame().
                T.drop(columns='label')),
            desc='Majority Label')


@dataclass
class MultiLabelPredictionsDataFrameFactory(PredictionsDataFrameFactory):
    """Like the super class but create predictions multilabel on sentences
    and documents.

    """
    def _narrow_encoder(self, batch: Batch) -> \
            Union[LabelEncoder, MultiLabelBinarizer]:
        from zensols.deeplearn.vectorize import \
            NominalMultiLabelEncodedEncodableFeatureVectorizer as NomVectorizer
        vec: FeatureVectorizer = self._narrow_vectorizer(batch)
        if not isinstance(vec, NomVectorizer):
            raise ModelResultError(
                'Expecting a category feature vectorizer but got: ' +
                f'{vec} ({vec.name if vec else "none"})')
        return vec.label_binarizer

    def _narrow_epoch(self, labs: np.ndarray, preds: np.ndarray,
                      start: int, end: int):
        from zensols.deeplearn.result import ResultContext
        ctx: ResultContext = self.epoch_result.context
        labs, preds = MultiLabelClassificationMetrics.\
            reshape_labels_predictions(labs, preds, ctx)
        labs = labs[start:end]
        preds = preds[start:end]
        return labs, preds

    def _get_metrics_dataframe(self) -> pd.DataFrame:
        mets: MultiLabelClassificationMetrics = self.epoch_result.metrics
        df: pd.DataFrame = mets.dataframes['label']
        df.insert(0, 'label', df.index)
        df = df.reset_index(drop=True)
        df = df['label f1 precision recall count'.split()]
        # micro-average is used for labels because "it corresponds to
        # accuracy otherwise and would be the same for all metrics"
        # https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.classification_report.html
        df = df.rename(columns={
            'precision': 'mP',
            'recall': 'mR',
            'f1': 'mF1'})
        return df


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
        self._assert_label_pred_batch_size(batch, labs, preds, False)
        for dp in batch.data_points:
            end: int = start + len(dp)
            df = pd.DataFrame({
                self.ID_COL: dp.id,
                self.LABEL_COL: labs[start:end],
                self.PREDICTION_COL: preds[start:end]})
            dp_data: Tuple[Tuple[Any, ...]] = transform(dp)
            if len(df.index) != len(dp_data):
                raise ModelResultError(
                    'Size of result does not match transformed data point:' +
                    f'<{tuple(df.index)}> != <{dp_data}> for: <{dp}>')
            df[list(self.column_names)] = dp_data
            dfs.append(df)
            start = end
        return pd.concat(dfs)
