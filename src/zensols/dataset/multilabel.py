"""A multilabel stratifier.

"""
__author__ = 'Paul Landes'

from typing import Any, Tuple, List, Dict, Iterable, Set, Union, Callable
from dataclasses import dataclass, field
import logging
from collections import defaultdict
import sys
from io import TextIOBase
import math
import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from zensols.persist import persisted
from . import StratifiedStashSplitKeyContainer

logger = logging.getLogger(__name__)


@dataclass
class MultiLabelStratifierSplitKeyContainer(StratifiedStashSplitKeyContainer):
    """Creates stratified two-way splits between token-level annotated feature
    sentences.

    """
    split_preference: Tuple[str, ...] = field(default=None)
    """The list of splits to give preference by moving data that has no
    instances.  For exaple, ``('test', 'validation')`` would move data points
    from ``validation`` to ``test`` for labels that have no occurances in
    ``test``.

    """
    move_portion: float = field(default=0.5)
    """The portion of data points per label to move based on
    :obj:`split_preference`.

    """
    min_source_occurances: int = field(default=0)
    """The minimum number of occurances for a label to trigger the key move
    described in :obj:`split_prefernce`.

    """
    @property
    @persisted('_count_dataframe')
    def count_dataframe(self) -> pd.DataFrame:
        """A dataframe with the counts of each label as columns."""
        rows: List[pd.Series] = []
        item: Any
        for k, item in self.stash:
            labels: Dict[str, int] = defaultdict(lambda: 0)
            for lb in getattr(item, self.partition_attr):
                labels[lb] += 1
            rows.append(pd.Series(labels, name=k))
        return pd.DataFrame(rows).fillna(0).astype(int)

    def _split(self, counts: pd.DataFrame, test_size: Union[int, float]) -> \
            Tuple[np.ndarray, np.ndarray]:
        """Return the dataset spilts as indexes into :obj:`sents`.

        :param test_size: the portion of the first element of the returned value

        :return: the tuple ``(train, test)`` with each element of the tuple a
                 :class:`numpy.ndarray` index array

        """
        msss = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=0)
        return next(msss.split(counts.index, counts))

    def _rebalance_zeros(self, splits: Dict[str, Set[str]],
                         split_preference: str):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'rebalancing for: {split_preference}')
        to_move: List[str] = []
        dst_split_name: str = split_preference[0]
        src_split_names: List[str] = split_preference[1:]
        dst_split = splits[dst_split_name]
        df: pd.DataFrame = self.count_dataframe
        df = df[df.index.isin(dst_split)]
        col: str
        for col in df.columns:
            lb_sum: int = df[col].sum()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'col {col} count: {lb_sum}')
            if lb_sum <= self.min_source_occurances:
                to_move.append(col)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'to move: {to_move} in {src_split_names}')
        src_split_name: str
        for src_split_name in src_split_names:
            src_split: Set[str] = splits[src_split_name]
            dfs: pd.DataFrame = self.count_dataframe
            dfs = dfs[dfs.index.isin(src_split)]
            for col in to_move:
                lb_sum: int = dfs[col].sum()
                n_take: int = math.ceil(self.move_portion * lb_sum)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'{col}: {src_split_name} -> ' +
                                 f'{dst_split_name} ({lb_sum}/{n_take})')
                for k in dfs[dfs[col] > 0].head(n_take).index.to_list():
                    # prevent it from being moved twice
                    if k in src_split:
                        if logger.isEnabledFor(logging.TRACE):
                            logger.trace(f'moving key {k} to {dst_split_name}')
                        src_split.remove(k)
                        dst_split.add(k)

    def _create_splits(self) -> Dict[str, Tuple[str, ...]]:
        rebalance: bool = self.split_preference is not None
        agg_fn: Callable = set if rebalance else tuple
        slen: int = len(self.stash)
        dfc: pd.DataFrame = self.count_dataframe
        assert len(dfc) == slen
        splits: Dict[str, Tuple[str, ...]] = {}
        dist: List[Tuple[str, float]] = sorted(
            self.distribution.items(),
            key=lambda t: t[1],
            reverse=False)
        for split_name, por in dist[:-1]:
            count: int = int(slen * por)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'{split_name}: portion: {por}, count: {count}')
            left, split = self._split(dfc, count)
            splits[split_name] = agg_fn(dfc.index[split])
            dfc = dfc.iloc[left]
        splits[dist[-1][0]] = agg_fn(dfc.index)
        assert slen == sum(map(lambda ks: len(ks), splits.values()))
        if rebalance:
            self._rebalance_zeros(splits, self.split_preference)
            splits = dict(map(lambda t: (t[0], tuple(t[1])), splits.items()))
        return splits

    def _get_stratified_split_labels(self) -> pd.DataFrame:
        kbs: Dict[str, Tuple[str, ...]] = self.keys_by_split
        rows: List[Tuple[Any, ...]] = []
        split_name: str
        for split_name in sorted(kbs.keys()):
            k: str
            for k in kbs[split_name]:
                item: Any = self.stash[k]
                lb: str
                for lb in getattr(item, self.partition_attr):
                    rows.append((split_name, k, lb))
        return pd.DataFrame(rows, columns='split_name id label'.split())

    def _get_stratified_split_portions(self) -> pd.DataFrame:
        df: pd.DataFrame = self.stratified_split_labels
        df_splits: List[pd.DataFrame] = []
        labels: Set[str] = set(df['label'].drop_duplicates())
        split_name: str
        dfs: pd.DataFrame
        for split_name, dfs in df.groupby('split_name'):
            dfc: pd.Series = dfs.groupby('label')['label'].count().\
                to_frame().rename(columns={'label': 'count'})
            dfc['portion'] = dfc['count'] / dfc['count'].sum()
            dfc.insert(0, 'split', split_name)
            dfc.insert(0, 'label', dfc.index)
            for lb in labels - set(dfc['label']):
                dfc.loc[len(df) + 1] = [lb, split_name, 0, 0]
            dfc = dfc.reset_index(drop=True).sort_values('label')
            df_splits.append(dfc)
        return pd.concat(df_splits)

    @property
    @persisted('_stratified_split_portions')
    def stratified_split_portions(self) -> pd.DataFrame:
        return self._get_stratified_split_portions()

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        def map_row(r: pd.Series):
            return f"{r['label']}: {r['count']} ({round(r['portion'] * 100)}%)"

        super().write(depth, writer)
        if self.stratified_write:
            df: pd.DataFrame = self.stratified_split_portions
            self._write_line('splits:', depth, writer)
            for split_name, dfc in df.groupby('split'):
                self._write_line(f'{split_name}:', depth + 1, writer)
                pors: Iterable[str] = \
                    map(map_row, map(lambda t: t[1], dfc.iterrows()))
                self._write_iterable(pors, depth + 2, writer)
