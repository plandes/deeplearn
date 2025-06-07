"""Implementations (some abstract) of split key containers.

"""
__author__ = 'Paul Landes'

from typing import Dict, Tuple, Sequence, Set, List, ClassVar
from dataclasses import dataclass, field
from abc import abstractmethod, ABCMeta
import sys
import logging
import collections
from functools import reduce
import math
from io import TextIOBase
from pathlib import Path
import shutil
import parse
import random
import pandas as pd
from zensols.config import Writable
from zensols.persist import (
    Primeable, persisted, PersistedWork, Stash, PersistableContainer
)
from zensols.dataset import SplitKeyContainer
from . import DatasetError

logger = logging.getLogger(__name__)


@dataclass
class AbstractSplitKeyContainer(PersistableContainer, SplitKeyContainer,
                                Primeable, Writable, metaclass=ABCMeta):
    """A default implementation of a :class:`.SplitKeyContainer`.  This
    implementation keeps the order of the keys consistent as well, which is
    stored at the path given in :obj:`key_path`.  Once the keys are generated
    for the first time, they will persist on the file system.

    This abstract class requires an implementation of :meth:`_create_splits`.

    .. document private functions
    .. automethod:: _create_splits

    """
    key_path: Path = field()
    """The directory to store the split keys."""

    pattern: str = field()
    """The file name pattern to use for the keys file :obj:`key_path` on the
    file system, each file is named after the key split.  For example, if
    ``{name}.dat`` is used, ``train.dat`` will be a file with the ordered keys.

    """
    def __post_init__(self):
        super().__init__()

    def prime(self):
        self._get_keys_by_split()

    @abstractmethod
    def _create_splits(self) -> Dict[str, Tuple[str, ...]]:
        """Create the key splits using keys as the split name (i.e. ``train``)
        and the values as a list of the keys for the corresponding split.

        """
        pass

    def _create_splits_and_write(self):
        """Write the keys in order to the file system.

        """
        self.key_path.mkdir(parents=True, exist_ok=True)
        for name, keys in self._create_splits().items():
            fname = self.pattern.format(**{'name': name})
            key_path = self.key_path / fname
            with open(key_path, 'w') as f:
                for k in keys:
                    f.write(k + '\n')

    def _read_splits(self):
        """Read the keys in order from the file system.

        """
        by_name = {}
        for path in self.key_path.iterdir():
            p = parse.parse(self.pattern, path.name)
            if p is not None:
                p = p.named
                if 'name' in p:
                    with open(path) as f:
                        by_name[p['name']] = tuple(
                            map(lambda ln: ln.strip(), f.readlines()))
        return by_name

    @persisted('_get_keys_by_split_pw')
    def _get_keys_by_split(self) -> Dict[str, Tuple[str, ...]]:
        if not self.key_path.exists():
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'creating key splits in {self.key_path}')
            self._create_splits_and_write()
        return self._read_splits()

    def clear(self):
        logger.debug('clearing split stash')
        if self.key_path.is_dir():
            logger.debug('removing key path: {self.key_path}')
            shutil.rmtree(self.key_path)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        by_name = self.counts_by_key
        total = sum(by_name.values())
        self._write_line('key splits:', depth, writer)
        for name, cnt in by_name.items():
            self._write_line(f'{name}: {cnt} ({cnt/total*100:.1f}%)',
                             depth + 1, writer)
        self._write_line(f'total: {total}', depth, writer)


@dataclass
class StashSplitKeyContainer(AbstractSplitKeyContainer):
    """A default implementation of :class:`.AbstractSplitKeyContainer` that uses
    a delegate stash for source of the keys.

    """
    stash: Stash = field()
    """The delegate stash from where to get the keys to store."""

    distribution: Dict[str, float] = field(
        default_factory=lambda: {'train': 0.8, 'validate': 0.1, 'test': 0.1})
    """The distribution as a percent across all key splits.  The distribution
    values must add to 1.

    """
    shuffle: bool = field(default=True)
    """If ``True``, shuffle the keys when creating the key splits.

    """
    def __post_init__(self):
        super().__post_init__()
        sm = float(sum(self.distribution.values()))
        err, errm = (1. - sm), 1e-1
        if sm < 0 or err > errm:
            raise DatasetError('Distriubtion must add to 1: ' +
                               f'{self.distribution} (err={err} > errm)')

    def prime(self):
        super().prime()
        if isinstance(self.stash, Primeable):
            self.stash.prime()

    @persisted('_split_names_pw')
    def _get_split_names(self) -> Tuple[str, ...]:
        return frozenset(self.distribution.keys())

    def _create_splits(self) -> Dict[str, Tuple[str, ...]]:
        if self.distribution is None:
            raise DatasetError('Must either provide `distribution` or ' +
                               'implement `_create_splits`')
        by_name = {}
        keys = list(self.stash.keys())
        if self.shuffle:
            random.shuffle(keys)
        klen = len(keys)
        dists = tuple(self.distribution.items())
        if len(dists) > 1:
            dists, last = dists[:-1], dists[-1]
        else:
            dists, last = (), dists[0]
        start = 0
        end = len(dists)
        for name, dist in dists:
            end = start + int((klen * dist))
            by_name[name] = tuple(keys[start:end])
            start = end
        by_name[last[0]] = keys[start:]
        assert sum(map(len, by_name.values())) == klen
        return by_name


@dataclass
class StratifiedStashSplitKeyContainer(StashSplitKeyContainer):
    """Like :class:`.StashSplitKeyContainer` but data is stratified by a label
    (:obj:`partition_attr`) across each split.

    """
    partition_attr: str = field(default=None)
    """The label used to partition the strata across each split"""

    stratified_write: bool = field(default=True)
    """Whether or not to include the stratified counts when writing with
    :meth:`write`.

    """
    split_labels_path: Path = field(default=None)
    """If provided, the path is a pickled cache of
    :obj:`stratified_count_dataframe`.

    """
    def __post_init__(self):
        super().__post_init__()
        if self.partition_attr is None:
            raise DatasetError("Missing 'partition_attr' field")
        dfpath = self.split_labels_path
        if dfpath is None:
            dfpath = '_strat_split_labels'
        self._strat_split_labels = PersistedWork(dfpath, self, mkdir=True)

    def _create_splits(self) -> Dict[str, Tuple[str, ...]]:
        dist_keys: Sequence[str] = self.distribution.keys()
        dist_last: str = next(iter(dist_keys))
        dists: Set[str] = set(dist_keys) - {dist_last}
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'spliting dataset n={len(dist_keys)}')
        rows = []
        for k, v in self.stash.items():
            rows.append((k, getattr(v, self.partition_attr)))
        df = pd.DataFrame(rows, columns=['key', self.partition_attr])
        lab_splits: Dict[str, Set[str]] = collections.defaultdict(set)
        for lab, dfg in df.groupby(self.partition_attr):
            splits = {}
            keys: List[str] = dfg['key'].to_list()
            if self.shuffle:
                random.shuffle(keys)
            count = len(keys)
            for dist in dists:
                prop = self.distribution[dist]
                n_samples = math.ceil(float(count) * prop)
                samp = set(keys[:n_samples])
                splits[dist] = samp
                lab_splits[dist].update(samp)
                keys = keys[n_samples:]
            samp = set(keys)
            splits[dist_last] = samp
            lab_splits[dist_last].update(samp)
        assert sum(map(len, lab_splits.values())) == len(df)
        assert reduce(lambda a, b: a | b, lab_splits.values()) == \
            set(df['key'].tolist())
        shuf_splits = {}
        for lab, keys in lab_splits.items():
            if self.shuffle:
                keys = list(keys)
                random.shuffle(keys)
            shuf_splits[lab] = tuple(keys)
        return shuf_splits

    def _count_proportions_by_split(self) -> Dict[str, Dict[str, str]]:
        lab_counts = {}
        kbs = self.keys_by_split
        for split_name in sorted(kbs.keys()):
            keys = kbs[split_name]
            counts = collections.defaultdict(lambda: 0)
            for k in keys:
                item = self.stash[k]
                lab = getattr(item, self.partition_attr)
                counts[lab] += 1
            lab_counts[split_name] = counts
        return lab_counts

    def _get_stratified_split_labels(self) -> pd.DataFrame:
        kbs = self.keys_by_split
        rows = []
        for split_name in sorted(kbs.keys()):
            keys = kbs[split_name]
            for k in keys:
                item = self.stash[k]
                lab = getattr(item, self.partition_attr)
                rows.append((split_name, k, lab))
        return pd.DataFrame(rows, columns='split_name id label'.split())

    @property
    @persisted('_strat_split_labels')
    def stratified_split_labels(self) -> pd.DataFrame:
        """A dataframe with all keys, their respective labels and split.

        """
        return self._get_stratified_split_labels()

    def clear(self):
        super().clear()
        self._strat_split_labels.clear()

    @property
    def stratified_count_dataframe(self) -> pd.DataFrame:
        """A count summarization of :obj:`stratified_split_labels`."""
        df: pd.DataFrame = self.stratified_split_labels
        df = df.groupby('split_name label'.split()).size().\
            reset_index(name='count')
        df['proportion'] = df['count'] / df['count'].sum()
        df = df.sort_values('split_name label'.split()).reset_index(drop=True)
        return df

    def _fmt_prop_by_split(self) -> Dict[str, Dict[str, str]]:
        df = self.stratified_count_dataframe
        tot = df['count'].sum()
        dsets: Dict[str, Dict[str, str]] = collections.OrderedDict()
        for split_name, dfg in df.groupby('split_name'):
            dfg['fmt'] = df['count'].apply(lambda x: f'{x} ({x/tot*100:.2f}%)')
            dsets[split_name] = dict(dfg[['label', 'fmt']].values)
        return dsets

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        if self.stratified_write:
            lab_counts: Dict[str, Dict[str, str]] = self._fmt_prop_by_split()
            self._write_line('dataset:', depth, writer)
            self._write_dict(lab_counts, depth + 1, writer)
            self._write_line(f'Total: {len(self.stash)}', depth + 1, writer)
        else:
            super().write(depth, writer)


@dataclass
class StratifiedCrossFoldSplitKeyContainer(StratifiedStashSplitKeyContainer):
    """Like :class:`.StratifiedStashSplitKeyContainer`, but create splits used
    for cross-fold validation when batching.  This creates a new dataset for
    each fold by settings :obj:`distribution`.

    """
    FOLD_FORMAT: ClassVar[str] = 'fold-{fold_ix}-{iter_ix}'
    """The format used for naming results in
    :class:`~zensols.deeplearn.model.exector.ModelExecutor`."""

    n_folds: int = field(default=None)
    """The number of folds across"""

    def __post_init__(self):
        if self.n_folds is None:
            raise DatasetError("Number of folds ('n_folds') must be configured")
        fold_por: float = 1. / self.n_folds
        self.distribution = dict(map(
            lambda nf: (f'fold-{nf}', fold_por),
            range(self.n_folds)))
        super().__post_init__()

    def get_by_fold(self, fold: int) -> Stash:
        split_name: str = f'fold-{fold}'
        return self[split_name]
