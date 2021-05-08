"""Implementations (some abstract) of split key containers.

"""
__author__ = 'Paul Landes'

from typing import Dict, Tuple
from dataclasses import dataclass, field
from abc import abstractmethod, ABCMeta
import sys
import logging
from io import TextIOBase
from pathlib import Path
import parse
import random
from zensols.config import Writable
from zensols.persist import Primeable, persisted, Stash, PersistableContainer
from zensols.dataset import SplitKeyContainer

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
    """The file name pattern to use for the keys file :obj:`key_path` on the file
    system, each file is named after the key split.  For example, if
    ``{name}.dat`` is used, ``train.dat`` will be a file with the ordered keys.

    """

    def __post_init__(self):
        super().__init__()

    def prime(self):
        self._get_keys_by_split()

    @abstractmethod
    def _create_splits(self) -> Dict[str, Tuple[str]]:
        """Create the key splits using keys as the split name (i.e. ``train``) and the
        values as a list of the keys for the corresponding split.

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

    @persisted('_split_names_pw')
    def _get_split_names(self) -> Tuple[str]:
        return frozenset(self.split_distribution.keys())

    @persisted('_get_keys_by_split_pw')
    def _get_keys_by_split(self) -> Dict[str, Tuple[str]]:
        if not self.key_path.exists():
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'creating key splits in {self.key_path}')
            self._create_splits_and_write()
        return self._read_splits()

    def clear(self):
        logger.debug('clearing split stash')
        self.key_stash.clear()

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
    """A default implementation of :class:`.AbstractSplitKeyContainer` that uses a
    delegate stash for source of the keys.

    """
    stash: Stash = field()
    """The delegate stash from whence to get the keys to store."""

    distribution: Dict[str, float] = field(default=None)
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
        if err > errm:
            raise ValueError('distriubtion must add to 1: ' +
                             f'{self.distribution} (err={err} > errm)')

    def _create_splits(self) -> Dict[str, Tuple[str]]:
        if self.distribution is None:
            raise ValueError('must either provide `distribution` or ' +
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
        for k, v in by_name.items():
            print(k, len(v), len(v)/klen)
        assert sum(map(len, by_name.values())) == klen
        return by_name
