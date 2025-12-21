"""Multi processing with torch.

"""
__author__ = 'Paul Landes'

from typing import Callable, List, Iterable, Any
from dataclasses import dataclass
import logging
from torch.multiprocessing import Pool as TorchPool
from zensols.util.time import time
from zensols.config import Configurable
from zensols.persist import Stash
from zensols.multi import (
    MultiProcessStash, MultiProcessFactoryStash, MultiProcessRobustStash
)

logger = logging.getLogger(__name__)


@dataclass
class TorchMultiProcessStash(MultiProcessStash):
    """A multiprocessing stash that interacts with PyTorch in a way that it can
    access the GPU(s) in forked subprocesses using the :mod:`multiprocessing`
    library.

    :see: :mod:`torch.multiprocessing`

    :see: :meth:`zensols.deeplearn.TorchConfig.init`

    """
    def _invoke_pool(self, pool: TorchPool, fn: Callable, data: iter) -> \
            List[int]:
        """Invoke on a torch pool (rather than a :class:`multiprocessing.Pool`).

        """
        if pool is None:
            return tuple(map(fn, data))
        else:
            return pool.map(fn, data)

    def _invoke_work(self, workers: int, chunk_size: int,
                     data: Iterable[Any]) -> int:
        fn = self.__class__._process_work
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'{self.name}: spawning work with ' +
                        f'chunk size {chunk_size} across {workers} workers')
        if workers == 1:
            with time('processed chunks'):
                cnt = self._invoke_pool(None, fn, data)
        else:
            with TorchPool(workers) as p:
                if logger.isEnabledFor(logging.INFO):
                    logger.info(f'using torch multiproc pool: {p}')
                with time('processed chunks'):
                    cnt = self._invoke_pool(p, fn, data)
        return cnt


@dataclass(init=False)
class TorchMultiProcessFactoryStash(MultiProcessFactoryStash):
    """Like :class:`~zensols.multi.factory.MultiProcessFactoryStash` but uses
    PyTorch processing methods from :class:`.TorchMultiProcessStash`.

    """
    def __init__(self, config: Configurable, name: str, factory: Stash,
                 enable_preemptive: bool = False, **kwargs):
        super().__init__(config, name, factory, enable_preemptive, **kwargs)

    def _invoke_pool(self, pool: TorchPool, fn: Callable, data: iter) -> \
            List[int]:
        return TorchMultiProcessStash._invoke_pool(self, pool, fn, data)

    def _invoke_work(self, workers: int, chunk_size: int,
                     data: Iterable[Any]) -> int:
        return TorchMultiProcessStash._invoke_work(
            self, workers, chunk_size, data)


@dataclass(init=False)
class TorchMultiProcessRobustStash(MultiProcessRobustStash):
    """Like :class:`~zensols.multi.factory.MultiProcessRobustStash` but uses
    PyTorch processing methods from :class:`.TorchMultiProcessStash`.

    """
    def __init__(self, config: Configurable, name: str,
                 protect_work: bool = True, **kwargs):
        super().__init__(config, name, protect_work, **kwargs)

    def _invoke_pool(self, pool: TorchPool, fn: Callable, data: iter) -> \
            List[int]:
        return TorchMultiProcessStash._invoke_pool(self, pool, fn, data)

    def _invoke_work(self, workers: int, chunk_size: int,
                     data: Iterable[Any]) -> int:
        return TorchMultiProcessStash._invoke_work(
            self, workers, chunk_size, data)
