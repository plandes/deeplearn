"""Initialize the deep learning framework.

"""
__author__ = 'Paul Landes'

from typing import Type, Dict, Any
import logging
from torch.multiprocessing import set_start_method

logger = logging.getLogger(__name__)

class TorchInitializerX(object):
    """Initialize the PyTorch framework.  This includes:

      * Configuration of PyTorch multiprocessing so subprocesses can access the
        GPU, and
      * Setting the random seed state.

    The needs to be initialized at the very beginning of your program.

    Example::
        def main():
            from zensols.deeplearn.init import TorchInitializer
            TorchInitializer.init()

    :see: :mod:`torch.multiprocessing`

    :see: :meth:`.TorchConfig.set_random_seed`

    """
    INITIALIZED = False

    @classmethod
    def init(cls: Type, spawn_multiproc: str = 'spawn',
             seed_kwargs: Dict[str, Any] = {}):
        """Tell PyTorch how to fork processes that can access the GPU.

        """
        print('INIT METH')
        if not cls.INITIALIZED:
            try:
                if logger.isEnabledFor(logging.INFO):
                    logger.info('invoking pool with torch spawn method')
                if spawn_multiproc:
                    set_start_method('spawn')
                else:
                    set_start_method('forkserver', force=True)
                cls.INITIALIZED = True
            except RuntimeError as e:
                logger.warning(f'could not invoke spawn on pool: {e}')
        # from .torchconfig import TorchConfig
        # TorchConfig.set_random_seed(**seed_kwargs)
