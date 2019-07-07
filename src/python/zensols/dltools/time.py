"""Peformance measure convenience utils.

"""
__author__ = 'Paul Landes'

import logging
import time as tm

time_logger = logging.getLogger(__name__)


class time(object):
    """Used in a ``with`` scope that executes the body and logs the elapsed time.

    Example:

        with time("test1", logger):
            tm.sleep(1)

    """
    def __init__(self, msg, logger=time_logger, level=logging.INFO):
        self.msg = msg
        self.logger = logger
        self.level = level

    def __enter__(self):
        self.t0 = tm.time()

    def __exit__(self, type, value, traceback):
        elapse = tm.time() - self.t0
        self.logger.log(self.level, f'{self.msg} finished in {elapse:.1f}s')
