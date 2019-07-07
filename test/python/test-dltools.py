import logging
import unittest
from zensols.dltools import CudaConfig

logger = logging.getLogger(__name__)


class TestCuda(unittest.TestCase):
    def test_cuda_config(self):
        conf = CudaConfig()
        self.assertNotEqual(None, conf.info)
        conf.info.write()
