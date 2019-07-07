import logging
import unittest
from zensols.dltools import CudaConfig

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('zensols.dltools.test')


class TestCuda(unittest.TestCase):
    def test_cuda_config(self):
        conf = CudaConfig()
        self.assertNotEqual(None, conf.info)
        conf.info.write()
