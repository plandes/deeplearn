import logging
import unittest
from zensols.dltools import CudaConfig

logger = logging.getLogger(__name__)


class TestCuda(unittest.TestCase):
    def test_cuda_config(self):
        conf = CudaConfig()
        self.assertNotEqual(None, conf.info)
        # even this fails on travis
        # self.assertTrue(conf.info.num_devices() >= 1)
        # conf.info.write()
