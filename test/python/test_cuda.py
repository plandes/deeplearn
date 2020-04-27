import unittest
import logging
from io import StringIO
from zensols.dltools import CudaConfig

logger = logging.getLogger(__name__)


class TestCuda(unittest.TestCase):
    def test_cuda_config(self):
        conf = CudaConfig()
        self.assertNotEqual(None, conf.info)
        # even this fails on travis
        # self.assertTrue(conf.info.num_devices() >= 1)
        # conf.info.write()

    def test_cuda_config_write(self):
        writer = StringIO()
        conf = CudaConfig()
        #conf._init_device()
        conf.write(writer)
        logger.debug(writer.getvalue())
        self.assertTrue(len(writer.getvalue()) > 0)

    def test_cuda_config_cpu(self):
        writer = StringIO()
        conf = CudaConfig(False)
        self.assertEqual(CudaConfig.CPU_DEVICE, conf.device.type)
