import unittest
import logging
from io import StringIO
from zensols.dltools import TorchConfig

logger = logging.getLogger(__name__)


class TestTorchConfig(unittest.TestCase):
    def test_cuda_config(self):
        conf = TorchConfig()
        self.assertNotEqual(None, conf.info)
        # even this fails on travis
        # self.assertTrue(conf.info.num_devices() >= 1)
        # conf.info.write()

    def test_cuda_config_write(self):
        writer = StringIO()
        conf = TorchConfig()
        #conf._init_device()
        conf.write(writer)
        logger.debug(writer.getvalue())
        self.assertTrue(len(writer.getvalue()) > 0)

    def test_cuda_config_cpu(self):
        writer = StringIO()
        conf = TorchConfig(False)
        self.assertEqual(TorchConfig.CPU_DEVICE, conf.device.type)
