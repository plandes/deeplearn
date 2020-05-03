import unittest
import logging
import itertools as it
from io import StringIO
import torch
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
        conf.write(writer)
        logger.debug(writer.getvalue())
        self.assertTrue(len(writer.getvalue()) > 0)

    def test_cuda_config_cpu(self):
        conf = TorchConfig(False)
        self.assertEqual(TorchConfig.CPU_DEVICE, conf.device.type)

    def test_config_type(self):
        conf = TorchConfig(False)
        self.assertEqual(torch.float32, conf.data_type)
        self.assertEqual(torch.FloatTensor, conf.tensor_class)

    def test_create_tensor(self):
        conf = TorchConfig(False)
        tensor = conf.from_iterable(it.islice(it.count(), 5))
        self.assertEqual(torch.float32, tensor.dtype)
        should = torch.FloatTensor([0, 1, 2, 3, 4])
        self.assertTrue(torch.all(should.eq(tensor)))
