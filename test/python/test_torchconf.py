import unittest
import logging
import itertools as it
from io import StringIO
import torch
from zensols.deeplearn import TorchConfig

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
        conf.write(writer=writer)
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

    def test_create_empty(self):
        conf = TorchConfig(False, data_type=torch.float16)
        tensor = conf.empty((3, 10))
        self.assertEqual(torch.float16, tensor.dtype)
        self.assertEqual(3, tensor.shape[0])
        self.assertEqual(10, tensor.shape[1])

    def test_sparse_create(self):
        conf = TorchConfig(False, data_type=torch.float16)
        arr = conf.sparse(
            [[7,  22,  22,  42,  60,  62,  70,  76, 112, 124, 124,
              128, 135, 141, 153],
             [3,   2,   5,   0,   4,   6,   1,   5,   6,   2,   5,
              4,   3,   0,   1]],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            (174, 30))
        self.assertTrue((174, 30), arr.shape)
        self.assertEqual(0., arr[7, 2].item())
        self.assertEqual(1., arr[7, 3].item())
