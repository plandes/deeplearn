from typing import Sequence
import torch
import unittest
import random as rand
from torch import Tensor
from functools import reduce
from zensols.deeplearn import TorchConfig
from zensols.deeplearn.vectorize import NonUniformDimensionEncoder


class TestNonUniformDimensionEncoder(unittest.TestCase):
    def setUp(self):
        tc = TorchConfig(False)
        self.de = NonUniformDimensionEncoder(tc)

    def _trans_test(self, arrs: Sequence[Tensor]):
        enc = self.de.encode(arrs)
        decs = self.de.decode(enc)
        for enc, dec in zip(arrs, decs):
            self.assertTrue(TorchConfig.equal(enc, dec))

    def _test_arange(self, dtype: torch.dtype):
        size = (2, 3, 4)
        arr = torch.arange(1, reduce(lambda x, y: x * y, size) + 1, dtype=dtype).view(size)
        arr2 = torch.arange(1, 11)
        size = (3, 2)
        arr3 = torch.arange(1, reduce(lambda x, y: x * y, size) + 1, dtype=dtype).view(size)
        encs = (arr, arr2, arr3)
        self._trans_test(encs)

    def test_float(self):
        # "arange_cpu" not implemented for 'Half'
        #self._test_arange(torch.float16)
        self._test_arange(torch.float32)
        self._test_arange(torch.float64)

    def test_int(self):
        self._test_arange(torch.int16)
        self._test_arange(torch.int32)
        self._test_arange(torch.int64)

    def test_rand(self):
        dtype: torch.dtype = torch.float
        arrs = []
        for i in range(5):
            sz = rand.randint(3, 8)
            shape = tuple(map(lambda _: rand.randint(1, 10), range(sz)))
            arr = torch.rand(shape, dtype=dtype)
            arrs.append(arr)
        self._trans_test(arrs)

    def test_diff(self):
        dtype: torch.dtype = torch.float
        size = (2, 3, 4)
        arr = torch.arange(1, reduce(lambda x, y: x * y, size) + 1, dtype=dtype).view(size)
        arr2 = torch.arange(1, 11)
        size = (3, 2)
        arr3 = torch.arange(1, reduce(lambda x, y: x * y, size) + 1, dtype=dtype).view(size)
        arrs = (arr, arr2, arr3)
        enc = self.de.encode(arrs)
        decs = self.de.decode(enc)
        decs[2][1][1] = 1.11
        for enc, dec, tf in zip(arrs, decs, [True, True, False]):
            if tf:
                self.assertTrue(TorchConfig.equal(enc, dec))
            else:
                self.assertFalse(TorchConfig.equal(enc, dec))
