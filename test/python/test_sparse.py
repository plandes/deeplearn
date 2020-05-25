import logging
import torch
from zensols.deeplearn import TorchConfig
from zensols.vectorize import SparseTensorFeatureContext
from util import TargetTestCase

logger = logging.getLogger(__name__)


class TestSparseMatrixContext(TargetTestCase):
    CONF = None

    def setUp(self):
        super().setUp()
        self.conf = TorchConfig(False, data_type=torch.float64)

    def test_sparse(self):
        conf = self.conf
        ctx = SparseTensorFeatureContext(
            'some_feature_type',
            torch.LongTensor([[7,  12,  13,  15,  18,  20],
                              [3,   2,   5,   0,   4,   6]]),
            conf.singleton([1., 1.5, 10.5, 2.5, 1., 13.2]),
            (21, 10))
        should = [
            [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, 0.00,  0.00],
            [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, 0.00,  0.00],
            [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, 0.00,  0.00],
            [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, 0.00,  0.00],
            [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, 0.00,  0.00],
            [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, 0.00,  0.00],
            [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, 0.00,  0.00],
            [ 0.00,  0.00,  0.00,  1.00,  0.00,  0.00,  0.00,  0.00, 0.00,  0.00],
            [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, 0.00,  0.00],
            [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, 0.00,  0.00],
            [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, 0.00,  0.00],
            [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, 0.00,  0.00],
            [ 0.00,  0.00,  1.50,  0.00,  0.00,  0.00,  0.00,  0.00, 0.00,  0.00],
            [ 0.00,  0.00,  0.00,  0.00,  0.00, 10.50,  0.00,  0.00, 0.00,  0.00],
            [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, 0.00,  0.00],
            [ 2.50,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, 0.00,  0.00],
            [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, 0.00,  0.00],
            [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, 0.00,  0.00],
            [ 0.00,  0.00,  0.00,  0.00,  1.00,  0.00,  0.00,  0.00, 0.00,  0.00],
            [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, 0.00,  0.00],
            [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00, 13.20,  0.00, 0.00,  0.00]]
        should = conf.singleton(should)
        self.assertTensorEquals(should, ctx.to_tensor(conf))

    def rand_assert(self, iters, size, conf):
        for i in range(iters):
            should = torch.rand(size, dtype=conf.data_type)
            should = conf.to(should)
            ctx = SparseTensorFeatureContext.instance(
                'some_feature_type', should, conf)
            self.assertTensorEquals(should, ctx.to_tensor(conf))

    def test_rand(self):
        conf = self.conf
        size = (10, 20)
        self.rand_assert(50, size, conf)
        conf = TorchConfig(True, data_type=torch.float64)
        self.rand_assert(50, size, conf)
