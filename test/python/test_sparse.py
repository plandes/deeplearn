import logging
import torch
from scipy.sparse.csr import csr_matrix
from zensols.deeplearn import TorchConfig
from zensols.deeplearn.vectorize import SparseTensorFeatureContext
from util import TargetTestCase

logger = logging.getLogger(__name__)


class TestSparseMatrixContext(TargetTestCase):
    CONF = None

    def setUp(self):
        super().setUp()
        self.conf = TorchConfig(False, data_type=torch.float64)

    def test_sparse(self):
        conf = self.conf
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
        tarr = torch.tensor(should)
        ctx = SparseTensorFeatureContext.instance('afeattype', tarr, conf)
        should = conf.singleton(should, dtype=tarr.dtype)
        dense = ctx.to_tensor(conf)
        self.assertTensorEquals(should, dense)

    def rand_assert(self, iters, size, conf):
        for i in range(iters):
            should = torch.rand(size, dtype=conf.data_type)
            should = conf.to(should)
            ctx = SparseTensorFeatureContext.instance(
                'some_feature_id', should, conf)
            self.assertTensorEquals(should, conf.to(ctx.to_tensor(conf)))

    def test_rand(self):
        conf = self.conf
        size = (10, 20)
        self.rand_assert(50, size, conf)
        conf = TorchConfig(True, data_type=torch.float64)
        self.rand_assert(50, size, conf)

    def test_1d_int_mat(self):
        should = torch.randint(0, 5, (11,))
        ctx = SparseTensorFeatureContext.instance('afeattype', should, self.conf)
        for m in ctx.sparse_arr:
            self.assertTrue(isinstance(m, csr_matrix))
        dense = ctx.to_tensor(self.conf)
        self.assertTensorEquals(should, dense)
        self.assertEqual(should.shape, dense.shape)

    def test_2d_int_mat(self):
        should = torch.randint(0, 5, (7, 11))
        ctx = SparseTensorFeatureContext.instance('afeattype', should, self.conf)
        for m in ctx.sparse_arr:
            self.assertTrue(isinstance(m, csr_matrix))
        dense = ctx.to_tensor(self.conf)
        self.assertTensorEquals(should, dense)
        self.assertEqual(should.shape, dense.shape)

    def test_2d_1_int_mat(self):
        should = torch.randint(0, 5, (1, 11))
        ctx = SparseTensorFeatureContext.instance('afeattype', should, self.conf)
        for m in ctx.sparse_arr:
            self.assertTrue(isinstance(m, csr_matrix))
        dense = ctx.to_tensor(self.conf)
        self.assertTensorEquals(should, dense)
        self.assertEqual(should.shape, dense.shape)

    def test_3d_int_mat(self):
        should = torch.randint(0, 5, (2, 7, 11))
        ctx = SparseTensorFeatureContext.instance('afeattype', should, self.conf)
        for m in ctx.sparse_arr:
            self.assertTrue(isinstance(m, csr_matrix))
        dense = ctx.to_tensor(self.conf)
        self.assertTensorEquals(should, dense)
        self.assertEqual(should.shape, dense.shape)

    def test_3d_1_int_mat(self):
        should = torch.randint(0, 5, (1, 7, 11))
        ctx = SparseTensorFeatureContext.instance('afeattype', should, self.conf)
        for m in ctx.sparse_arr:
            self.assertTrue(isinstance(m, csr_matrix))
        dense = ctx.to_tensor(self.conf)
        self.assertTensorEquals(should, dense)
        self.assertEqual(should.shape, dense.shape)

    def test_3d_1_1_int_mat(self):
        should = torch.randint(0, 5, (1, 1, 11))
        ctx = SparseTensorFeatureContext.instance('afeattype', should, self.conf)
        for m in ctx.sparse_arr:
            self.assertTrue(isinstance(m, csr_matrix))
        dense = ctx.to_tensor(self.conf)
        self.assertTensorEquals(should, dense)
        self.assertEqual(should.shape, dense.shape)

    def test_1d_float_mat(self):
        should = torch.rand((11,))
        ctx = SparseTensorFeatureContext.instance('afeattype', should, self.conf)
        for m in ctx.sparse_arr:
            self.assertTrue(isinstance(m, csr_matrix))
        dense = ctx.to_tensor(self.conf)
        self.assertTensorEquals(should, dense)
        self.assertEqual(should.shape, dense.shape)

    def test_2d_float_mat(self):
        should = torch.rand((7, 11))
        ctx = SparseTensorFeatureContext.instance('afeattype', should, self.conf)
        for m in ctx.sparse_arr:
            self.assertTrue(isinstance(m, csr_matrix))
        dense = ctx.to_tensor(self.conf)
        self.assertTensorEquals(should, dense)
        self.assertEqual(should.shape, dense.shape)

    def test_3d_float_mat(self):
        should = torch.rand((2, 7, 11))
        ctx = SparseTensorFeatureContext.instance('afeattype', should, self.conf)
        for m in ctx.sparse_arr:
            self.assertTrue(isinstance(m, csr_matrix))
        dense = ctx.to_tensor(self.conf)
        self.assertTensorEquals(should, dense)
        self.assertEqual(should.shape, dense.shape)
