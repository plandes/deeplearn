import logging
import unittest
from zensols.deeplearn.layer import Convolution2DLayerFactory

logger = logging.getLogger(__name__)


class TestConvolution(unittest.TestCase):
    def test_conv_dim(self):
        lf = Convolution2DLayerFactory(
            width=227,
            height=227,
            depth=3,
            n_filters=96,
            kernel_filter=(11, 11),
            stride=4,
            padding=0)
        self.assertEqual(lf.W_row, (96, 363))
        self.assertEqual(lf.W_out, 55)
        self.assertEqual(lf.X_col, (363, 3025))
        self.assertEqual(lf.out_conv_shape, (96, 55, 55))
