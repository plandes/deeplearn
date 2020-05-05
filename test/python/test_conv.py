import logging
import unittest
from zensols.deeplearn import (
    TorchConfig,
    Im2DimCalculator,
)

logger = logging.getLogger(__name__)


class TestConvolution(unittest.TestCase):
    def test_conv_dim(self):
        ic = Im2DimCalculator(227, 227, 3, 96, (11, 11), 4, 0)
        self.assertEqual(ic.W_row, (96, 363))
        self.assertEqual(ic.W_out, 55)
        self.assertEqual(ic.X_col, (363, 3025))
        self.assertEqual(ic.out_shape, (96, 55, 55))
