"""Convolution network creation utilities.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Any
from dataclasses import dataclass, field
from abc import abstractmethod, ABCMeta
import copy as cp
from functools import reduce
import math
from torch import nn
from . import LayerError


class Flattenable(object):
    """A class with a :obj:`flatten_dim` and :obj:`out_shape` properties.

    """
    @property
    def out_shape(self) -> Tuple[int]:
        """Return the shape of the layer after flattened in to one dimension.

        """
        pass

    @property
    def flatten_dim(self) -> int:
        """Return the number or neurons of the layer after flattening in to one
        dimension.

        """
        return reduce(lambda x, y: x * y, self.out_shape)

    def __str__(self):
        sup = super().__str__()
        return f'{sup}, out: {self.out_shape}'


class Im2DimCalculator(Flattenable):
    """Convolution matrix dimension calculation utility.

    Implementation as Matrix Multiplication section.

    Example (im2col)::
      W_in = H_in = 227
      Ch_in = D_in = 3
      Ch_out = D_out = 3
      K = 96
      F = (11, 11)
      S = 4
      P = 0
      W_out = H_out = 227 - 11 + (2 * 0) / 4 = 55 output locations
      X_col = Fw^2 * D_out x W_out * H_out = 11^2 * 3 x 55 * 55 = 363 x 3025

    Example (im2row)::
      W_row = 96 filters of size 11 x 11 x 3 => K x 11 * 11 * 3 = 96 x 363

    Result of convolution: transpose(W_row) dot X_col.  Must reshape back to 55
    x 55 x 96

    :see: `Stanford <http://cs231n.github.io/convolutional-networks/#conv>`_

    """
    def __init__(self, W: int, H: int, D: int = 1, K: int = 1,
                 F: Tuple[int, int] = (2, 2), S: int = 1, P: int = 0):
        """Initialize.

        :param W: width

        :param H: height

        :param D: depth [of volume] (usually same as K)

        :param K: number of filters

        :param F: tuple of kernel/filter (width, height)

        :param S: stride

        :param P: padding
        """
        self.W = W
        self.H = H
        self.D = D
        self.K = K
        self.F = F
        self.S = S
        self.P = P

    def validate(self):
        W, H, F, P, S = self.W, self.H, self.F, self.P, self.S
        if ((W - F[0] + (2 * P)) % S):
            raise LayerError('Incongruous convolution width layer parameters')
        if ((H - F[1] + (2 * P)) % S):
            raise LayerError('Incongruous convolution height layer parameters')
        if (F[0] > (W + (2 * P))):
            raise LayerError(f'Kernel/filter {F} must be <= width {W} + 2 * padding {P}')
        if (F[1] > (H + (2 * P))):
            raise LayerError(f'Kernel/filter {F} must be <= height {H} + 2 * padding {P}')
        if self.W_row[1] != self.X_col[0]:
            raise LayerError(f'Columns of W_row {self.W_row} do not match ' +
                             f'rows of X_col {self.X_col}')

    @property
    def W_out(self):
        return int(((self.W - self.F[0] + (2 * self.P)) / self.S) + 1)

    @property
    def H_out(self):
        return int(((self.H - self.F[1] + (2 * self.P)) / self.S) + 1)

    @property
    def X_col(self):
        # TODO: not supported for non-square filters
        return (self.F[0] ** 2 * self.D, self.W_out * self.H_out)

    @property
    def W_row(self):
        # TODO: not supported for non-square filters
        return (self.K, (self.F[0] ** 2) * self.D)

    @property
    def out_shape(self):
        return (self.K, self.W_out, self.H_out)

    def flatten(self, axis: int = 1):
        fd = self.flatten_dim
        W, H = (1, fd) if axis else (fd, 1)
        return self.__class__(W, H, F=(1, 1), D=1, K=1)

    def __str__(self):
        attrs = 'W H D K F S P W_out H_out W_row X_col out_shape'.split()
        return ', '.join(map(lambda x: f'{x}={getattr(self, x)}', attrs))

    def __repr__(self):
        return self.__str__()


@dataclass
class ConvolutionLayerFactory(object):
    """Create convolution layers.  Each attribute maps a corresponding attribuate
    variable in :class:`.Im2DimCalculator`, which documented in the parenthesis
    in the parameter documentation below.

    :param width: the width of the image/data (``W``)

    :param height: the height of the image/data (``H``)

    :param depth: the volume, which is usually same as ``n_filters`` (``D``)

    :param n_filters: the number of filters, aka the filter depth/volume
                      (``K``)

    :param kernel_filter: the kernel filter dimension in width X height (``F``)

    :param stride: the stride, which is the number of cells to skip for each
                   convolution (``S``)

    :param padding: the zero'd number of cells on the ends of the image/data
                    (``P``)

    :see: `Stanford <http://cs231n.github.io/convolutional-networks/#conv>`_

    """
    width: int = field(default=1)
    height: int = field(default=1)
    depth: int = field(default=1)
    n_filters: int = field(default=1)
    kernel_filter: Tuple[int, int] = field(default=(2, 2))
    stride: int = field(default=1)
    padding: int = field(default=0)

    @property
    def calc(self) -> Im2DimCalculator:
        return Im2DimCalculator(**{
            'W': self.width,
            'H': self.height,
            'D': self.depth,
            'K': self.n_filters,
            'F': self.kernel_filter,
            'S': self.stride,
            'P': self.padding})

    def copy_calc(self, calc: Im2DimCalculator):
        self.width = calc.W
        self.height = calc.H
        self.depth = calc.D
        self.n_filters = calc.K
        self.kernel_filter = calc.F
        self.stride = calc.S
        self.padding = calc.P

    def flatten(self) -> Any:
        """Return a new flattened instance of this class.

        """
        clone = self.clone()
        calc = clone.calc.flatten()
        clone.copy_calc(calc)
        return clone

    @property
    def flatten_dim(self) -> int:
        """Return the dimension of a flattened array of the convolution layer
        represented by this instance.

        """
        return self.calc.flatten_dim

    def clone(self, **kwargs) -> Any:
        """Return a clone of this factory instance.

        """
        clone = cp.deepcopy(self)
        clone.__dict__.update(kwargs)
        return clone

    def conv1d(self) -> nn.Conv1d:
        """Return a convolution layer in one dimension.

        """
        c = self.calc
        return nn.Conv1d(c.D, c.K, c.F, padding=c.P, stride=c.S)

    def conv2d(self) -> nn.Conv2d:
        """Return a convolution layer in two dimensions.

        """
        c = self.calc
        return nn.Conv2d(c.D, c.K, c.F, padding=c.P, stride=c.S)

    def batch_norm2d(self) -> nn.BatchNorm2d:
        """Return a 2D batch normalization layer.
        """
        return nn.BatchNorm2d(self.calc.K)

    def __str__(self):
        return str(self.calc)


@dataclass
class PoolFactory(Flattenable, metaclass=ABCMeta):
    """Create a 2D max pool and output it's shape.

    :see: `Stanford <https://cs231n.github.io/convolutional-networks/#pool>`_

    """
    layer_factory: ConvolutionLayerFactory = field(repr=False, default=None)
    stride: int = field(default=1)
    padding: int = field(default=0)

    @abstractmethod
    def _calc_out_shape(self) -> Tuple[int]:
        pass

    @abstractmethod
    def create_pool(self) -> nn.Module:
        pass

    @property
    def out_shape(self) -> Tuple[int]:
        """Calculates the dimensions for a max pooling filter and creates a layer.

        :param F: the spacial extent (kernel filter)

        :param S: the stride

        """
        return self._calc_out_shape()

    def __call__(self) -> nn.Module:
        """Return the pooling layer.

        """
        return self.create_pool()


@dataclass
class MaxPool1dFactory(PoolFactory):
    """Create a 1D max pool and output it's shape.

    """
    kernel_filter: Tuple[int] = field(default=2)
    """The filter used for max pooling."""

    def _calc_out_shape(self) -> Tuple[int]:
        calc = self.layer_factory.calc
        L = calc.flatten_dim
        F = self.kernel_filter
        S = self.stride
        P = self.padding
        Lo = math.floor((((L + (2 * P) - (F - 1) - 1)) / S) + 1)
        return (1, Lo)

    def create_pool(self) -> nn.Module:
        return nn.MaxPool1d(
            self.kernel_filter, stride=self.stride, padding=self.padding)


@dataclass
class MaxPool2dFactory(PoolFactory):
    """Create a 2D max pool and output it's shape.

    """
    kernel_filter: Tuple[int, int] = field(default=(2, 2))
    """The filter used for max pooling."""

    def _calc_out_shape(self) -> Tuple[int]:
        calc = self.layer_factory.calc
        K, W, H = calc.out_shape
        F = self.kernel_filter
        S = self.stride
        P = self.padding
        W_2 = ((W - F[0] + (2 * P)) / S) + 1
        H_2 = ((H - F[1] + (2 * P)) / S) + 1
        return (K, int(W_2), int(H_2))

    def create_pool(self) -> nn.Module:
        return nn.MaxPool2d(
            self.kernel_filter, stride=self.stride, padding=self.padding)
