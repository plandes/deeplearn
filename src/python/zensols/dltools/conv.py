"""Convolution network creation utilities.

"""
__author__ = 'Paul Landes'

import logging
from functools import reduce
from torch import nn
from zensols.persist import persisted

logger = logging.getLogger(__name__)


class Im2DimCalculator(object):
    """
    Convolution matrix dimension calculation utility.

    http://cs231n.github.io/convolutional-networks/#conv
    Implementation as Matrix Multiplication section.

    Example (im2col):
    W_in = H_in = 227
    Ch_in = D_in = 3
    Ch_out = D_out = 3
    K = 96
    F = (11, 11)
    S = 4
    P = 0
    W_out = H_out = 227 - 11 + (2 * 0) / 4 = 55 output locations
    X_col = Fw^2 * D_out x W_out * H_out = 11^2 * 3 x 55 * 55 = 363 x 3025

    Example (im2row):
    W_row = 96 filters of size 11 x 11 x 3 => K x 11 * 11 * 3 = 96 x 363

    Result of convolution: transpose(W_row) dot X_col
    Must reshape back to 55 x 55 x 96
    """

    def __init__(self, W, H, D=1, K=1, F=(2, 2), S=1, P=0):
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

    def clone(self, W=None, H=None, **kwargs):
        nkw = {'W': self.W,
               'H': self.H,
               'D': self.D,
               'K': self.K,
               'F': self.F,
               'S': self.S,
               'P': self.P}
        nkw.update(kwargs)
        W = self.W if W is None else W
        H = self.H if H is None else H
        return self.__class__(W, H, **kwargs)

    def validate(self):
        W, H, F, P, S = self.W, self.H, self.F, self.P, self.S
        if ((W - F[0] + (2 * P)) % S):
            raise ValueError('incongruous convolution width layer parameters')
        if ((H - F[1] + (2 * P)) % S):
            raise ValueError('incongruous convolution height layer parameters')
        if (F[0] > (W + (2 * P))):
            raise ValueError(f'kernel/filter {F} must be <= width {W} + 2 * padding {P}')
        if (F[1] > (H + (2 * P))):
            raise ValueError(f'kernel/filter {F} must be <= height {H} + 2 * padding {P}')
        if self.W_row[1] != self.X_col[0]:
            raise ValueError(f'columns of W_row {self.W_row} do not match ' +
                             f'rows of X_col {self.X_col}')

    @property
    @persisted('_W_out')
    def W_out(self):
        return int(((self.W - self.F[0] + (2 * self.P)) / self.S) + 1)

    @property
    @persisted('_H_out')
    def H_out(self):
        return int(((self.H - self.F[1] + (2 * self.P)) / self.S) + 1)

    @property
    @persisted('_X_col')
    def X_col(self):
        # TODO: not supported for non-square filters
        return (self.F[0] ** 2 * self.D, self.W_out * self.H_out)

    @property
    @persisted('_W_col')
    def W_row(self):
        # TODO: not supported for non-square filters
        return (self.K, (self.F[0] ** 2) * self.D)

    @property
    @persisted('_out_shape')
    def out_shape(self):
        return (self.K, self.W_out, self.H_out)

    @property
    @persisted('_flatten_dim')
    def flatten_dim(self):
        return reduce(lambda x, y: x * y, self.out_shape)

    def flatten(self, axis=1):
        fd = self.flatten_dim
        W, H = (1, fd) if axis else (fd, 1)
        return self.__class__(W, H, F=(1, 1), D=1, K=1)

    def __str__(self):
        attrs = 'W H D K F S P W_out H_out W_row X_col out_shape'.split()
        return ', '.join(map(lambda x: f'{x}={getattr(self, x)}', attrs))

    def __repr__(self):
        return self.__str__()


class ConvolutionLayerFactory(object):
    """Create convolution layers.
    """
    def __init__(self, *args, **kwargs):
        """Create a layer factory using the same arguments as given in
        ``Im2DimCalculator``.

        """
        if len(args) > 0 and isinstance(args[0], Im2DimCalculator):
            calc = args[0]
        else:
            calc = Im2DimCalculator(*args, **kwargs)
        self.calc = calc

    def flatten(self, *args, **kwargs):
        """Return a flat layer with arguments given to ``Im2DimCalculator``.

        """
        return self.__class__(self.calc.flatten(*args, **kwargs))

    @property
    def flatten_dim(self):
        """Return the dimension of a flattened array of the convolution layer
        represented by this instance.

        """
        return self.calc.flatten_dim

    def clone(self, *args, **kwargs):
        """Return a clone of this factory instance.

        """
        return self.__class__(self.calc.clone(*args, **kwargs))

    def conv1d(self):
        """Return a convolution layer in one dimension.

        """
        c = self.calc
        return nn.Conv1d(c.D, c.K, c.F, padding=c.P, stride=c.S)

    def conv2d(self):
        """Return a convolution layer in two dimensions.

        """
        c = self.calc
        return nn.Conv2d(c.D, c.K, c.F, padding=c.P, stride=c.S)

    def batch_norm2d(self):
        """Return a 2D batch normalization layer.
        """
        return nn.BatchNorm2d(self.calc.K)

    def max_pool1d(self):
        """Return a one dimensional max pooling layer.

        """
        return nn.MaxPool1d(self.calc.F[1], stride=self.calc.S)

    def __str__(self):
        return str(self.calc)
