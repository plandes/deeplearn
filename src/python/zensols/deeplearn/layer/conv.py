
"""Convolution network creation utilities.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import Tuple, Set, Iterable, ClassVar
import dataclasses
from dataclasses import dataclass, field
from abc import abstractmethod, ABCMeta
import logging
import math
from torch import nn
from zensols.config import Dictable
from . import LayerError

logger = logging.getLogger(__name__)


@dataclass
class ConvolutionLayerFactory(Dictable, metaclass=ABCMeta):
    """Create convolution layers and output shape calculator.

    """
    _DICTABLE_ATTRIBUTES: ClassVar[Set[str]] = {'out_conv_shape'}

    stride: int = field(default=1)
    """The stride, which is the number of cells to skip for each (``S``)."""

    padding: int = field(default=0)
    """The zero'd number of cells on the ends of the image/data (``P``)."""

    pool_stride: int = field(default=1)
    """The pooling stride, which is the number of cells to skip for each."""

    pool_padding: int = field(default=0)
    """The pooling zero'd number of cells on the ends of the image/data."""

    @property
    def S(self) -> int:
        """Stride."""
        return self.stride

    @property
    def P(self) -> int:
        """Padding."""
        return self.padding

    @abstractmethod
    def _get_dim(self) -> int:
        pass

    @abstractmethod
    def _calc_conv_out_shape(self) -> Tuple[int, ...]:
        pass

    @abstractmethod
    def _calc_pool_out_shape(self) -> Tuple[int, ...]:
        pass

    @abstractmethod
    def _validate(self) -> str:
        pass

    @abstractmethod
    def create_conv_layer(self) -> nn.Module:
        """Create the convolution layer for this layer in the stack."""
        pass

    @abstractmethod
    def create_pool_layer(self) -> nn.Module:
        """Create the pool layer that follows the convolutional layer."""
        pass

    @abstractmethod
    def create_batch_norm_layer(self) -> nn.Module:
        """Create the batch norm layer that follows the pool layer."""
        pass

    @property
    def dim(self) -> int:
        return self._get_dim()

    def validate(self, raise_error: bool = True) -> str:
        """Validate the parameters of the factory.

        :param raise_error: if ``True`` raises and error when invalid

        :raises LayerError: if invalid and ``raise_error`` is ``True``

        """
        err: str = self._validate()
        if raise_error and err is not None:
            raise LayerError(f'Invalid convolution: {type(self)}: {err}')
        return err

    @property
    def out_conv_shape(self) -> Tuple[int, ...]:
        """The convolution layer shape before flattened in to one dimension."""
        return self._calc_conv_out_shape()

    @property
    def out_pool_shape(self) -> Tuple[int, ...]:
        """The pooling layer shape before flattened in to one dimension."""
        return self._calc_pool_out_shape()

    @abstractmethod
    def _next_layer(self, use_pool: bool = True) -> ConvolutionLayerFactory:
        pass

    def next_layer(self, use_pool: bool = True) -> ConvolutionLayerFactory:
        """Get a new factory that represents the next layer of the convolution
        stack.

        :param use_pool: whether to use the output shape of the pool for the
                         next layer's intput and output chanel settings

        """
        return self._next_layer(use_pool)

    def iter_layers(self, use_pool: bool = True) -> \
            Iterable[ConvolutionLayerFactory]:
        """Iterate through over subsequent convolution and pooled stacked
        networks.  Use with :function:`itertools.islice` to limit the output.

        :return: subsequent layers *after* the current instance for all valid
                 layers

        """
        fac: ConvolutionLayerFactory = self
        while fac.validate(False) is None:
            fac = fac._next_layer()
            yield fac

    def clone(self) -> ConvolutionLayerFactory:
        """Return a clone of this factory instance."""
        return dataclasses.replace(self)

    def __str__(self):
        return f'{self.dim}D convolution, out shape: {self.out_pool_shape}'


@dataclass
class Convolution1DLayerFactory(ConvolutionLayerFactory):
    """Two dimensional convoluation and output shape factory.

    """
    in_channels: int = field(default=1)
    """Number of channels in the input image (``C_in``)."""

    out_channels: int = field(default=1)
    """Number of channels/filters produced by the convolution."""

    kernel_filter: int = field(default=2)
    """Size of the kernel filter dimension in length (``F``)."""

    pool_kernel_filter: Tuple[int] = field(default=2)
    """The filter used for max pooling."""

    @property
    def C_in(self) -> int:
        return self.in_channels

    @property
    def L_in(self) -> int:
        return self.out_channels

    @property
    def F(self) -> int:
        return self.kernel_filter

    def _get_dim(self) -> int:
        return 1

    def _calc_conv_out_shape(self) -> Tuple[int, ...]:
        L_out = math.floor(
            (((self.L_in + (2 * self.P) - (self.F - 1) - 1)) / self.S) + 1)
        return (self.out_channels, L_out)

    def _calc_pool_out_shape(self) -> Tuple[int, ...]:
        L_out = self.out_conv_shape[1]
        S = self.pool_stride
        P = self.pool_padding
        F = self.pool_kernel_filter
        L_out_pool = math.floor((((L_out + (2 * P) - (F - 1) - 1)) / S) + 1)
        return (self.out_channels, L_out_pool)

    def _next_layer(self, use_pool: bool = True) -> ConvolutionLayerFactory:
        prev_shape: Tuple[int, int]
        if use_pool:
            prev_shape = self.out_pool_shape
        else:
            prev_shape = self.out_conv_shape
        clone = self.clone()
        clone.in_channels = prev_shape[0]
        clone.out_channels = prev_shape[1]
        return clone

    def _validate(self) -> str:
        if self.in_channels <= 0:
            return 'input length must be greater than 0'
        if self.kernel_filter <= 0:
            return 'kernel size must be greater than 0'
        if self.stride <= 0:
            return 'stride must be greater than 0'
        if self.padding < 0:
            return 'padding must be non-negative'
        out: float = (((self.L_in + (2 * self.P) - (self.F - 1) - 1)) / self.S)
        if out <= 0:
            return f'output length ({out}) is non-positive'
        # if not out.is_integer():
        #     return f'output length ({out}) is not an integer'

    def create_conv_layer(self) -> nn.Module:
        return nn.Conv1d(
            in_channels=self.in_channels,  # C_in
            out_channels=self.out_channels,
            kernel_size=self.kernel_filter,  # F
            padding=self.padding,
            stride=self.stride)

    def create_pool_layer(self) -> nn.Module:
        return nn.MaxPool1d(
            kernel_size=self.pool_kernel_filter,
            stride=self.pool_stride,
            padding=self.pool_padding)

    def create_batch_norm_layer(self) -> nn.Module:
        return nn.BatchNorm1d(self.out_pool_shape[0])


@dataclass
class Convolution2DLayerFactory(ConvolutionLayerFactory):
    """Two dimensional convoluation and output shape factory.  Implementation as
    matrix multiplication section taken from the `Standford CNN`_ class.

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

    .. _Stanford CNN: <http://cs231n.github.io/convolutional-networks/#conv>

    """
    width: int = field(default=1)
    """The width of the image/data (``W``)."""

    height: int = field(default=1)
    """The height of the image/data (``H``)."""

    depth: int = field(default=1)
    """The volume, which is usually same as ``n_filters`` (``D``)."""

    kernel_filter: Tuple[int, int] = field(default=(2, 2))
    """The kernel filter dimension in width X height (``F``)."""

    n_filters: int = field(default=1)
    """The number of filters, aka the filter depth/volume (``K``)."""

    pool_kernel_filter: Tuple[int] = field(default=(2, 2))
    """The filter used for max pooling."""

    @property
    def W(self) -> int:
        return self.width

    @property
    def H(self) -> int:
        return self.height

    @property
    def D(self) -> int:
        return self.depth

    @property
    def K(self) -> int:
        return self.n_filters

    @property
    def F(self) -> int:
        return self.kernel_filter

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

    def _get_dim(self) -> int:
        return 2

    def _calc_conv_out_shape(self) -> Tuple[int, ...]:
        return (self.K, self.W_out, self.H_out)

    def _calc_pool_out_shape(self) -> Tuple[int, ...]:
        K, W, H = self.out_conv_shape
        F = self.pool_kernel_filter
        S = self.pool_stride
        P = self.pool_padding
        W_2 = ((W - F[0] + (2 * P)) / S) + 1
        H_2 = ((H - F[1] + (2 * P)) / S) + 1
        return (K, int(W_2), int(H_2))

    def _next_layer(self, use_pool: bool = True) -> ConvolutionLayerFactory:
        prev_shape: Tuple[int, int]
        if use_pool:
            prev_shape = self.out_pool_shape
        else:
            prev_shape = self.out_conv_shape
        clone = self.clone()
        clone.depth, clone.width, clone.height = prev_shape
        return clone

    def _validate(self) -> str:
        err: str = None
        W, H, F, P, S = self.W, self.H, self.F, self.P, self.S
        if ((W - F[0] + (2 * P)) % S):
            err = 'incongruous convolution width layer parameters'
        if ((H - F[1] + (2 * P)) % S):
            err = 'incongruous convolution height layer parameters'
        if (F[0] > (W + (2 * P))):
            err = f'kernel/filter {F} must be <= width {W} + 2 * padding {P}'
        if (F[1] > (H + (2 * P))):
            err = f'kernel/filter {F} must be <= height {H} + 2 * padding {P}'
        if self.W_row[1] != self.X_col[0]:
            err = (f'columns of W_row {self.W_row} do not match ' +
                   f'rows of X_col {self.X_col}')
        return err

    def create_conv_layer(self) -> nn.Module:
        return nn.Conv2d(self.D, self.K, self.F, padding=self.P, stride=self.S)

    def create_pool_layer(self) -> nn.Module:
        return nn.MaxPool2d(
            self.pool_kernel_filter,
            stride=self.pool_stride,
            padding=self.pool_padding)

    def create_batch_norm_layer(self) -> nn.Module:
        return nn.BatchNorm2d(self.out_pool_shape[0])

    def __str__(self):
        attrs = 'W H D K F S P W_out H_out W_row X_col out_shape'.split()
        return ', '.join(map(lambda x: f'{x}={getattr(self, x)}', attrs))
