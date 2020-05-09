"""Convenience classes for linear layers.

"""
__author__ = 'Paul Landes'

import logging
from torch import nn
from typing import List, Any
from torch.functional import F
from zensols.persist import persisted

logger = logging.getLogger(__name__)


class LinearLayerFactory(object):
    """Utility class to create linear layers.

    """
    def __init__(self, in_shape, out=None, out_percent=None):
        """Initialize the factory.

        :param in_shape: the shape of the layer
        :param out: the output size of the reshaped layer or use
                    ``out_percent`` if ``None``
        :param out_percent: the output reshaped layer as a percentage of the
                            input size ``in_shape`` if not None, othewise use
                            ``out`.

        """
        self.in_shape = in_shape
        if out is None:
            self.out_features = int(self.flatten_dim * out_percent)
        else:
            self.out_features = out

    @property
    @persisted('_flatten_dim')
    def flatten_dim(self):
        """Return the dimension of a flattened layer represened by the this instances's
        layer.

        """
        from functools import reduce
        return reduce(lambda x, y: x * y, self.in_shape)

    @property
    @persisted('_out_shape')
    def out_shape(self):
        """Return the shape of the output layer.

        """
        return (1, self.flatten_dim)

    def linear(self):
        """Return a new PyTorch linear layer.
        """
        return nn.Linear(self.flatten_dim, self.out_features)

    def __str__(self):
        return f'{self.in_shape} -> {self.out_shape}'


class DeepLinearLayer(nn.Module):
    """A layer that has contains one more nested layers.  The input and output
    layer shapes are given and an optional 0 or more middle layers are given as
    percent changes in size or exact numbers.

    """
    def __init__(self, in_features: int, out_features: int,
                 middle_features: List[Any] = None, dropout: float = None,
                 activation_function=F.relu, proportions: bool = True):
        """Initialize the deep linear layer.

        :param in_features: the number of features coming in to th network
        :param out_features: the number of output features leaving the network
        :param middle_features: a list of percent differences or exact
                                parameter counts of each middle layer; if the
                                former, the next shape is a function of the
                                scaler multiplied by the previous layer; for
                                example ``[1.0]`` creates a nested layer with
                                the exact same shape as the input layer (see
                                ``proportions`` parameter)

        :param dropout: the droput used in all layers or ``None`` to disable

        :param activation_function: the function between all layers, or
                                    ``None`` for no activation

        :param proportions: whether or not to interpret ``middle_features`` as
                            a proportion of the previous layer or use directly
                            as the size of the middle layer

        """
        super().__init__()
        self.layer_attrs = []
        middle_features = () if middle_features is None else middle_features
        last_feat = in_features
        for mf in middle_features:
            if proportions:
                next_feat = int(last_feat * mf)
            else:
                next_feat = int(mf)
            self._add_layer(last_feat, next_feat)
            last_feat = next_feat
        self._add_layer(last_feat, out_features)
        self.dropout = None if dropout is None else nn.Dropout(dropout)
        self.activation_function = activation_function

    def _add_layer(self, in_features, out_features):
        name = f'_layer_{len(self.layer_attrs)}'
        logger.debug(f'{name}: in={in_features} out={out_features}')
        setattr(self, name, nn.Linear(in_features, out_features))
        self.layer_attrs.append(name)

    def get_layers(self):
        layers = []
        for layer_name in self.layer_attrs:
            layers.append(getattr(self, f'{layer_name}'))
        return layers

    def n_features_after_layer(self, nth_layer):
        return self.get_layers()[nth_layer].out_features

    def train(self, mode=True):
        super(DeepLinearLayer, self).train(mode)
        self.is_training = mode

    def eval(self):
        super(DeepLinearLayer, self).eval()
        self.is_training = False

    def forward(self, x):
        for i, aname in enumerate(self.layer_attrs):
            if i > 0:
                x = self.activation_function(x)
            layer = getattr(self, aname)
            x = layer.forward(x)
            if self.is_training and self.dropout is not None:
                x = self.dropout(x)
        return x


class MaxPool1dFactory(object):
    """A factor for 1D max pooling layers.

    """
    def __init__(self, W: int, F: int, S: int = 1, P: int = 1):
        """Initialize

        :param W: width
        :param F: kernel size
        :param S: stride
        :param P: padding

        """
        self.W = W
        self.F = F
        self.S = S
        self.P = P

    @property
    @persisted('_W_out')
    def W_out(self):
        return int(((self.W - self.F + (2 * self.P)) / self.S) + 1)

    def max_pool1d(self):
        """Return a one dimensional max pooling layer.

        """
        return nn.MaxPool1d(self.F, self.S, self.P)
