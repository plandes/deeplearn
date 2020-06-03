"""Convenience classes for linear layers.

"""
__author__ = 'Paul Landes'

import logging
from torch import nn
from typing import List, Any
from torch.functional import F
from zensols.persist import persisted

logger = logging.getLogger(__name__)


class DeepLinear(nn.Module):
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
        middle_features = () if middle_features is None else middle_features
        last_feat = in_features
        layers = []
        self.activation_function = activation_function
        self.dropout = dropout
        for mf in middle_features:
            if proportions:
                next_feat = int(last_feat * mf)
            else:
                next_feat = int(mf)
            self._add_layer(last_feat, next_feat, dropout, layers)
            last_feat = next_feat
        self._add_layer(last_feat, out_features, dropout, layers)
        self.seq_layers = nn.Sequential(*layers)

    def _add_layer(self, in_features: int, out_features: int, dropout: float,
                   layers: list):
        n_layer = len(layers)
        logger.debug(f'{n_layer}: in={in_features} out={out_features}')
        layer = nn.Linear(in_features, out_features)
        layers.append(layer)
        if self.dropout is not None:
            logger.debug(f'adding dropout layer with droput={self.dropout}')
            layers.append(nn.Dropout(self.dropout))

    def get_layers(self):
        return tuple(self.seq_layers)

    def n_features_after_layer(self, nth_layer):
        return self.get_layers()[nth_layer].out_features

    def forward(self, x):
        layers = self.get_layers()
        llen = len(layers)
        for i, layer in enumerate(layers):
            if i > 0 and i < llen - 1 and self.activation_function is not None:
                x = self.activation_function(x)
            x = layer(x)

        return x


class MaxPool1dFactory(object):
    """A factor for 1D max pooling layers.

    """
    def __init__(self, W: int, F: int, S: int = 1, P: int = 1):
        """Initialize.

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
