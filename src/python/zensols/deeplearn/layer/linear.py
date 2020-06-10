"""Convenience classes for linear layers.

"""
__author__ = 'Paul Landes'

from typing import Any, Tuple
from dataclasses import dataclass
import logging
import torch
from torch import nn
from zensols.persist import persisted, Deallocatable
from zensols.deeplearn import BasicNetworkSettings
from zensols.deeplearn.model import BaseNetworkModule

logger = logging.getLogger(__name__)


@dataclass
class DeepLinearNetworkSettings(BasicNetworkSettings):
    """Settings for a deep fully connected network.

    :param in_features: the number of features to the first layer

    :param out_features: the number of features as output from the last layer

    :param middle_features: the number of features in the middle layers; if
                            ``proportions`` is ``True``, then each number is
                            how much to grow or shrink as a percetage of the
                            last layer, otherwise, it's the number of features

    :param proportions: how to treat the ``middle_features`` parameter

    :param repeats: the number of repeats of the ``middle_features``
                    configuration

    :param debug: whether or not debug ``DeepLinear`` if used as the sole
                  module

    """
    in_features: int
    out_features: int
    middle_features: Tuple[Any]
    proportions: bool
    repeats: int
    debug: bool

    def get_module_class_name(self) -> str:
        return __name__ + '.DeepLinear'


class DeepLinear(BaseNetworkModule, Deallocatable):
    """A layer that has contains one more nested layers.  The input and output
    layer shapes are given and an optional 0 or more middle layers are given as
    percent changes in size or exact numbers.

    """
    def __init__(self, net_settings: DeepLinearNetworkSettings,
                 logger: logging.Logger = None):
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
        super().__init__(net_settings, logger)
        ns = net_settings
        last_feat = ns.in_features
        layers = []
        self.activation_function = ns.activation_function
        self.dropout = ns.dropout
        for mf in ns.middle_features:
            for i in range(ns.repeats):
                if ns.proportions:
                    next_feat = int(last_feat * mf)
                else:
                    next_feat = int(mf)
                self._add_layer(last_feat, next_feat, ns.dropout, layers)
                last_feat = next_feat
        self._add_layer(last_feat, ns.out_features, ns.dropout, layers)
        self.seq_layers = nn.Sequential(*layers)

    def deallocate(self):
        super().deallocate()
        if hasattr(self, 'seq_layers'):
            del self.seq_layers

    def _guard_debug(self):
        pass

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

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
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
