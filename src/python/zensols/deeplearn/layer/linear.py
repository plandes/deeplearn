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

    :param proportions: whether or not to interpret ``middle_features`` as
                        a proportion of the previous layer or use directly
                        as the size of the middle layer

    :param repeats: the number of repeats of the ``middle_features``
                    configuration

    """
    in_features: int
    out_features: int
    middle_features: Tuple[Any]
    proportions: bool
    repeats: int

    def get_module_class_name(self) -> str:
        return __name__ + '.DeepLinear'


class DeepLinear(BaseNetworkModule, Deallocatable):
    """A layer that has contains one more nested layers.  The input and output
    layer shapes are given and an optional 0 or more middle layers are given as
    percent changes in size or exact numbers.

    The drop out and activation function (if any) are applied in between each
    layer allowing other drop outs and activation functions to be applied
    before and after.

    """
    def __init__(self, net_settings: DeepLinearNetworkSettings,
                 logger: logging.Logger = None):
        """Initialize the deep linear layer.

        """
        super().__init__(net_settings, logger)
        ns = net_settings
        last_feat = ns.in_features
        layers = []
        self.activation_function = ns.activation_function
        self.dropout = None if ns.dropout is None else nn.Dropout(ns.dropout)
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

    def _add_layer(self, in_features: int, out_features: int, dropout: float,
                   layers: list):
        n_layer = len(layers)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'add {n_layer}: in={in_features} out={out_features}')
        layer = nn.Linear(in_features, out_features)
        layers.append(layer)

    def get_layers(self):
        return tuple(self.seq_layers)

    def n_features_after_layer(self, nth_layer):
        return self.get_layers()[nth_layer].out_features

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        layers = self.get_layers()
        llen = len(layers)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'num layers: {llen}')
        for i, layer in enumerate(layers):
            x = layer(x)
            self._shape_debug('deep linear', x)
            if i < llen - 1:
                if self.activation_function is not None:
                    x = self.activation_function(x)
                if self.dropout is not None:
                    x = self.dropout(x)
        return x
