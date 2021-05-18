"""Convenience classes for linear layers.

"""
__author__ = 'Paul Landes'

from typing import Any, Tuple
from dataclasses import dataclass, field
import logging
import sys
import torch
from torch import nn
from zensols.deeplearn import (
    ActivationNetworkSettings,
    DropoutNetworkSettings,
    BatchNormNetworkSettings,
)
from zensols.deeplearn.model import BaseNetworkModule
from . import LayerError


@dataclass
class DeepLinearNetworkSettings(ActivationNetworkSettings,
                                DropoutNetworkSettings,
                                BatchNormNetworkSettings):
    """Settings for a deep fully connected network using :class:`.DeepLinear`.

    """
    ## TODO: centralize on either in_features or input_size:
    # embedding_output_size, RecurrentCRFNetworkSettings.input_size
    in_features: int = field()
    """The number of features to the first layer."""

    out_features: int = field()
    """The number of features as output from the last layer."""

    middle_features: Tuple[Any] = field()
    """The number of features in the middle layers; if ``proportions`` is
    ``True``, then each number is how much to grow or shrink as a percetage of
    the last layer, otherwise, it's the number of features.

    """

    proportions: bool = field()
    """Whether or not to interpret ``middle_features`` as a proportion of the
    previous layer or use directly as the size of the middle layer.

    """

    repeats: int = field()
    """The number of repeats of the :obj:`middle_features` configuration."""

    def get_module_class_name(self) -> str:
        return __name__ + '.DeepLinear'


class DeepLinear(BaseNetworkModule):
    """A layer that has contains one more nested layers, including batch
    normalization and activation.  The input and output layer shapes are given
    and an optional 0 or more middle layers are given as percent changes in
    size or exact numbers.

    If the network settings are configured to have batch normalization, batch
    normalization layers are added after each linear layer.

    The drop out and activation function (if any) are applied in between each
    layer allowing other drop outs and activation functions to be applied
    before and after.  Note that the activation is implemented as a function,
    and not a layer.

    For example, if batch normalization and an activation function is
    configured and two layers are configured, the network is configured as:

      1. linear
      2. batch normalization
      3. activation
      4. linear
      5. batch normalization
      6. activation

    The module also provides the output features of each layer with
    :py:meth:`n_features_after_layer` and ability to forward though only the
    first given set of layers with :meth:`forward_n_layers`.

    """
    MODULE_NAME = 'linear'

    def __init__(self, net_settings: DeepLinearNetworkSettings,
                 sub_logger: logging.Logger = None):
        """Initialize the deep linear layer.

        :param net_settings: the deep linear layer configuration

        :param sub_logger: the logger to use for the forward process in this
                           layer

        """
        super().__init__(net_settings, sub_logger)
        ns = net_settings
        last_feat = ns.in_features
        lin_layers = []
        bnorm_layers = []
        if self.logger.isEnabledFor(logging.DEBUG):
            self._debug(f'in: {ns.in_features}, ' +
                        f'middle: {ns.middle_features}, ' +
                        f'out: {ns.out_features}')
        for mf in ns.middle_features:
            for i in range(ns.repeats):
                if ns.proportions:
                    next_feat = int(last_feat * mf)
                else:
                    next_feat = int(mf)
                self._add_layer(last_feat, next_feat, ns.dropout,
                                lin_layers, bnorm_layers)
                last_feat = next_feat
        if ns.out_features is not None:
            self._add_layer(last_feat, ns.out_features, ns.dropout,
                            lin_layers, bnorm_layers)
        self.lin_layers = nn.Sequential(*lin_layers)
        if len(bnorm_layers) > 0:
            self.bnorm_layers = nn.Sequential(*bnorm_layers)
        else:
            self.bnorm_layers = None

    def deallocate(self):
        super().deallocate()
        if hasattr(self, 'lin_layers'):
            del self.lin_layers

    def _add_layer(self, in_features: int, out_features: int, dropout: float,
                   lin_layers: list, bnorm_layers):
        ns = self.net_settings
        n_layer = len(lin_layers)
        if self.logger.isEnabledFor(logging.DEBUG):
            self._debug(f'add {n_layer}: in={in_features} out={out_features}')
        lin_layer = nn.Linear(in_features, out_features)
        lin_layers.append(lin_layer)
        if ns.batch_norm_d is not None:
            if out_features is None:
                raise LayerError('Bad out features')
            if ns.batch_norm_features is None:
                bn_features = out_features
            else:
                bn_features = ns.batch_norm_features
            layer = ns.create_batch_norm_layer(ns.batch_norm_d, bn_features)
            if layer is None:
                raise LayerError(f'Bad layer params: D={ns.batch_norm_d}, ' +
                                 f'features={out_features}')
            bnorm_layers.append(layer)

    def get_linear_layers(self) -> Tuple[nn.Module]:
        """Return all linear layers.

        """
        return tuple(self.lin_layers)

    def get_batch_norm_layers(self) -> Tuple[nn.Module]:
        """Return all batch normalize layers.

        """
        if self.bnorm_layers is not None:
            return tuple(self.bnorm_layers)

    @property
    def out_features(self) -> int:
        """The number of features output from all layers of this module.

        """
        n_layers = len(self.get_linear_layers())
        return self.n_features_after_layer(n_layers - 1)

    def n_features_after_layer(self, nth_layer) -> int:
        """Get the output features of the Nth (0 index based) layer.

        :param nth_layer: the layer to use for getting the output features

        """
        return self.get_linear_layers()[nth_layer].out_features

    def forward_n_layers(self, x: torch.Tensor, n_layers: int,
                         full_forward: bool = False) -> torch.Tensor:
        """Forward throught the first 0 index based N layers.

        :param n_layers: the number of layers to forward through (0-based
                         index)

        :param full_forward: if ``True``, also return the full forward as a
                             second parameter

        :return: the tensor output of all layers or a tuple of ``(N-th layer,
                 all layers)``

        """
        return self._forward(x, n_layers, full_forward)

    def _forward(self, x: torch.Tensor,
                 n_layers: int = sys.maxsize,
                 full_forward: bool = False) -> torch.Tensor:
        lin_layers = self.get_linear_layers()
        bnorm_layers = self.get_batch_norm_layers()
        n_layers = min(len(lin_layers) - 1, n_layers)
        x_ret = None

        if self.logger.isEnabledFor(logging.DEBUG):
            self._debug(f'linear: num layers: {len(lin_layers)}')

        for i, layer in enumerate(lin_layers):
            x = layer(x)
            self._shape_debug('deep linear', x)
            x = self._forward_dropout(x)
            if bnorm_layers is not None:
                blayer = bnorm_layers[i]
                if self.logger.isEnabledFor(logging.DEBUG):
                    self._debug(f'batch norm: {blayer}')
                x = blayer(x)
            x = self._forward_activation(x)
            if i == n_layers:
                x_ret = x
                if self.logger.isEnabledFor(logging.DEBUG):
                    self._debug(f'reached {i}th layer = n_layers')
                    self._shape_debug('x_ret', x_ret)
                if not full_forward:
                    self._debug('breaking')
                    break

        if full_forward:
            return x_ret, x
        else:
            return x
