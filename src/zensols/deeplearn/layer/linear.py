"""Convenience classes for linear layers.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Dict, List, Any, Union
from dataclasses import dataclass, field
import logging
import sys
from torch import nn
from torch import Tensor
from zensols.deeplearn import (
    ActivationNetworkSettings,
    DropoutNetworkSettings,
    BatchNormNetworkSettings,
)
from zensols.deeplearn.model import BaseNetworkModule
from . import LayerError

logger = logging.getLogger(__name__)


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

    out_features: Union[int, Dict[str, Any]] = field()
    """The number of features as output from the last layer.  If a dictionary,
    it follows the same rules as :obj:`middle_features`.

    """
    middle_features: Tuple[Union[int, float, Dict[str, Any]], ...] = field()
    """The number of features in the middle layers; if ``proportions`` is
    ``True``, then each number is how much to grow or shrink as a percetage of
    the last layer, otherwise, it's the number of features.

    If any element is a dictionary, then it iterprets the keys as:

      * ``value``: the value as if the entry was a number, and defaults to 1

      * ``apply``: a sequence of strings indicating the order or the layers to
                   apply with default ``linear, bnorm, activation, dropout``; if
                   a layer is omitted it won't be applied

      * ``batch_norm_features``: the number of features to use in a batch, which
                                 might change based on ordering or ``last`` to
                                 use the last number of parameters computed in
                                 the deep linear network; otherwise it is
                                 computed as the size of the current linear
                                 input

    """
    proportions: bool = field()
    """Whether or not to interpret ``middle_features`` as a proportion of the
    previous layer or use directly as the size of the middle layer.

    """
    repeats: int = field()
    """The number of repeats of the :obj:`middle_features` configuration."""

    def get_module_class_name(self) -> str:
        return __name__ + '.DeepLinear'

    def __str__(self) -> str:
        return f'linear: {self.in_features} -> {self.middle_features}'


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
      5. dropout
      6. linear
      7. batch normalization
      8. activation
      9. dropout

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
        def proc_apply(mf: Tuple[Union[int, float, Dict[str, Any]], ...],
                       repeats: int, last_feat: int, use_portions: bool) -> int:
            for i in range(repeats):
                feat_val: Union[int, float]
                if isinstance(mf, dict):
                    apply_conf = dict(mf)
                    feat_val = apply_conf.get('value', 1)
                else:
                    apply_conf = {}
                    feat_val = mf
                if 'apply' not in apply_conf:
                    apply_conf['apply'] = default_applies
                if use_portions:
                    next_feat = int(last_feat * feat_val)
                else:
                    next_feat = int(feat_val)
                bn_feat: Union[str, int] = mf
                if not isinstance(bn_feat, int):
                    bn_feat = mf.get('batch_norm_features', next_feat)
                if bn_feat == 'last':
                    bn_feat = last_feat
                self._add_layer(last_feat, next_feat, bn_feat,
                                lin_layers, bnorm_layers)
                apply_confs.append(apply_conf)
                last_feat = next_feat
                return last_feat

        ns = net_settings
        # skip the singleton batch norm layer creation; instead this class has
        # one for each configured linear layer
        batch_norm_d, batch_norm_features = \
            ns.batch_norm_d, ns.batch_norm_features
        ns.batch_norm_d, ns.batch_norm_features = None, None
        super().__init__(net_settings, sub_logger)
        ns.batch_norm_d, ns.batch_norm_features = \
            batch_norm_d, batch_norm_features
        last_feat: int = ns.in_features
        lin_layers: List[nn.Linear] = []
        bnorm_layers: List[nn.Module] = []
        apply_confs: List[Dict[str, Any]] = []
        default_applies: List[str] = 'linear bnorm activation dropout'.split()
        for mf in ns.middle_features:
            last_feat = proc_apply(mf, ns.repeats, last_feat, ns.proportions)
        if ns.out_features is not None:
            if isinstance(ns.out_features, int):
                mf = {'apply': default_applies, 'value': ns.out_features}
            else:
                mf = ns.out_features
            proc_apply(mf, 1, last_feat, False)
        self.lin_layers = nn.Sequential(*lin_layers)
        if len(bnorm_layers) > 0:
            self.bnorm_layers = nn.Sequential(*bnorm_layers)
        else:
            self.bnorm_layers = None
        self._apply_confs = apply_confs

    def deallocate(self):
        super().deallocate()
        if hasattr(self, 'lin_layers'):
            del self.lin_layers
        if hasattr(self, 'bnorm_layers') and self.bnorm_layers is not None:
            del self.bnorm_layers

    def _add_layer(self, in_features: int, out_features: int, bn_features: int,
                   lin_layers: List[nn.Linear], bnorm_layers: List[nn.Module]):
        ns = self.net_settings
        n_layer = len(lin_layers)
        if self.logger.isEnabledFor(logging.DEBUG):
            self._debug(f'add {n_layer}: in={in_features} out={out_features}')
        lin_layer = nn.Linear(
            in_features, out_features,
            dtype=self.net_settings.torch_config.data_type)
        lin_layers.append(lin_layer)
        if ns.batch_norm_d is not None:
            if bn_features is None:
                raise LayerError('Bad out features')
            layer = ns.create_batch_norm_layer(ns.batch_norm_d, bn_features)
            if layer is None:
                raise LayerError(f'Bad layer params: D={ns.batch_norm_d}, ' +
                                 f'features={bn_features}')
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

    def forward_n_layers(self, x: Tensor, n_layers: int,
                         full_forward: bool = False) -> Tensor:
        """Forward throught the first 0 index based N layers.

        :param n_layers: the number of layers to forward through (0-based
                         index)

        :param full_forward: if ``True``, also return the full forward as a
                             second parameter

        :return: the tensor output of all layers or a tuple of ``(N-th layer,
                 all layers)``

        """
        return self._forward(x, n_layers, full_forward)

    def _forward(self, x: Tensor, n_layers: int = sys.maxsize,
                 full_forward: bool = False) -> Tensor:
        lin_layers: List[nn.Linear] = self.get_linear_layers()
        bnorm_layers: List[nn.Module] = self.get_batch_norm_layers()
        n_layers: int = min(len(lin_layers) - 1, n_layers)
        x_ret: Tensor = None
        if self.logger.isEnabledFor(logging.DEBUG):
            self._debug(f'linear: num layers: {len(lin_layers)}')
            self._debug(f'layer in features: {self.net_settings.in_features}')
        self._shape_debug('input', x)
        i: int
        layer: nn.Linear
        ac: Dict[str, Any]
        for i, (layer, ac) in enumerate(zip(lin_layers, self._apply_confs)):
            ap: str
            for ap in ac['apply']:
                if 'linear' == ap:
                    x = layer(x)
                    self._shape_debug('linear', x)
                elif 'bnorm' == ap:
                    if bnorm_layers is not None:
                        blayer = bnorm_layers[i]
                        if self.logger.isEnabledFor(logging.DEBUG):
                            self._debug(f'batch norm: {blayer}')
                        x = blayer(x)
                elif 'activation' == ap:
                    x = self._forward_activation(x)
                elif 'dropout' == ap:
                    x = self._forward_dropout(x)
                else:
                    raise LayerError(f'Unkonwn apply: {ap}')
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

    def __str__(self) -> str:
        ns = self.net_settings
        return (f'in: {ns.in_features}, ' +
                f'middle: {ns.middle_features}, ' +
                f'out: {ns.out_features}')
