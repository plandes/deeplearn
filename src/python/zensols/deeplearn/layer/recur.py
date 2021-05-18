"""This file contains a convenience wrapper around RNN, GRU and LSTM modules in
PyTorch.

"""
__author__ = 'Paul Landes'

from typing import Union, Tuple
from dataclasses import dataclass, field
import logging
import torch
from torch import Tensor
from zensols.config import ClassImporter
from zensols.deeplearn import DropoutNetworkSettings
from zensols.deeplearn.model import BaseNetworkModule
from . import LayerError


@dataclass
class RecurrentAggregationNetworkSettings(DropoutNetworkSettings):
    """Settings for a recurrent neural network.  This configures a
    :class:`.RecurrentAggregation` layer.

    """
    network_type: str = field()
    """One of ``rnn``, ``lstm`` or ``gru``."""

    aggregation: str = field()
    """A convenience operation to aggregate the parameters; this is one of:
    ``max``: return the max of the output states ``ave``: return the average of
    the output states ``last``: return the last output state ``none``: do not
    apply an aggregation function.

    """

    bidirectional: bool = field()
    """Whether or not the network is bidirectional."""

    input_size: int = field()
    """The input size to the network."""

    hidden_size: int = field()
    """The size of the hidden states of the network."""

    num_layers: int = field()
    """The number of *"stacked"* layers."""

    def get_module_class_name(self) -> str:
        return __name__ + '.RecurrentAggregation'


class RecurrentAggregation(BaseNetworkModule):
    """A recurrent neural network model with an output aggregation.  This includes
    RNNs, LSTMs and GRUs.

    """
    MODULE_NAME = 'recur'

    def __init__(self, net_settings: RecurrentAggregationNetworkSettings,
                 sub_logger: logging.Logger = None):
        """Initialize the recurrent layer.

        :param net_settings: the reccurent layer configuration

        :param sub_logger: the logger to use for the forward process in this
                           layer

        """
        super().__init__(net_settings, sub_logger)
        ns = net_settings
        self.logger.info(f'creating {ns.network_type} network')
        class_name = f'torch.nn.{ns.network_type.upper()}'
        ci = ClassImporter(class_name, reload=False)
        hidden_size = ns.hidden_size // (2 if ns.bidirectional else 1)
        param = {'input_size': ns.input_size,
                 'hidden_size': hidden_size,
                 'num_layers': ns.num_layers,
                 'bidirectional': ns.bidirectional,
                 'batch_first': True}
        if ns.num_layers > 1 and ns.dropout is not None:
            param['dropout'] = ns.dropout
        self.rnn = ci.instance(**param)
        self.dropout = ns.dropout_layer

    def deallocate(self):
        super().deallocate()
        if hasattr(self, 'rnn'):
            del self.rnn

    @property
    def out_features(self) -> int:
        """The number of features output from all layers of this module.

        """
        ns = self.net_settings
        return ns.hidden_size

    def _forward(self, x: Tensor, x_init: Tensor = None) -> \
            Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        if x_init is None:
            x, hidden = self.rnn(x)
        else:
            x, hidden = self.rnn(x, x_init)
        self._shape_debug('recur', x)
        agg = self.net_settings.aggregation
        if agg == 'max':
            x = torch.max(x, dim=1)[0]
            self._shape_debug('max out shape', x)
        elif agg == 'ave':
            x = torch.mean(x, dim=1)
            self._shape_debug('ave out shape', x)
        elif agg == 'last':
            x = x[:, -1, :]
            self._shape_debug('last out shape', x)
        elif agg == 'none':
            pass
        else:
            raise LayerError(f'Unknown aggregate function: {agg}')
        self._shape_debug('aggregation', x)
        return x, hidden
