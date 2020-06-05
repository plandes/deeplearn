"""This file contains the .

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass
import logging
import torch
from torch import nn
from zensols.persist import Deallocatable
from zensols.config import ClassImporter
from zensols.deeplearn import NetworkSettings
from zensols.deeplearn.model import BaseNetworkModule

logger = logging.getLogger(__name__)


@dataclass
class RecurrentAggregationNetworkSettings(NetworkSettings):
    """Settings for a recurrent neural network

    :param network_type: one of ``rnn``, ``lstm`` or ``gru``

    :param aggregation: one of:
                        ``max``: return the max of the output states
                        ``ave``: return the average of the output states
                        ``last``: return the last output state

    :param bidirectional: whether or not the network is bidirectional

    :param intput_size: the input size to the network

    :param hidden_size: the size of the hidden states of the network

    :param num_layers: the number of *"stacked"* layers

    """
    network_type: str
    aggregation: str
    bidirectional: bool
    input_size: int
    hidden_size: int
    num_layers: int

    def get_module_class_name(self) -> str:
        return __name__ + '.RecurrentAggregation'


class RecurrentAggregation(BaseNetworkModule, Deallocatable):
    """A recurrent neural network model with an output aggregation

    """
    def __init__(self, net_settings: RecurrentAggregationNetworkSettings,
                 mod_logger: logging.Logger = None):
        super().__init__(net_settings, mod_logger)
        ns = net_settings
        logger.info(f'creating {ns.network_type} network')
        class_name = f'torch.nn.{ns.network_type.upper()}'
        ci = ClassImporter(class_name)
        self.rnn = ci.instance(ns.input_size, ns.hidden_size, ns.num_layers,
                               bidirectional=ns.bidirectional,
                               batch_first=True)
        self.dropout = None if ns.dropout is None else nn.Dropout(ns.dropout)
        self.activation = ns.activation_function

    def deallocate(self):
        super().deallocate()
        if hasattr(self, 'rnn'):
            del self.rnn

    def _forward(self, x):
        logger.debug(f'in shape: {x.shape}')
        x = self.rnn(x)[0]
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
        else:
            raise ValueError(f'unknown aggregate function: {agg}')
        if self.dropout is not None:
            x = self.dropout(x)
            self._shape_debug('dropout', x)
        if self.activation:
            x = self.activation(x)
            self._shape_debug('activation', x)
        return x
