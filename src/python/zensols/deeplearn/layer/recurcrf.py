"""Contains an implementation of a recurrent with a conditional random field
layer.  This is usually configured as a BiLSTM CRF.

"""
__author__ = 'Paul Landes'


from dataclasses import dataclass
import logging
import torch
from torchcrf import CRF
from zensols.deeplearn import (
    ActivationNetworkSettings,
    DropoutNetworkSettings,
    BatchNormNetworkSettings,
)
from zensols.deeplearn.model import BaseNetworkModule
from zensols.deeplearn.batch import Batch
from . import (
    RecurrentAggregation,
    RecurrentAggregationNetworkSettings,
    DeepLinearNetworkSettings,
)

logger = logging.getLogger(__name__)


@dataclass
class RecurrentCRFNetworkSettings(ActivationNetworkSettings,
                                  DropoutNetworkSettings,
                                  BatchNormNetworkSettings):
    """Settings for a recurrent neural network

    :param network_type: one of ``rnn``, ``lstm`` or ``gru`` (usually ``lstm``)

    :param bidirectional: whether or not the network is bidirectional (usually
                          ``True``)

    :param intput_size: the input size to the network

    :param hidden_size: the size of the hidden states of the network

    :param num_layers: the number of *"stacked"* layers

    :param num_labels: the number of output labels from the CRF

    """
    network_type: str
    bidirectional: bool
    input_size: int
    hidden_size: int
    num_layers: int
    num_labels: int
    decoder_settings: DeepLinearNetworkSettings

    def to_recurrent_aggregation(self) -> RecurrentAggregationNetworkSettings:
        attrs = 'name config_factory dropout network_type bidirectional input_size hidden_size num_layers'
        params = {k: getattr(self, k) for k in attrs.split()}
        params['aggregation'] = 'none'
        return RecurrentAggregationNetworkSettings(**params)

    def get_module_class_name(self) -> str:
        return __name__ + '.RecurrentCRF'
