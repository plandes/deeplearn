"""Contains an implementation of a recurrent with a conditional random field
layer.  This is usually configured as a BiLSTM CRF.

"""
__author__ = 'Paul Landes'


from typing import Tuple
from dataclasses import dataclass
import logging
from torch import nn
from torch import Tensor
from zensols.deeplearn import (
    ActivationNetworkSettings,
    DropoutNetworkSettings,
    BatchNormNetworkSettings,
)
from zensols.deeplearn.model import BaseNetworkModule
from . import (
    RecurrentAggregation,
    RecurrentAggregationNetworkSettings,
    DeepLinearNetworkSettings,
)
from . import CRF, DeepLinear


@dataclass
class RecurrentCRFNetworkSettings(ActivationNetworkSettings,
                                  DropoutNetworkSettings,
                                  BatchNormNetworkSettings):
    """Settings for a recurrent neural network using :class:`.RecurrentCRF`.

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
    score_reduction: str

    def to_recurrent_aggregation(self) -> RecurrentAggregationNetworkSettings:
        attrs = ('name config_factory dropout network_type bidirectional ' +
                 'input_size hidden_size num_layers')
        params = {k: getattr(self, k) for k in attrs.split()}
        params['aggregation'] = 'none'
        return RecurrentAggregationNetworkSettings(**params)

    def get_module_class_name(self) -> str:
        return __name__ + '.RecurrentCRF'


class RecurrentCRF(BaseNetworkModule):
    """Adapt the :class:`.CRF` module using the framework based
    :class:`.BaseNetworkModule` class.  This provides methods
    :meth:`forward_recur_decode` and :meth:`decode` which

    """
    MODULE_NAME = 'recur crf'

    def __init__(self, net_settings: RecurrentCRFNetworkSettings,
                 sub_logger: logging.Logger = None):
        """Initialize the reccurent CRF layer.

        :param net_settings: the recurrent layer configuration

        :param logger: the logger to use for the forward process in this layer

        """
        super().__init__(net_settings, sub_logger)
        ns = self.net_settings
        rs = ns.to_recurrent_aggregation()
        self.logger.debug(f'recur settings: {rs}')
        self.hidden_dim = rs.hidden_size
        self.recur = RecurrentAggregation(rs, sub_logger)
        if ns.decoder_settings is None:
            self.decoder = nn.Linear(rs.hidden_size, ns.num_labels)
        else:
            ln = ns.decoder_settings
            ln.in_features = rs.hidden_size
            ln.out_features = ns.num_labels
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f'linear: {ln}')
            self.decoder = DeepLinear(ln)
        self.crf = CRF(ns.num_labels, batch_first=True,
                       score_reduction=ns.score_reduction)
        self.crf.reset_parameters()
        self.hidden = None

    def deallocate(self):
        super().deallocate()
        self.decoder.deallocate()
        self.recur.deallocate()

    def forward_recur_decode(self, x: Tensor) -> Tensor:
        self._shape_debug('recur in', x)
        x = self.recur(x)[0]
        # don't apply dropout since the recur last layer already has when
        # configured
        x = self._forward_batch_norm(x)
        x = self._forward_activation(x)
        x = self.decoder(x)
        self._shape_debug('decode', x)
        return x

    def _forward(self, x: Tensor, mask: Tensor, labels: Tensor) -> Tensor:
        self._shape_debug('mask', mask)
        self._shape_debug('labels', labels)
        x = self.forward_recur_decode(x)
        x = -self.crf(x, labels, mask=mask)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f'training loss: {x}')
        return x

    def decode(self, x: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        self._shape_debug('mask', mask)
        x = self.forward_recur_decode(x)
        seq, score = self.crf.decode(x, mask=mask)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f'decoded: {seq.shape}, score: {score.shape}')
        return seq, score
