"""Contains an implementation of a recurrent with a conditional random field
layer.  This is usually configured as a BiLSTM CRF.

"""
__author__ = 'Paul Landes'


from typing import Tuple
from dataclasses import dataclass, field
import logging
import torch
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

    """
    network_type: str = field()
    """One of ``rnn``, ``lstm`` or ``gru`` (usually ``lstm``)."""

    bidirectional: bool = field()
    """Whether or not the network is bidirectional (usually ``True``)."""

    input_size: int = field()
    """The input size to the layer."""

    hidden_size: int = field()
    """The size of the hidden states of the network."""

    num_layers: int = field()
    """The number of *"stacked"* layers."""

    num_labels: int = field()
    """The number of output labels from the CRF."""

    decoder_settings: DeepLinearNetworkSettings = field()
    """The decoder feed forward network."""

    score_reduction: str = field()
    """Reduces how the score output over batches.

    :see: :class:`.CRF`

    """

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
    :meth:`forward_recur_decode` and :meth:`decode`, which decodes the input.

    This adds a recurrent neural network and a fully connected feed forward
    decoder layer before the CRF layer.

    """
    MODULE_NAME = 'recur crf'

    def __init__(self, net_settings: RecurrentCRFNetworkSettings,
                 sub_logger: logging.Logger = None):
        """Initialize the reccurent CRF layer.

        :param net_settings: the recurrent layer configuration

        :param sub_logger: the logger to use for the forward process in this
                           layer

        """
        super().__init__(net_settings, sub_logger)
        ns = self.net_settings
        self.recur_settings = rs = ns.to_recurrent_aggregation()
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
        self._zero = None

    def deallocate(self):
        super().deallocate()
        self.decoder.deallocate()
        self.recur.deallocate()
        self.recur_settings.deallocate()

    def forward_recur_decode(self, x: Tensor) -> Tensor:
        """Forward the input through the recurrent network (i.e. LSTM), batch
        normalization and activation (if confgiured), and decoder output.

        :param x: the network input

        :return: the fully connected linear feed forward decoded output

        """
        self._shape_debug('recur in', x)
        x = self.recur(x)[0]
        # don't apply dropout since the recur last layer already has when
        # configured
        x = self._forward_batch_norm(x)
        x = self._forward_activation(x)
        x = self.decoder(x)
        self._shape_debug('decode', x)
        return x

    def to(self, *args, **kwargs):
        self._zero = None
        return super().to(*args, **kwargs)

    def _forward(self, x: Tensor, mask: Tensor, labels: Tensor) -> Tensor:
        self._shape_debug('mask', mask)
        self._shape_debug('labels', labels)
        if self._zero is None:
            self._zero = torch.tensor(
                [0.], dtype=labels.dtype, device=labels.device)
        x = self.forward_recur_decode(x)
        # zero out negative values, since otherwise invalid transitions are
        # indexed with negative values, which come from the default cross
        # entropy loss functions `ignore_index`
        labels = torch.max(labels, self._zero)
        x = -self.crf(x, labels, mask=mask)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f'training loss: {x}')
        return x

    def decode(self, x: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward the input though the recurrent network, decoder, and then the CRF.

        :param x: the input

        :param mask: the mask used to block the last N states not provided

        :return: the CRF sequence output and the score provided by the CRF's
                 veterbi algorithm as a tuple

        """
        self._shape_debug('mask', mask)
        x = self.forward_recur_decode(x)
        seq, score = self.crf.decode(x, mask=mask)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f'decoded: {len(seq)} seqs, score: {score}')
        return seq, score
