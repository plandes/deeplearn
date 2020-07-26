"""Base class PyTorch module and utilities.

"""
__author__ = 'Paul Landes'

from typing import Union
from abc import abstractmethod, ABCMeta
import logging
import torch
from torch import nn
from zensols.persist import PersistableContainer
from zensols.deeplearn import (
    NetworkSettings,
    ActivationNetworkSettings,
    DropoutNetworkSettings,
    BatchNormNetworkSettings,
    EarlyBailException,
)
from zensols.deeplearn.batch import Batch

logger = logging.getLogger(__name__)


class BaseNetworkModule(nn.Module, PersistableContainer, metaclass=ABCMeta):
    """A recurrent neural network model that is used to classify sentiment.  This
    can be used for its utility methods, or a as a base class that accepts
    instances of :class:`.Batch`.

    """
    def __init__(self, net_settings: NetworkSettings,
                 sub_logger: logging.Logger = None):
        super().__init__()
        self.net_settings = net_settings
        if sub_logger is None:
            self.logger = logger
        else:
            self.logger = sub_logger
        if isinstance(self.net_settings, DropoutNetworkSettings):
            self.dropout = self.net_settings.dropout_layer
        else:
            self.dropout = None
        if isinstance(self.net_settings, BatchNormNetworkSettings):
            self.batch_norm = self.net_settings.batch_norm_layer
        else:
            self.batch_norm = None
        if isinstance(self.net_settings, ActivationNetworkSettings):
            self.activation_function = self.net_settings.activation_function
        else:
            self.activation_function = None

    def _deallocate_children_modules(self):
        for layer in self.children():
            self._try_deallocate(layer)

    def __getstate__(self):
        raise ValueError('layers should not be pickeled')

    @abstractmethod
    def _forward(self, x: Union[Batch, torch.Tensor]) -> torch.Tensor:
        """The model's forward implementation.  Normal backward semantics are no
        different.

        :param batch: the batch to train, validate or test on

        """
        pass

    @staticmethod
    def device_from_module(module):
        return next(module.parameters()).device

    @property
    def device(self):
        """Return the device on which the model is configured.

        """
        return self.device_from_module(self)

    def _bail(self):
        """A convenience method to assist in debugging.  This is useful when the output
        isn't in the correct form for the :class:`.ModelExecutor`.

        """
        self.logger.debug('-' * 60)
        raise EarlyBailException()

    def _forward_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the dropout if there is one configured.

        """
        if self.dropout is not None:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f'dropout: {self.dropout}')
            x = self.dropout(x)
        return x

    def _forward_batch_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the batch normalization if there is one configured.

        """
        if self.batch_norm is not None:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f'batch norm: {self.batch_norm}')
            x = self.batch_norm(x)
        return x

    def _forward_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Transform using the activation function if there is one configured.

        """
        if self.activation_function is not None:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f'activation: {self.activation_function}')
            x = self.activation_function(x)
        return x

    def _forward_drop_batch_act(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_dropout(x)
        x = self._forward_batch_norm(x)
        x = self._forward_activation(x)
        return x

    def forward(self, x: Union[Batch, torch.Tensor]) -> torch.Tensor:
        if self.logger.isEnabledFor(logging.DEBUG) and isinstance(x, Batch):
            self.logger.debug(f'input batch: {x}')
        return self._forward(x)

    def _shape_debug(self, msg, x):
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f'{msg} shape: {x.shape}, device: {x.device}')
