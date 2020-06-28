"""Base class PyTorch module and utilities.

"""
__author__ = 'Paul Landes'

from abc import abstractmethod, ABCMeta
import logging
import torch
from torch import nn
from zensols.deeplearn import (
    NetworkSettings,
    BasicNetworkSettings,
    EarlyBailException,
)
from zensols.deeplearn.batch import Batch

logger = logging.getLogger(__name__)


class BaseNetworkModule(nn.Module, metaclass=ABCMeta):
    """A recurrent neural network model that is used to classify sentiment.

    """
    def __init__(self, net_settings: NetworkSettings,
                 sub_logger: logging.Logger = None):
        super().__init__()
        self.net_settings = net_settings
        if sub_logger is None:
            self.logger = logger
        else:
            self.logger = sub_logger
        if isinstance(self.net_settings, BasicNetworkSettings):
            self.activation_function = self.net_settings.activation_function
        else:
            self.activation_function = None

    def __getstate__(self):
        raise ValueError('layers should not be pickeled')

    @abstractmethod
    def _forward(self, batch: Batch) -> torch.Tensor:
        """The model's forward implementation.  Normal backward semantics are no
        different.

        :param batch: the batch to train, validate or test on

        """
        pass

    @property
    def device(self):
        """Return the device on which the model is configured.

        """
        return next(self.parameters()).device

    def _bail(self):
        """A convenience method to assist in debugging.  This is useful when the output
        isn't in the correct form for the :class:`.ModelExecutor`.

        """
        raise EarlyBailException()

    def forward(self, batch: Batch):
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f'input batch: {batch}')
        x = self._forward(batch)
        if self.activation_function is not None:
            x = self.activation_function(x)
        return x

    def _shape_debug(self, msg, x):
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f'{msg} shape: {x.shape}')
