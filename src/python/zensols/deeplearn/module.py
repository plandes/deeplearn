"""Base PyTorch module and utilities.

"""
__author__ = 'Paul Landes'

from abc import abstractmethod, ABCMeta
import logging
import torch
from torch import nn
from zensols.deeplearn import (
    Batch,
    NetworkSettings,
    EarlyBailException,
)

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

    def forward(self, batch: Batch):
        x = self._forward(batch)
        if self.net_settings.debug:
            raise EarlyBailException()
        return x

    def _shape_debug(self, msg, x):
        if self.net_settings.debug:
            self.logger.debug(f'{msg}: x: {x.shape}')
