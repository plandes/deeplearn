"""This file contains classes that configure the network and classifier runs.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field
from abc import ABC, abstractmethod, ABCMeta
import sys
import logging
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from zensols.deeplearn import TorchConfig, Batch

logger = logging.getLogger(__name__)


class EarlyBailException(Exception):
    """Convenience used for helping debug the network.

    """
    def __init__(self):
        super().__init__('early bail to debug the network')


@dataclass
class NetworkSettings(ABC):
    """A utility container settings class for network models.

    :param sentence_length: the number of tokens a window, which is also the
                        number of time steps in the recurrent neural
                        network
    :param debug: if ``True``, raise an error on the first forward pass
    :param activation: if ``True`` use a rectified linear activation function
    :param dropout: if not ``None``, add a dropout on the fully connected
                    layer

    """
    torch_config: TorchConfig
    dropout: float
    activation: str
    debug: bool

    @abstractmethod
    def get_module_class_name(self) -> str:
        pass

    @property
    def activation_function(self):
        if self.activation == 'relu':
            activation = F.relu
        elif self.activation == 'softmax':
            activation = F.softmax
        else:
            activation = None
        return activation

    def __str__(self):
        return f'{super().__str__()},  activation={self.activation}'


@dataclass
class ModelSettings(object):
    """Settings on a classifier.

    :param learning_rate: learning_rate used for the gradient descent step
                          (done in the optimzer)
    :param epochs: the number of epochs to train the network

    :param batch_iteration: how the batches are buffered; one of ``gpu``, which
                            buffers all data in the GPU, ``cpu``, which means
                            keep all batches in CPU memory (the default), or
                            ``buffered`` which means to buffer only one batch
                            at a time (only for *very* large data)

    :param console: if ``True`` create a nice progress bar with training status

    """
    path: Path
    learning_rate: float
    epochs: int
    batch_limit: int = field(default=sys.maxsize)
    batch_iteration: str = field(default='cpu')
    use_gc: bool = field(default=True)


class BaseNetworkModule(nn.Module, metaclass=ABCMeta):
    """A recurrent neural network model that is used to classify sentiment.

    """
    def __init__(self, net_settings: NetworkSettings):
        super().__init__()
        self.net_settings = net_settings

    @abstractmethod
    def _forward(self, batch: Batch) -> torch.Tensor:
        pass

    def forward(self, batch: Batch):
        x = self._forward(batch)
        if self.net_settings.debug:
            raise EarlyBailException()
        return x

    def _shape_debug(self, msg, x):
        if self.net_settings.debug:
            logger.debug(f'{msg}: x: {x.shape}')
