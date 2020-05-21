"""This file contains classes that configure the network and classifier runs.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
import sys
import logging
from pathlib import Path
import torch.nn.functional as F
from zensols.persist import persisted, PersistableContainer
from zensols.deeplearn import TorchConfig

logger = logging.getLogger(__name__)


class EarlyBailException(Exception):
    """Convenience used for helping debug the network.

    """
    def __init__(self):
        super().__init__('early bail to debug the network')


@dataclass
class NetworkSettings(PersistableContainer, metaclass=ABCMeta):
    """A container settings class for network models.  This abstract class must
    return the fully qualified (with module name) PyTorch `model
    (`torch.nn.Module``) that goes along with these settings.  An instance of
    this class is saved in the model file and given back to it when later
    restored.

    :torch_config: the configuration used to copy memory (i.e. GPU) of the
                   model

    :param dropout: if not ``None``, add a dropout on the fully connected
                    layer

    :param activation: if ``True`` use a rectified linear activation function

    :param debug: if ``True``, raise an error on the first forward pass


    :see ModelSettings:

    """
    name: str
    torch_config: TorchConfig
    dropout: float
    activation: str
    debug: bool

    @abstractmethod
    def get_module_class_name(self) -> str:
        pass

    @property
    @persisted('_activation_function', transient=True)
    def activation_function(self):
        return self.get_activation_function(self.activation)

    @staticmethod
    def get_activation_function(activation: str):
        if activation == 'relu':
            activation = F.relu
        elif activation == 'softmax':
            activation = F.softmax
        elif activation is None:
            activation = None
        else:
            raise ValueError(f'known activation function: {activation}')
        return activation

    def __str__(self):
        return f'{super().__str__()},  activation={self.activation}'


@dataclass
class ModelSettings(object):
    """This configures and instance of ``ModelExecutor``.  This differes from
    ``NetworkSettings`` in that it configures the model parameters, and not the
    neural network parameters.

    Another reason for these two separate classes is data in this class is not
    needed to rehydrate an instance of ``torch.nn..Module``.

    :param path: the path to save and load the model

    :param learning_rate: learning_rate used for the gradient descent step
                          (done in the optimzer)
    :param epochs: the number of epochs to train the network

    :param batch_iteration: how the batches are buffered; one of ``gpu``, which
                            buffers all data in the GPU, ``cpu``, which means
                            keep all batches in CPU memory (the default), or
                            ``buffered`` which means to buffer only one batch
                            at a time (only for *very* large data)

    :param use_gc: if ``True``, invoke the garbage collector periodically to
                   reduce memory overhead

    :see NetworkSettings:

    """
    name: str
    path: Path
    learning_rate: float
    epochs: int
    batch_limit: int = field(default=sys.maxsize)
    batch_iteration: str = field(default='cpu')
    use_gc: bool = field(default=False)
