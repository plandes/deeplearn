"""This file contains classes that configure the network and classifier runs.

"""
__author__ = 'Paul Landes'

import logging
import sys
from dataclasses import dataclass, field
import torch.nn.functional as F
from zensols.deeplearn import TorchConfig

logger = logging.getLogger(__name__)


class EarlyBailException(Exception):
    """Convenience used for helping debug the network.

    """
    def __init__(self):
        super().__init__('early bail to debug the network')


@dataclass
class NetworkSettings(object):
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
class NetworkModelSettings(object):
    """Settings on a classifier.

    :param net_class: the name of the network class used in the classifier,
                      which is used for creating and instance of the class and
                      for reporting

    :param learning_rate: learning_rate used for the gradient descent step
                          (done in the optimzer)
    :param epochs: the number of epochs to train the network
    :param console: if ``True`` create a nice progress bar with training status

    """
    model_path_format: str
    results_path_format: str
    net_class: type
    learning_rate: float
    epochs: int
    batch_limit: int = field(default=sys.maxsize)
    batch_iteration: str = field(default=False)
    use_gc: bool = field(default=False)
    console: bool = field(default=True)
    use_arg_model: bool = field(default=True)

    @property
    def model_type(self):
        return self.net_class.__name__
