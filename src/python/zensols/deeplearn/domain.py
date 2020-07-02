"""This file contains classes that configure the network and classifier runs.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field, InitVar
from typing import Any
from abc import ABCMeta, abstractmethod
import sys
import logging
from pathlib import Path
import torch.nn.functional as F
from io import TextIOBase
from torch import nn
from zensols.config import Writeback, Writable
from zensols.persist import persisted, PersistableContainer

logger = logging.getLogger(__name__)


class EarlyBailException(Exception):
    """Convenience used for helping debug the network.

    """
    def __init__(self):
        super().__init__('early bail to debug the network')


@dataclass
class NetworkSettings(Writeback, PersistableContainer,
                      Writable, metaclass=ABCMeta):
    """A container settings class for network models.  This abstract class must
    return the fully qualified (with module name) PyTorch `model
    (`torch.nn.Module``) that goes along with these settings.  An instance of
    this class is saved in the model file and given back to it when later
    restored.

    **Note**: Instances of this class are pickled as parts of the results in
    :class:`zensols.deeplearn.result.ModelResult`, so they must be able to
    serialize.  However, they are not used to restore the executor or model,
    which are instead, recreated from the configuration for each (re)load (see
    the package documentation for more information).

    :param dropout: if not ``None``, add a dropout on the fully connected
                    layer

    :param activation: if ``True`` use a rectified linear activation function

    :see: :class:`.ModelSettings`

    """
    def __post_init__(self):
        PersistableContainer.__init__(self)

    def _allow_config_adds(self) -> bool:
        return True

    @abstractmethod
    def get_module_class_name(self) -> str:
        pass

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_dict(self.asdict(), depth, writer)


@dataclass
class BasicNetworkSettings(NetworkSettings):
    """A network settings that contains typical hyperparameters, such as dropout
    and an activation function.

    :param dropout: the droput used in all layers or ``None`` to disable

    :param activation: the function between all layers, or ``None`` for no
                       activation

    """
    dropout: float
    activation: str

    def _set_option(self, name: str, value: Any):
        super()._set_option(name, value)
        if name == 'dropout' and hasattr(self, '_dropout_layer'):
            self._dropout_layer().p = value
        elif name == 'activation' and hasattr(self, '_activation_function'):
            self._activation_function.clear()

    @property
    @persisted('_dropout_layer', transient=True)
    def dropout_layer(self):
        if self.dropout is not None:
            return nn.Dropout(self.dropout)

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
class ModelSettings(Writeback, Writable, PersistableContainer):
    """This configures and instance of ``ModelExecutor``.  This differes from
    ``NetworkSettings`` in that it configures the model parameters, and not the
    neural network parameters.

    Another reason for these two separate classes is data in this class is not
    needed to rehydrate an instance of ``torch.nn..Module``.

    The loss function strategy across parameters ``nominal_labels``,
    ``criterion_class`` and ``optimizer_class``, must be consistent.  The
    defaults uses nominal labels, which means a single integer, rather than one
    hot encoding, is used for the labels.  Most loss function, including the
    default :class:`nn.CrossEntropyLoss`` uses nominal labels.

    However, if ``nominal_labels`` is set to ``False``, it is expected that the
    label output is a ``Long`` one hot encoding of the class label that must be
    decoded with :meth:`_decode_outcomes` and uses a loss function such as
    :class:`nn.BCEWithLogitsLoss`, which applies a softmax over the output to
    narow to a nominal.

    If the ``criterion_classs`` is left as the default, the class the
    corresponding class across these two is selected based on
    ``nominal_labels``.

    **Note**: Instances of this class are pickled as parts of the results in
    :class:`zensols.deeplearn.result.ModelResult`, so they must be able to
    serialize.  However, they are not used to restore the executor or model,
    which are instead, recreated from the configuration for each (re)load
    (see the package documentation for more information).

    :param path: the path to save and load the model

    :param learning_rate: learning_rate used for the gradient descent step
                          (done in the optimzer)
    :param epochs: the number of epochs to train the network

    :param nominal_labels: ``True`` if using numbers to identify the class as
                           an enumeration rather than a one hot encoded array

    :param criterion_class_name: the loss function class name (see class doc)

    :param optimizer_class_name: the optimization algorithm class name (see
                                 class doc)

    :param batch_limit: the max number of batches to train, validate and test
                        on, which is useful for limiting while debuging;
                        defaults to `sys.maxsize`.

    :param batch_iteration: how the batches are buffered; one of ``gpu``, which
                            buffers all data in the GPU, ``cpu``, which means
                            keep all batches in CPU memory (the default), or
                            ``buffered`` which means to buffer only one batch
                            at a time (only for *very* large data)

    :param use_gc: if ``True``, invoke the garbage collector periodically to
                   reduce memory overhead

    :see: :class:`.NetworkSettings`

    """
    path: Path
    learning_rate: float
    epochs: int
    nominal_labels: bool = field(default=True)
    criterion_class_name: InitVar[str] = field(default=None)
    optimizer_class_name: InitVar[str] = field(default=None)
    batch_limit: int = field(default=sys.maxsize)
    batch_iteration: str = field(default='cpu')
    cache_batches: bool = field(default=True)
    use_gc: bool = field(default=False)

    def __post_init__(self,
                      criterion_class_name: str,
                      optimizer_class_name: str):
        if criterion_class_name is None:
            if self.nominal_labels:
                self.criterion_class_name = 'torch.nn.CrossEntropyLoss'
            else:
                self.criterion_class_name = 'torch.nn.BCEWithLogitsLoss'
        else:
            self.criterion_class_name = criterion_class_name
        if optimizer_class_name is None:
            self.optimizer_class_name = 'torch.optim.Adam'
        else:
            self.optimizer_class_name = optimizer_class_name

    def _allow_config_adds(self) -> bool:
        return True

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_dict(self.asdict(), depth, writer)
