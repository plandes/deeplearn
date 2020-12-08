"""This file contains classes that configure the network and classifier runs.

"""
__author__ = 'Paul Landes'

from typing import Any, Dict
from dataclasses import dataclass, field, InitVar
from abc import ABCMeta, abstractmethod
import sys
import logging
from pathlib import Path
import torch.nn.functional as F
from torch import nn
from zensols.config import Writeback
from zensols.persist import persisted, PersistableContainer

logger = logging.getLogger(__name__)


class EarlyBailException(Exception):
    """Convenience used for helping debug the network.

    """
    def __init__(self):
        super().__init__('early bail to debug the network')


@dataclass
class NetworkSettings(Writeback, PersistableContainer, metaclass=ABCMeta):
    """A container settings class for network models.  This abstract class must
    return the fully qualified (with module name) PyTorch `model
    (`torch.nn.Module``) that goes along with these settings.  An instance of
    this class is saved in the model file and given back to it when later
    restored.

    **Note**: Instances of this class are pickled as parts of the results in
    :class:`zensols.deeplearn.result.domain.ModelResult`, so they must be able
    to serialize.  However, they are not used to restore the executor or model,
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


@dataclass
class ActivationNetworkSettings(NetworkSettings):
    """A network settings that contains a activation setting and creates a
    activation layer.

    :param activation: the function between all layers, or ``None`` for no
                       activation

    """
    activation: float

    def _set_option(self, name: str, value: Any):
        super()._set_option(name, value)
        if name == 'activation' and hasattr(self, '_activation_function'):
            self._activation_function.clear()

    @property
    @persisted('_activation_function', transient=True)
    def activation_function(self):
        return self.get_activation_function(self.activation)

    @staticmethod
    def get_activation_function(activation: str):
        if activation == 'relu':
            activation = F.relu
        elif activation == 'leaky_relu':
            activation = F.leaky_relu
        elif activation == 'softmax':
            activation = F.softmax
        elif activation == 'sigmoid':
            activation = nn.Sigmoid()
        elif activation is None:
            activation = None
        else:
            raise ValueError(f'known activation function: {activation}')
        return activation

    def __str__(self):
        return f'{super().__str__()},  activation={self.activation}'


@dataclass
class DropoutNetworkSettings(NetworkSettings):
    """A network settings that contains a dropout setting and creates a dropout
    layer.

    :param dropout: the droput used in all layers or ``None`` to disable

    """
    dropout: float

    def _set_option(self, name: str, value: Any):
        super()._set_option(name, value)
        if name == 'dropout' and hasattr(self, '_dropout_layer'):
            self._dropout_layer().p = value

    @property
    @persisted('_dropout_layer', transient=True)
    def dropout_layer(self):
        if self.dropout is not None:
            return nn.Dropout(self.dropout)


@dataclass
class BatchNormNetworkSettings(NetworkSettings):
    """A network settings that contains a batchnorm setting and creates a batchnorm
    layer.

    :param batch_norm_d: the dimension of the batch norm or ``None`` to disable

    :param batchnorm: the droput used in all layers or ``None`` to disable

    """
    batch_norm_d: int
    batch_norm_features: int

    @staticmethod
    def create_batch_norm_layer(batch_norm_d: int, batch_norm_features: int):
        cls = {None: None,
               1: nn.BatchNorm1d,
               2: nn.BatchNorm2d,
               3: nn.BatchNorm3d}[batch_norm_d]
        if cls is not None:
            if batch_norm_features is None:
                raise ValueError('missing batch norm features')
            return cls(batch_norm_features)

    def create_new_batch_norm_layer(self, batch_norm_d: int = None,
                                    batch_norm_features: int = None):
        if batch_norm_d is None:
            batch_norm_d = self.batch_norm_d
        if batch_norm_features is None:
            batch_norm_features = self.batch_norm_features
        return self.create_batch_norm_layer(batch_norm_d, batch_norm_features)

    @property
    @persisted('_batch_norm_layer', transient=True)
    def batch_norm_layer(self):
        return self.create_new_batch_norm_layer()


@dataclass
class ModelSettings(Writeback, PersistableContainer):
    """This configures and instance of
    :class:`zensols.deeplearn.model.executor.ModelExecutor`.  This differes
    from :class:`.NetworkSettings` in that it configures the model parameters,
    and not the neural network parameters.

    Another reason for these two separate classes is data in this class is not
    needed to rehydrate an instance of ``torch.nn..Module``.

    The loss function strategy across parameters ``nominal_labels``,
    ``criterion_class`` and ``optimizer_class``, must be consistent.  The
    defaults uses nominal labels, which means a single integer, rather than one
    hot encoding, is used for the labels.  Most loss function, including the
    default :class:`nn.CrossEntropyLoss`` uses nominal labels.  The optimizer
    defaults to :class:`torch.nn.Adam`.

    However, if ``nominal_labels`` is set to ``False``, it is expected that the
    label output is a ``Long`` one hot encoding of the class label that must be
    decoded with :meth:`_decode_outcomes` and uses a loss function such as
    :class:`torch.nn.BCEWithLogitsLoss`, which applies a softmax over the
    output to narow to a nominal.

    If the ``criterion_class`` is left as the default, the class the
    corresponding class across these two is selected based on
    ``nominal_labels``.

    **Note**: Instances of this class are pickled as parts of the results in
    :class:`zensols.deeplearn.result.domain.ModelResult`, so they must be able
    to serialize.  However, they are not used to restore the executor or model,
    which are instead, recreated from the configuration for each (re)load (see
    the package documentation for more information).

    :param path: the path to save and load the model

    :param learning_rate: learning_rate used for the gradient descent step
                          (done in the optimzer)
    :param epochs: the number of epochs to train the network

    :param max_consecutive_increased_count:

      the maximum number of times the validation loss can increase per epoch
      before the executor "gives up" and early stops training

    :param nominal_labels: ``True`` if using numbers to identify the class as
                           an enumeration rather than a one hot encoded array

    :param criterion_class_name: the loss function class name (see class doc)

    :param optimizer_class_name: the optimization algorithm class name (see
                                 class doc)

    :param scheduler_class_name:

        the fully qualified class name of the learning rate scheduler used for
        the optimizer (if not ``None``) such as:
        :class:`torch.optim.lr_scheduler.StepLR` or
        :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`

    :param scheduler_class_params: the parameters given to the scheduler's
                                   initializer (see ``scheduler_class_name``)

    :param reduce_outcomes:

        The method by which the labels and output is reduced.  The output is
        optionally reduced, which is one of the following:

        * ``argmax``: uses the index of the largest value,
          which is used for classification models and the
          default
        * ``softmax``: just like ``argmax`` but applies a
          softmax
        * ``none``: return the identity

    :param batch_limit: the max number of batches to train, validate and test
                        on, which is useful for limiting while debuging;
                        defaults to `sys.maxsize`.

    :param batch_iteration:

        how the batches are buffered, which is one of:

        * ``gpu``, buffers all data in the GPU
        * ``cpu``, which means keep all batches in CPU memory (the default)
        * ``buffered`` which means to buffer only one batch at a time (only
          for *very* large data)

    :param gc_level: the frequency by with the garbage collector is invoked:

         * 0: never
         * 1: before and after training or testing
         * 2: after each epoch
         * 3: after each batch

    :see: :class:`.NetworkSettings`

    """
    path: Path
    learning_rate: float
    epochs: int
    max_consecutive_increased_count: int = field(default=sys.maxsize)
    nominal_labels: bool = field(default=True)
    batch_iteration_class_name: InitVar[str] = field(default=None)
    criterion_class_name: InitVar[str] = field(default=None)
    optimizer_class_name: InitVar[str] = field(default=None)
    scheduler_class_name: str = field(default=None)
    scheduler_params: Dict[str, Any] = field(default=None)
    reduce_outcomes: str = field(default='argmax')
    shuffle_training: bool = field(default=False)
    batch_limit: int = field(default=sys.maxsize)
    batch_iteration: str = field(default='cpu')
    cache_batches: bool = field(default=True)
    gc_level: int = field(default=0)

    def __post_init__(self,
                      batch_iteration_class_name: str,
                      criterion_class_name: str,
                      optimizer_class_name: str):
        if batch_iteration_class_name is None:
            self.batch_iteration_class_name = 'zensols.deeplearn.model.BatchIterator'
        else:
            self.batch_iteration_class_name = batch_iteration_class_name
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
