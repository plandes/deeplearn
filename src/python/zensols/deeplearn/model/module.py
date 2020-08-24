"""Base class PyTorch module and utilities.

"""
__author__ = 'Paul Landes'

from typing import Union
from abc import abstractmethod, ABCMeta
import logging
from torch import nn
from torch import Tensor
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
    DEBUG_DEVICE = False
    DEBUG_SHAPE = False
    MODULE_NAME = None

    def __init__(self, net_settings: NetworkSettings,
                 sub_logger: logging.Logger = None):
        super().__init__()
        self.net_settings = ns = net_settings
        if sub_logger is None:
            self.logger = logger
        else:
            self.logger = sub_logger
        if isinstance(ns, DropoutNetworkSettings):
            self.dropout = ns.dropout_layer
        else:
            self.dropout = None
        if isinstance(ns, BatchNormNetworkSettings) and \
           ns.batch_norm_d is not None and \
           ns.batch_norm_features is not None:
            self.batch_norm = ns.batch_norm_layer
        else:
            self.batch_norm = None
        if isinstance(ns, ActivationNetworkSettings):
            self.activation_function = ns.activation_function
        else:
            self.activation_function = None

    def _deallocate_children_modules(self):
        for layer in self.children():
            self._try_deallocate(layer)

    def __getstate__(self):
        raise ValueError('layers should not be pickeled')

    @abstractmethod
    def _forward(self, x: Union[Batch, Tensor], *args, **kwargs) -> Tensor:
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

    def _forward_dropout(self, x: Tensor) -> Tensor:
        """Forward the dropout if there is one configured.

        """
        if self.dropout is not None:
            if self.logger.isEnabledFor(logging.DEBUG):
                self._debug(f'dropout: {self.dropout}')
            x = self.dropout(x)
        return x

    def _forward_batch_norm(self, x: Tensor) -> Tensor:
        """Forward the batch normalization if there is one configured.

        """
        if self.batch_norm is not None:
            if self.logger.isEnabledFor(logging.DEBUG):
                self._debug(f'batch norm: {self.batch_norm}')
            x = self.batch_norm(x)
        return x

    def _forward_activation(self, x: Tensor) -> Tensor:
        """Transform using the activation function if there is one configured.

        """
        if self.activation_function is not None:
            if self.logger.isEnabledFor(logging.DEBUG):
                self._debug(f'activation: {self.activation_function}')
            x = self.activation_function(x)
        return x

    def _forward_drop_batch_act(self, x: Tensor) -> Tensor:
        """Forward dropout, batch norm, and activation, in this order for those layers
        that are configured.

        """
        x = self._forward_dropout(x)
        x = self._forward_batch_norm(x)
        x = self._forward_activation(x)
        return x

    def forward(self, x: Union[Batch, Tensor], *args, **kwargs) -> Tensor:
        """Main forward takes a batch for top level modules, or a tensor for framework
        based layers.  Return the transformed tensor.

        """
        if self.logger.isEnabledFor(logging.DEBUG) and isinstance(x, Batch):
            self._debug(f'input batch: {x}')
        return self._forward(x, *args, **kwargs)

    def _debug(self, msg: str):
        """Debug a message using the module name in the description.

        """
        if self.logger.isEnabledFor(logging.DEBUG):
            mname = self.MODULE_NAME
            if mname is None:
                mname = self.__class__.__name__
            self.logger.debug(f'[{mname}] {msg}')

    def _shape_debug(self, msg: str, x: Tensor):
        if self.logger.isEnabledFor(logging.DEBUG):
            if x is None:
                shape, device, dtype = [None] * 3
            else:
                shape, device, dtype = x.shape, x.device, x.dtype
            msg = f'{msg} shape: {shape}'
            if self.DEBUG_DEVICE:
                msg += f', device: {device}'
            if self.DEBUG_SHAPE:
                msg += f', type: {dtype}'
            self._debug(msg)


class ScoredNetworkModule(BaseNetworkModule):
    """A module that has a forward training pass and a separate **scoring** phase.
    Examples include layers with an ending linear CRF layer, such as a BiLSTM
    CRF.  This module has a ``decode`` method that returns a 2D list of integer
    label indexes of a nominal class.

    :see: :class:`zensols.deeplearn.layer.RecurrentCRFNetwork`

    """
    @abstractmethod
    def _score(self, batch: Batch) -> Tensor:
        pass

    def score(self, batch: Batch) -> Tensor:
        return self._score(batch)

    def get_loss(self, batch: Batch) -> Tensor:
        if self.training:
            raise ValueError('resolving a loss while training results ' +
                             'in the model training on the valdiation set--' +
                             'use model.eval() first')
        return self.forward(batch)
