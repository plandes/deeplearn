"""Base class PyTorch module and utilities.

"""
__author__ = 'Paul Landes'

from typing import Union
from abc import abstractmethod, ABCMeta
import logging
import torch
from torch import nn
from torch import Tensor
from zensols.persist import PersistableContainer
from zensols.deeplearn import (
    ModelError,
    NetworkSettings,
    ActivationNetworkSettings,
    DropoutNetworkSettings,
    BatchNormNetworkSettings,
    EarlyBailError,
)
from zensols.deeplearn.batch import Batch

logger = logging.getLogger(__name__)


class DebugModule(nn.Module):
    """A utility base class that makes logging more understandable.

    """
    DEBUG_DEVICE = False
    """If ``True``, add tensor devices to log messages."""

    DEBUG_TYPE = False
    """If ``True``, add tensor shapes to log messages."""

    MODULE_NAME = None
    """The module name used in the logging message.  This is set in each inherited
    class.

    """

    def __init__(self, sub_logger: logging.Logger = None):
        """Initialize.

        :param sub_logger: used to log activity in this module so they logged
                           module comes from some parent model

        """
        super().__init__()
        if sub_logger is None:
            self.logger = logger
        else:
            self.logger = sub_logger

    def _debug(self, msg: str):
        """Debug a message using the module name in the description.

        """
        if self.logger.isEnabledFor(logging.DEBUG):
            mname = self.MODULE_NAME
            if mname is None:
                mname = self.__class__.__name__
            self.logger.debug(f'[{mname}] {msg}')

    def _shape_debug(self, msg: str, x: Tensor):
        """Debug a message using the module name in the description and include the
        shape.

        """
        if self.logger.isEnabledFor(logging.DEBUG):
            if x is None:
                shape, device, dtype = [None] * 3
            else:
                shape, device, dtype = x.shape, x.device, x.dtype
            msg = f'{msg} shape: {shape}'
            if self.DEBUG_DEVICE:
                msg += f', device: {device}'
            if self.DEBUG_TYPE:
                msg += f', type: {dtype}'
            self._debug(msg)

    def _bail(self):
        """A convenience method to assist in debugging.  This is useful when the output
        isn't in the correct form for the :class:`.ModelExecutor`.

        """
        self.logger.debug('-' * 60)
        raise EarlyBailError()


class BaseNetworkModule(DebugModule, PersistableContainer, metaclass=ABCMeta):
    """A utility base network module that contains ubiquitous, but optional layers,
    such as dropout and batch layeres, activation, etc.

    .. document private functions
    .. automethod:: _forward

    """
    def __init__(self, net_settings: NetworkSettings,
                 sub_logger: logging.Logger = None):
        """Initialize.

        :param net_settings: contains common layers such as droput and batch
                             normalization

        :param sub_logger: used to log activity in this module so they logged
                           module comes from some parent model

        """
        super().__init__(sub_logger)
        self.net_settings = ns = net_settings
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
        raise ModelError(f'Layers should not be pickeled: {self}')

    @abstractmethod
    def _forward(self, x: Union[Batch, Tensor], *args, **kwargs) -> Tensor:
        """The model's forward implementation.  Normal backward semantics are no
        different.

        :param x: the batch or tensor to train, validate or test on; the type
                  depends on the needs of the model

        :param args: additional model specific arguments needed by classes that
                     need more context

        :param kwargs: additional model specific arguments needed by classes
                       that need more context

        """
        pass

    @staticmethod
    def device_from_module(module: nn.Module) -> torch.device:
        """Return the device on which the model is configured.

        :param module: the module containing the parameters used to get the
                       device

        """
        return next(module.parameters()).device

    @property
    def device(self) -> torch.device:
        """Return the device on which the model is configured."""
        return self.device_from_module(self)

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
