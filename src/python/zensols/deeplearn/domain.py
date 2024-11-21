"""This file contains classes that configure the network and classifier runs.

"""
__author__ = 'Paul Landes'

from typing import Any, Dict, Tuple, Union, Callable
from dataclasses import dataclass, field, InitVar
from abc import ABCMeta, abstractmethod
from enum import Enum, auto
import sys
import logging
from pathlib import Path
import torch.nn.functional as F
from torch import nn
from zensols.util import APIError
from zensols.config import Writeback, ConfigFactory
from zensols.persist import persisted, PersistableContainer, FileTextUtil
from . import ModelObserverManager, TorchConfig

logger = logging.getLogger(__name__)


class DeepLearnError(APIError):
    """Raised for any frame originated error."""
    pass


class ModelError(DeepLearnError):
    """Raised for any model related error."""
    pass


class EarlyBailError(DeepLearnError):
    """Convenience used for helping debug the network.

    """
    def __init__(self):
        super().__init__('early bail to debug the network')


class DatasetSplitType(Enum):
    """Indicates an action on the model, which is first trained, validated, then
    tested.

    *Implementation note:* for now :obj:`test` is used for both testing the
    model and ad-hoc prediction

    """
    train = auto()
    validation = auto()
    test = auto()


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

    :see: :class:`.ModelSettings`

    """
    config_factory: ConfigFactory = field()
    """The configuration factory used to create the module."""

    torch_config: TorchConfig = field()
    """The GPU configuration for :class:`torch.nn.Module` layers."""

    def __post_init__(self):
        PersistableContainer.__init__(self)

    def _allow_config_adds(self) -> bool:
        return True

    def create_module(self, *args, **kwargs) -> nn.Module:
        """Create a new instance of the network model.

        """
        cls_name = self.get_module_class_name()
        resolver = self.config_factory.class_resolver
        cls = resolver.find_class(cls_name)
        model = cls(self, *args, **kwargs)
        # force the model on the CPU to let the executor manage, otherwise, the
        # model could be on the GPU but only certain parameters on the CPU
        # after load in `load_model_optim_weights'
        model = model.cpu()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'created model {cls} on device: {model.device}')
        return model

    @abstractmethod
    def get_module_class_name(self) -> str:
        """Returns the fully qualified class name of the module to create by
        :class:`~zensols.deeplearn.model.ModelManager`.  This module takes as
        the first parameter an instance of this class.

        **Important**: This method is not used for nested modules.  You must
        declare specific class names in the configuration for those nested
        class naems.

        """
        pass


@dataclass
class ActivationNetworkSettings(NetworkSettings):
    """A network settings that contains a activation setting and creates a
    activation layer.

    """
    activation: Union[Callable, nn.Module, str] = field()
    """The function between all layers, or ``None`` for no activation.

    """
    def _set_option(self, name: str, value: Any):
        super()._set_option(name, value)
        if name == 'activation' and hasattr(self, '_activation_function'):
            self._activation_function.clear()

    @property
    @persisted('_activation_function', transient=True)
    def activation_function(self) -> Callable:
        if isinstance(self.activation, str):
            return self.get_activation_function(self.activation)
        else:
            return self.activation

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
            raise ModelError(f'Known activation function: {activation}')
        return activation

    def __str__(self):
        return f'{super().__str__()},  activation={self.activation}'


@dataclass
class DropoutNetworkSettings(NetworkSettings):
    """A network settings that contains a dropout setting and creates a dropout
    layer.

    """
    dropout: float = field()
    """The droput used in all layers or ``None`` to disable."""

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
    """A network settings that contains a batchnorm setting and creates a
    batchnorm layer.

    """
    batch_norm_d: int = field()
    """The dimension of the batch norm or ``None`` to disable.  Based on this
    one of the following is used as a layer:

    * :class:`torch.nn.BatchNorm1d`
    * :class:`torch.nn.BatchNorm2d`
    * :class:`torch.nn.BatchNorm3d`

    """
    batch_norm_features: int = field()
    """The number of features to use in the batch norm layer."""

    @staticmethod
    def create_batch_norm_layer(batch_norm_d: int, batch_norm_features: int):
        cls = {None: None,
               1: nn.BatchNorm1d,
               2: nn.BatchNorm2d,
               3: nn.BatchNorm3d}[batch_norm_d]
        if cls is not None:
            return cls(batch_norm_features)

    def create_new_batch_norm_layer(self, batch_norm_d: int = None,
                                    batch_norm_features: int = None):
        if batch_norm_d is None:
            batch_norm_d = self.batch_norm_d
        if batch_norm_features is None:
            batch_norm_features = self.batch_norm_features
        if batch_norm_features is None:
            raise ModelError('Missing batch norm features')
        return self.create_batch_norm_layer(batch_norm_d, batch_norm_features)

    @property
    @persisted('_batch_norm_layer', transient=True)
    def batch_norm_layer(self):
        return self.create_new_batch_norm_layer()


@dataclass
class ModelSettings(Writeback, PersistableContainer):
    """This configures and instance of :class:`.ModelExecutor`.  This differes
    from :class:`.NetworkSettings` in that it configures the model parameters,
    and not the neural network parameters.

    Another reason for these two separate classes is data in this class is not
    needed to rehydrate an instance of :class:`torch.nn.Module`.

    The loss function strategy across parameters ``nominal_labels``,
    ``criterion_class`` and ``optimizer_class``, must be consistent.  The
    defaults uses nominal labels, which means a single integer, rather than one
    hot encoding, is used for the labels.  Most loss function, including the
    default :class:`torch.nn.CrossEntropyLoss`` uses nominal labels.  The
    optimizer defaults to :class:`torch.optim.Adam`.

    However, if ``nominal_labels`` is set to ``False``, it is expected that the
    label output is a ``Long`` one hot encoding of the class label that must be
    decoded with :meth:`.BatchIterator._decode_outcomes` and uses a loss
    function such as :class:`torch.nn.BCEWithLogitsLoss`, which applies a
    softmax over the output to narow to a nominal.

    If the ``criterion_class`` is left as the default, the class the
    corresponding class across these two is selected based on
    ``nominal_labels``.

    **Note**: Instances of this class are pickled as parts of the results in
    :class:`zensols.deeplearn.result.domain.ModelResult`, so they must be able
    to serialize.  However, they are not used to restore the executor or model,
    which are instead, recreated from the configuration for each (re)load (see
    the package documentation for more information).

    :see: :class:`.NetworkSettings`

    """
    model_name: str = field()
    """A human readable name for the model."""

    path: Path = field()
    """The path to save and load the model."""

    learning_rate: float = field()
    """Learning_rate used for the gradient descent step (done in the optimzer).

    """
    epochs: int = field()
    """The number of epochs to train the network."""

    append_model_path: str = field(default=None)
    """Whether and how to append the model's name to the end of :obj:`path`.  If
    this value is ``verbatim``, append the model name as provided with
    :obj:`model_name`, if ``normalize`` use :meth:`normalize_name` to normalize
    it, and if ``None`` do not append anything.

    """
    max_consecutive_increased_count: int = field(default=sys.maxsize)
    """The maximum number of times the validation loss can increase per epoch
    before the executor "gives up" and early stops training.

    """
    nominal_labels: bool = field(default=True)
    """``True`` if using numbers to identify the class as an enumeration rather
    than a one hot encoded array.

    """
    labels: Tuple[str, ...] = field(default=None)
    """The list of labels for the model.  This is currently only used for
    multi-label classification so the results and give more useful information.

    """
    batch_iteration_class_name: InitVar[str] = field(default=None)
    """A string fully qualified class name of type :class:`.BatchIterator`.
    This must be set to a class such as :class:`.ScoredBatchIterator` to handle
    descrete states in the output layer such as terminating CRF states.  The
    default is :class:`.BatchIterator`, which expects continuous output layers.

    """
    criterion_class_name: InitVar[str] = field(default=None)
    """The loss function class name (see class doc)."""

    optimizer_class_name: InitVar[str] = field(default=None)
    """The optimization algorithm class name (see class doc)."""

    optimizer_params: Dict[str, Any] = field(default=None)
    """The parameters given as ``**kwargs`` when creating the optimizer.  Do
    **not** add the learning rate, instead see :obj:`learning_rate`."""

    clip_gradient_threshold: float = field(default=None)
    """Parameters passed to :func:`torch.nn.utils.clip_grad_value_` to clip
    gradients above this threshold.

    """
    scale_gradient_params: Dict[str, Union[float, bool]] = field(default=None)
    """Parameters passed to :func:`torch.nn.utils.clip_grad_norm_` to scale
    the gradient norm.

    """
    scheduler_class_name: str = field(default=None)
    """The fully qualified class name of the learning rate scheduler used for
    the optimizer (if not ``None``) such as:

      * :class:`torch.optim.lr_scheduler.StepLR` or,
      * :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`.

    :see: :obj:`scheduler_params`

    """
    scheduler_params: Dict[str, Any] = field(default=None)
    """The parameters given as ``**kwargs`` when creating the scheduler (if
    any).

    :see: :obj:`scheduler_class_name`

    """
    reduce_outcomes: str = field(default='argmax')
    """The method by which the labels and output is reduced.  The output is
    optionally reduced, which is one of the following:

        * ``argmax``: uses the index of the largest value, which is used for
          classification models and the default

        * ``softmax``: just like ``argmax`` but applies a softmax

        * ``none``: return the identity.

        * otherwise, a class name whose instances takes the tensor batch as
          input and returns the the replacement batch tensor as the output

    """
    shuffle_training: bool = field(default=False)
    """If ``True`` shuffle the training data set split before the training
    process starts.  The shuffling only happens once for all epocs.

    """
    batch_limit: Union[int, float] = field(default=sys.maxsize)
    """The max number of batches to train, validate and test on, which is useful
    for limiting while debuging; defaults to `sys.maxsize`.  If this value is a
    float, it is assumed to be a number between [0, 1] and the number of
    batches is multiplied by the value.

    """
    batch_iteration: str = field(default='cpu')
    """How the batches are buffered, which is one of:

        * ``gpu``, buffers all data in the GPU
        * ``cpu``, which means keep all batches in CPU memory (the default)
        * ``buffered`` which means to buffer only one batch at a time (only
          for *very* large data).

    """
    prediction_mapper_name: str = field(default=None)
    """Creates data points from a client for the purposes of prediction.  This
    value is the string class name of an instance of :class:`.PredictionMapper`
    used to create predictions.  While optional, if not set, ad-hoc predictions
    (i.e. from the command line) can not be created.

    Instances of :class:`.PredictionMapper` are created and managed in the
    :class:`~zensols.deeplearn.model.ModelFacade`.

    """
    cache_batches: bool = field(default=True)
    """If ``True`` cache unthawed/processed batches when possible."""

    gc_level: int = field(default=0)
    """The frequency by with the garbage collector is invoked.  The *higher* the
    value, the more often it will be run during training, testing and
    validation.

    """
    observer_manager: ModelObserverManager = field(
        default_factory=ModelObserverManager)
    """The model observer used by the entire train, test, validation process.

    """
    store_model_result: str = field(default='test')
    """Whether to store the :class:`~.result.domain.ModelResult` instance in the
    state file, which is one of:

        * ``test``: only tested models, as apposed to using
          :meth:`~.model.facade.FacadeModel.train_production`
        * ``always``: always save results, even in production models
        * ``never``: there will be no training or validation results in output

    The results are also stored as ``.dat`` files in the results directory.

    :see: :obj:`store_report`

    """
    store_report: bool = field(default=True)
    """Whether to store the contents of the results report when the results
    aren't persisted with the model.  The report is store as key
    ``model_result_report`` in the ``state.pt`` that lives in the model
    directory with ``weight.pt``.

    :see: :obj:`store_model_result`

    """
    def __post_init__(self,
                      batch_iteration_class_name: str,
                      criterion_class_name: str,
                      optimizer_class_name: str):
        if batch_iteration_class_name is None:
            self.batch_iteration_class_name = \
                'zensols.deeplearn.model.BatchIterator'
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
        if self.append_model_path is not None:
            if self.append_model_path == 'verbatim':
                self.path = self.path / self.model_name
            elif self.append_model_path == 'normalize':
                self.path = self.path / self.normalize_name(self.model_name)
            else:
                raise ModelError("Unknown 'append_model_path' " +
                                 f"value: '{self.append_model_path}'")

    @staticmethod
    def normalize_name(name: str) -> str:
        """Normalize the name in to a string that is more file system friendly.
        This is used for the :obj:`model_name` by API components that write data
        to the file system about the model this class configures such as
        :class:`~zensols.deeplearn.result.ModelResultManager`.

        :return: the normalized name

        """
        return FileTextUtil.normalize_text(name)

    @property
    def normal_model_name(self) -> str:
        """Return the normalized :obj:`model_name` using :meth:`normalize_name`.

        """
        return self.normalize_name(self.model_name)

    def _allow_config_adds(self) -> bool:
        return True
