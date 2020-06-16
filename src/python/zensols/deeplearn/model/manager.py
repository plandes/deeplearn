"""A utility class to help with a ``ModelExecutor`` life cycle.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field
from typing import Any, Tuple, Dict
import logging
from pathlib import Path
import torch
from zensols.util import time
from zensols.config import ConfigFactory
from zensols.deeplearn import TorchConfig, NetworkSettings
from zensols.deeplearn.result import ModelResult
from . import BaseNetworkModule

logger = logging.getLogger(__name__)


@dataclass
class ModelManager(object):
    """This class manages the lifecycle of an instance of a ``ModelExecutor``.
    This class is mostly used by the executor to control it's lifecycle.
    However, a client can also use an instance of this to revive a model that's
    been saved to disk with the ``ModelResultManager``

    :param path: the path of where the model results saved to disk by
                 ``ModelResultManager``

    :param config_factory: the configuration factory to be used to create
                           the ``ModelExecutor``

    :param model_executor_name: the configuration entry and name of the
                                ``ModelExecutor`` instance.

    :param keep_last_state_dict: whether or not to store the PyTorch module
                                 state in attribute ``last_saved_state_dict``

    :see ModelExecutor:

    """
    path: Path
    config_factory: ConfigFactory
    model_executor_name: str = field(default=None)
    persist_random_seed_context: bool = field(default=True)
    keep_last_state_dict: bool = field(default=False)

    @classmethod
    def load_from_path(cls, path: Path):
        """Load and return an instance of this class from a previously saved model.
        This method exists to recreate a :class:`.ModelManager` from a saved
        file from scratch.  The returned model manager can be used to create
        the executor or :class:`ModelFacade` using
        :py:attrib:~``config_factory``.

        :param path: points to the model file persisted with
                     :py:meth:`save_executor`

        :return: an instance of :class:`.ModelManager` that was used to save
                 the executor pointed by ``path``

        """
        checkpoint = cls._load_checkpoint(path)
        logger.debug(f'keys: {checkpoint.keys()}')
        config_factory = checkpoint['config_factory']
        model_executor = checkpoint['model_executor']
        persist_random = checkpoint['random_seed_context'] is not None
        return cls(path, config_factory, model_executor, persist_random)

    def load_executor(self) -> Any:
        """Load the model the last saved model from the disk.  This is used load an
        instance of a ``ModelExecutor`` with all previous state completely in
        tact.  It does this by using an instance of
        ``zensols.config.Configurable`` and a
        ``zensols.config.ImportConfigFactory`` to reconstruct the executor and
        it's state by recreating all instances.

        After the executor has been recreated with the factory, the previous
        model results and model weights are restored.

        :return: an instance of :class:`zensols.deeplearn.model.ModelExecutor`
        :see: :class:`zensols.deeplearn.model.ModelExecutor`

        """
        checkpoint = self.checkpoint
        config_factory = checkpoint['config_factory']
        logger.debug(f'loading config factory: {config_factory}')
        # executor: ModelExecutor
        executor = config_factory.instance(checkpoint['model_executor'])
        model = self.load_model(executor.net_settings, checkpoint)[0]
        executor.model = model
        executor.model_result = checkpoint['model_result']
        optimizer = executor.criterion_optimizer[1]
        optimizer.load_state_dict(checkpoint['model_optim_state_dict'])
        logger.info(f'loaded model from {executor.model_settings.path} ' +
                    f'on device {model.device}')
        return executor

    def create_module(self, net_settings: NetworkSettings) -> BaseNetworkModule:
        """Create a new instance of the network model instance.

        """
        cls_name = net_settings.get_module_class_name()
        resolver = self.config_factory.class_resolver
        initial_reload = resolver.reload
        try:
            resolver.reload = net_settings.debug
            cls = resolver.find_class(cls_name)
        finally:
            resolver.reload = initial_reload
        model = cls(net_settings)
        return model

    def save_executor(self, executor: Any):
        """Save a ``ModelExecutor`` instance.

        :param executor: the executor to persost to disk
        """
        logger.debug('saving model state')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.persist_random_seed_context:
            random_seed_context = TorchConfig.get_random_seed_context()
        else:
            random_seed_context = None
        optimizer = executor.criterion_optimizer[1]
        state_dict = executor.model.state_dict()
        if self.keep_last_state_dict:
            self.last_saved_state_dict = self._copy_state_dict(state_dict)
        checkpoint = {'config_factory': self.config_factory,
                      'random_seed_context': random_seed_context,
                      'model_executor': self.model_executor_name,
                      'model_result': executor.model_result,
                      'model_optim_state_dict': optimizer.state_dict(),
                      'model_state_dict': state_dict}
        logger.debug(f'saving model to {self.path}')
        self._save_checkpoint(checkpoint)

    @staticmethod
    def _copy_state_dict(state_dict):
        """Copy the PyTorch module state (weights) and return them as a dict.

        """
        return {k: state_dict[k].clone() for k in state_dict.keys()}

    def update_results(self, executor):
        """Update the ``ModelResult``, which is typically called when the validation
        loss decreases.

        """
        logger.debug(f'updating results: {self.path}')
        checkpoint = self.checkpoint
        checkpoint['model_result'] = executor.model_result
        self._save_checkpoint(checkpoint)

    def _save_checkpoint(self, checkpoint: Dict[str, Any]):
        with time(f'saved check point to {self.path}'):
            torch.save(checkpoint, str(self.path))

    @property
    def checkpoint(self) -> Dict[str, Any]:
        """The check point from loaded by the PyTorch framework.  This contains the
        executor, model results, and model weights.

        """
        if not self.path.exists():
            raise OSError(f'no such model file: {self.path}')
        logger.debug(f'loading check point from: {self.path}')
        checkpoint = self._load_checkpoint(self.path)
        random_seed_context = checkpoint['random_seed_context']
        if random_seed_context is not None:
            TorchConfig.set_random_seed(**random_seed_context)
        return checkpoint

    @staticmethod
    def _load_checkpoint(path):
        with time(f'loaded check point from {path}'):
            return torch.load(str(path))

    def load_model(self, net_settings: NetworkSettings,
                   checkpoint: dict = None) -> \
            Tuple[BaseNetworkModule, ModelResult]:
        """Load the model state found in the check point and the neural network
        settings configuration object.  This returns the PyTorch module with
        the populated unpersisted weights and the model results previouly
        persisted.

        This is called when recreating the ``ModelExecutor`` instance.

        """
        if checkpoint is None:
            logger.debug(f'loading model from: {self.path}')
            checkpoint = self.checkpoint
        model: BaseNetworkModule = self.create_module(net_settings)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint['model_result']
