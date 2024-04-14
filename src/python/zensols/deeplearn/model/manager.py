"""A utility class to help with a ``ModelExecutor`` life cycle.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import Any, Dict, Tuple
from dataclasses import dataclass, field
import logging
from io import StringIO
from pathlib import Path
import torch
from zensols.util import time
from zensols.config import ConfigFactory, Configurable
from .. import ModelError, TorchConfig, NetworkSettings
from . import BaseNetworkModule

logger = logging.getLogger(__name__)


@dataclass
class ModelManager(object):
    """This class manages the lifecycle of an instance of a ``ModelExecutor``.
    This class is mostly used by the executor to control it's lifecycle.
    However, a client can also use an instance of this to revive a model that's
    been saved to disk with the ``ModelResultManager``

    :see ModelExecutor:

    """
    path: Path = field()
    """The path of where the model results saved to disk by
    :class:`.zensols.deeplearn.results.ModelResultManager`.

    """
    config_factory: ConfigFactory = field()
    """The configuration factory to be used to create the ``ModelExecutor``."""

    model_executor_name: str = field(default=None)
    """The configuration entry and name of the ``ModelExecutor`` instance."""

    persist_random_seed_context: bool = field(default=True)
    """If ``True`` persist the current random seed state, which helps in
    creating consistent results across train/test/validate.

    """
    keep_last_state_dict: bool = field(default=False)
    """Whether to store the PyTorch module state in attribute
    ``last_saved_state_dict``.

    """
    @staticmethod
    def _get_paths(path: Path) -> Tuple[Path, Path]:
        return (path / 'state.pt', path / 'weight.pt')

    @classmethod
    def load_from_path(cls, path: Path) -> ModelManager:
        """Load and return an instance of this class from a previously saved
        model.  This method exists to recreate a :class:`.ModelManager` from a
        saved file from scratch.  The returned model manager can be used to
        create the executor or :class:`ModelFacade` using
        :obj:``config_factory``.

        :param path: points to the model file persisted with
                     :py:meth:`_save_executor`

        :return: an instance of :class:`.ModelManager` that was used to save
                 the executor pointed by ``path``

        """
        checkpoint = cls._load_checkpoint(*cls._get_paths(path))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'keys: {checkpoint.keys()}')
        config_factory = checkpoint['config_factory']
        model_executor_name = checkpoint['model_executor_name']
        persist_random = checkpoint['random_seed_context'] is not None
        return cls(path, config_factory, model_executor_name, persist_random)

    def load_executor(self, config_overwrites: Configurable = None) -> \
            'ModelExecutor':
        """Load the model the last saved model from the disk.  This is used load
        an instance of a ``ModelExecutor`` with all previous state completely in
        tact.  It does this by using an instance of
        :class:`zensols.config.factory.Configurable` and a
        :class:`zensols.config.factory.ImportConfigFactory` to reconstruct the
        executor and it's state by recreating all instances.

        After the executor has been recreated with the factory, the previous
        model results and model weights are restored.

        :param load_factory: whether to load the configuration factory from the
                             check point; which you probably don't want when
                             loading from :meth:`load_from_path`

        :return: an instance of :class:`.ModelExecutor`

        :see: :class:`zensols.deeplearn.model.ModelExecutor`

        """
        checkpoint: Dict[str, Any] = self._get_checkpoint(True)
        # reload the config factory even if loaded from `load_from_path` since,
        # in that case, this instance will be deallcated in the facade
        config_factory: ConfigFactory = checkpoint['config_factory']
        self._set_random_seed(checkpoint)
        # overwrite model configuration before the executor is instantiated
        if config_overwrites is not None:
            config_overwrites.copy_sections(config_factory.config)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'loading config factory: {config_factory}')
        # create the executor from the executor section
        executor: 'Executor' = config_factory.instance(
            checkpoint['model_executor_name'])
        # create the PyTorch model
        model: BaseNetworkModule = self._create_module(executor.net_settings)
        # load and set the state
        self._load_optimizer_state(executor, model, checkpoint)
        if 'model_result_report' in checkpoint:
            executor._model_result_report = checkpoint['model_result_report']
        return executor

    def _load_optimizer_state(self, executor: Any, model: BaseNetworkModule,
                              checkpoint: Dict[str, Any]):
        model.load_state_dict(checkpoint['model_state_dict'])
        executor._set_model(model, True, False)
        executor.model_result = checkpoint['model_result']
        criterion, optimizer, scheduler = executor.criterion_optimizer_scheduler
        if 'model_scheduler_state_dict' in checkpoint:
            scheduler_state = checkpoint['model_scheduler_state_dict']
        else:
            scheduler_state = None
        optimizer.load_state_dict(checkpoint['model_optim_state_dict'])
        if scheduler is not None and scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'loaded model from {executor.model_settings.path} ' +
                        f'on device {model.device}')

    def _load_model_optim_weights(self, executor):
        """Load the model and optimizer weights from the last check point.  A
        side effect is that the optimizer is recreated.

        """
        model = executor._get_or_create_model()
        checkpoint = self._get_checkpoint(True)
        self._load_optimizer_state(executor, model, checkpoint)

    def _save_executor(self, executor: Any, save_model_result: bool):
        """Save a ``ModelExecutor`` instance.

        :param executor: the executor to persost to disk
        """
        logger.debug('saving model state')
        if self.persist_random_seed_context:
            random_seed_context = TorchConfig.get_random_seed_context()
        else:
            random_seed_context = None
        criterion, optimizer, scheduler = executor.criterion_optimizer_scheduler
        if scheduler is None:
            scheduler_state = None
        else:
            scheduler_state = scheduler.state_dict()
        state_dict = executor.model.state_dict()
        if self.keep_last_state_dict:
            self.last_saved_state_dict = self._copy_state_dict(state_dict)
        if save_model_result:
            model_result = executor.model_result
        else:
            model_result = None
        checkpoint = {'config_factory': self.config_factory,
                      'random_seed_context': random_seed_context,
                      'model_executor_name': self.model_executor_name,
                      'net_settings_name': executor.net_settings.name,
                      'model_result': model_result,
                      'model_optim_state_dict': optimizer.state_dict(),
                      'model_scheduler_state_dict': scheduler_state,
                      'model_state_dict': state_dict}
        if model_result is None and executor.model_settings.store_report:
            sio = StringIO()
            executor.model_result.write(writer=sio)
            checkpoint['model_result_report'] = sio.getvalue().strip()
        self._save_checkpoint(checkpoint, True)

    def _create_module(self, net_settings: NetworkSettings,
                       reload: bool = False) -> BaseNetworkModule:
        """Create a new instance of the network model.

        """
        resolver = self.config_factory.class_resolver
        initial_reload = resolver.reload
        try:
            resolver.reload = reload
            return net_settings.create_module()
        finally:
            resolver.reload = initial_reload

    @staticmethod
    def _copy_state_dict(state_dict):
        """Copy the PyTorch module state (weights) and return them as a dict.

        """
        return {k: state_dict[k].clone() for k in state_dict.keys()}

    def _set_random_seed(self, checkpoint: Dict[str, Any]):
        random_seed_context = checkpoint['random_seed_context']
        if random_seed_context is not None:
            TorchConfig.set_random_seed(**random_seed_context)

    def _save_final_trained_results(self, executor):
        """Save the results of the :class:`.ModelResult`, which is typically
        called when the validation loss decreases.  Note this does not save the
        model weights since doing so might clobber with an overtrained model
        (assuming the last converved with the lowest validation loss was saved).
        Instead it only saves the model results.

        :param executor: the executor with the model results to save

        """
        self._update_config_checkpoint({'model_result': executor.model_result})

    def _update_config_factory(self, config_factory: ConfigFactory):
        self._update_config_checkpoint({'config_factory': config_factory})

    def _update_config_checkpoint(self, data: Dict[str, Any]):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'updating results ({set(data.keys())}): {self.path}')
        checkpoint = self._get_checkpoint(False)
        checkpoint.update(data)
        self._save_checkpoint(checkpoint, False)

    def _save_checkpoint(self, checkpoint: Dict[str, Any], save_weights: bool):
        """Save the check point to disk.

        :param checkpoint: all model state (results, random seed, weights etc)

        :param save_weights: if ``True`` then save the weights to the weight
                             file (in addition to the state to the state file)

        """
        state_path, weight_path = self._get_paths(self.path)
        weights = {}
        for k in 'model_optim_state_dict model_state_dict'.split():
            wval = checkpoint.pop(k, None)
            if save_weights and wval is None:
                raise ModelError(
                    f'Missing checkpoint key while saving weights: {k}')
            weights[k] = wval
        self.path.mkdir(parents=True, exist_ok=True)
        if save_weights:
            with time(f'saved model weights to {weight_path}'):
                torch.save(weights, str(weight_path))
        with time(f'saved model state to {state_path}'):
            torch.save(checkpoint, str(state_path))

    def _get_checkpoint(self, load_weights: bool) -> Dict[str, Any]:
        """The check point from loaded by the PyTorch framework.  This contains
        the executor, model results, and model weights.

        :param load_weights: if ``True`` load the weights from the weights file
                             and add it to the checkpoint state

        """
        state_path, weight_path = self._get_paths(self.path)
        if not load_weights:
            weight_path = None
        return self._load_checkpoint(state_path, weight_path)

    @staticmethod
    def _load_checkpoint(state_path: Path, weight_path: Path) -> \
            Dict[str, Any]:
        if not state_path.is_file():
            raise ModelError(f'No such state file: {state_path}')
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'loading check point from: {state_path}')
        with time(f'loaded check point from {state_path}'):
            cp = torch.load(str(state_path))
        if weight_path is not None:
            params = {}
            if not torch.cuda.is_available():
                params['map_location'] = torch.device('cpu')
            weights = torch.load(str(weight_path), **params)
            cp.update(weights)
        return cp
