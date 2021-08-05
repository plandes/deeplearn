"""This file contains the network model and data that holds the results.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field
from typing import List, Callable, Tuple, Any, Union
import sys
import gc
import logging
import itertools as it
from itertools import chain
from io import TextIOBase, StringIO
import random as rand
from pathlib import Path
import torch
from torch import nn
from tqdm import tqdm
from zensols.util import time
from zensols.config import Configurable, ConfigFactory, Writable
from zensols.persist import (
    Deallocatable,
    persisted, PersistedWork, PersistableContainer,
    Stash, UnionStash,
)
from zensols.dataset import SplitStashContainer, DatasetSplitStash
from zensols.deeplearn import (
    ModelError, EarlyBailError,
    TorchConfig, DatasetSplitType, NetworkSettings
)
from zensols.deeplearn.result import (
    EpochResult, ModelResult, ModelSettings, ModelResultManager,
)
from zensols.deeplearn.batch import BatchStash, Batch
from . import (
    ModelResourceFactory, BaseNetworkModule,
    ModelManager, UpdateAction,
    BatchIterator, TrainManager,
)

# default message logger
logger = logging.getLogger(__name__ + '.status')
# logger for messages, which is active when the progress bar is not
progress_logger = logging.getLogger(__name__ + '.progress')


@dataclass
class ModelExecutor(PersistableContainer, Deallocatable, Writable):
    """This class creates and uses a network to train, validate and test the model.
    This class is either configured using a
    :class:`~zensols.config.factory.ConfigFactory` or is unpickled with
    :class:`.ModelManager`.  If the later, it's from a previously trained (and
    possibly tested) state.

    Typically, after creating a nascent instance, :meth:`train` is called to
    train the model.  This returns the results, but the results are also
    available via the :class:`ResultManager` using the :obj:`model_manager`
    property.  To load previous results, use
    ``executor.result_manager.load()``.

    During training, the training set is used to train the weights of the model
    provided by the executor in the :obj:`model_settings`, then validated using
    the validation set.  When the validation loss is minimized, the following
    is saved to disk:

        * Settings: :obj:`net_settings`, :obj:`model_settings`,
        * the model weights,
        * the results of the training and validation thus far,
        * the entire configuration (which is later used to restore the
          executor),
        * random seed information, which includes Python, Torch and GPU random
          state.

    After the model is trained, you can immediately test the model with
    :meth:`test`.  To be more certain of being able to reproduce the same
    results, it is recommended to load the model with
    ``model_manager.load_executor()``, which loads the last instance of the
    model that produced a minimum validation loss.

    :see: :class:`.ModelExecutor`
    :see: :class:`.NetworkSettings`
    :see: :class:`zensols.deeplearn.model.ModelSettings`

    """
    ATTR_EXP_META = ('model_settings',)

    config_factory: ConfigFactory = field()
    """The configuration factory that created this instance."""

    config: Configurable = field()
    """The configuration used in the configuration factory to create this
    instance.

    """

    name: str = field()
    """The name given in the configuration."""

    model_name: str = field()
    """A human readable name for the model."""

    model_settings: ModelSettings = field()
    """The configuration of the model."""

    net_settings: NetworkSettings = field()
    """The settings used to configure the network."""

    dataset_stash: DatasetSplitStash = field()
    """The split data set stash that contains the ``BatchStash``, which
    contains the batches on which to train and test.

    """

    dataset_split_names: List[str] = field()
    """The list of split names in the ``dataset_stash`` in the order: train,
    validation, test (see :meth:`_get_dataset_splits`)

    """

    result_path: Path = field(default=None)
    """If not ``None``, a path to a directory where the results are to be
    dumped; the directory will be created if it doesn't exist when the results
    are generated.

    """

    update_path: Path = field(default=None)
    """The path to check for commands/updates to make while training.  If this is
    set, and the file exists, then it is parsed as a JSON file.  If the file
    cannot be parsed, or 0 size etc., then the training is (early) stopped.

    If the file can be parsed, and there is a single ``epoch`` dict entry, then
    the current epoch is set to that value.

    """

    intermediate_results_path: Path = field(default=None)
    """If this is set, then save the model and results to this path after
    validation for each training epoch.

    """

    progress_bar: bool = field(default=False)
    """Create text/ASCII based progress bar if ``True``."""

    progress_bar_cols: int = field(default=None)
    """The number of console columns to use for the text/ASCII based progress
    bar.

    """

    def __post_init__(self):
        super().__init__()
        if not isinstance(self.dataset_stash, DatasetSplitStash) and False:
            raise ModelError('Expecting type DatasetSplitStash but ' +
                             f'got {self.dataset_stash.__class__}')
        self._model = None
        self._dealloc_model = False
        self.model_result: ModelResult = None
        self.batch_stash.delegate_attr: bool = True
        self._criterion_optimizer_scheduler = PersistedWork(
            '_criterion_optimizer_scheduler', self)
        self._result_manager = PersistedWork('_result_manager', self)
        self._train_manager = PersistedWork('_train_manager', self)
        self.cached_batches = {}
        self.debug = False

    @property
    def batch_stash(self) -> DatasetSplitStash:
        """The stash used to obtain the data for training and testing.  This stash
        should have a training, validation and test splits.  The names of these
        splits are given in the ``dataset_split_names``.

        """
        return self.dataset_stash.split_container

    @property
    def feature_stash(self) -> Stash:
        """The stash used to generate the feature, which is not to be confused
        with the batch source stash``batch_stash``.

        """
        return self.batch_stash.split_stash_container

    @property
    def torch_config(self) -> TorchConfig:
        """Return the PyTorch configuration used to convert models and data (usually
        GPU) during training and test.

        """
        return self.batch_stash.model_torch_config

    @property
    @persisted('_result_manager')
    def result_manager(self) -> ModelResultManager:
        """Return the manager used for controlling the life cycle of the results
        generated by this executor.

        """
        if self.result_path is not None:
            return self._create_result_manager(self.result_path)

    def _create_result_manager(self, path: Path) -> ModelResultManager:
        return ModelResultManager(
            name=self.model_name, path=path,
            model_path=self.model_settings.path)

    @property
    @persisted('_model_manager')
    def model_manager(self) -> ModelManager:
        """Return the manager used for controlling the train of the model.

        """
        model_path = self.model_settings.path
        return ModelManager(model_path, self.config_factory, self.name)

    @property
    @persisted('_batch_iterator')
    def batch_iterator(self) -> BatchIterator:
        """The train manager that assists with the training process.

        """
        resolver = self.config_factory.class_resolver
        batch_iter_class_name = self.model_settings.batch_iteration_class_name
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'batch_iteration: {batch_iter_class_name}')
        batch_iter_class = resolver.find_class(batch_iter_class_name)
        batch_iter = batch_iter_class(self, logger)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'batch_iter={batch_iter}')
        return batch_iter

    @property
    def debug(self) -> Union[bool, int]:
        return self._debug

    @debug.setter
    def debug(self, debug: Union[bool, int]):
        self._debug = debug
        self.batch_iterator.debug = debug

    @property
    @persisted('_train_manager')
    def train_manager(self) -> TrainManager:
        """Return the train manager that assists with the training process.

        """
        return TrainManager(
            logger, progress_logger, self.update_path,
            self.model_settings.max_consecutive_increased_count)

    def _weight_reset(self, m):
        if hasattr(m, 'reset_parameters') and callable(m.reset_parameters):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'resetting parameters on {m}')
            m.reset_parameters()

    def reset(self):
        """Reset the executor's to it's nascent state.

        """
        if logger.isEnabledFor(logging.INFO):
            logger.info('resetting executor')
        self._criterion_optimizer_scheduler.clear()
        self._deallocate_model()

    def load(self) -> nn.Module:
        """Clear all results and trained state and reload the last trained model from
        the file system.

        :return: the model that was loaded and registered in this instance of
                 the executor

        """
        if logger.isEnabledFor(logging.INFO):
            logger.info('reloading model weights')
        self._deallocate_model()
        self.model_manager._load_model_optim_weights(self)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'copied model to {self.model.device}')
        return self.model

    def deallocate(self):
        super().deallocate()
        self._deallocate_model()
        self.deallocate_batches()
        self._try_deallocate(self.dataset_stash)
        self._deallocate_settings()
        self._criterion_optimizer_scheduler.deallocate()
        self._result_manager.deallocate()
        self.model_result = None

    def _deallocate_model(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('dealloc model: model exists/dealloc: ' +
                         f'{self._model is not None}/{self._dealloc_model}')
        if self._model is not None and self._dealloc_model:
            self._try_deallocate(self._model)
        self._model = None

    def _deallocate_settings(self):
        self.model_settings.deallocate()
        self.net_settings.deallocate()

    def deallocate_batches(self):
        set_of_ds_sets = self.cached_batches.values()
        ds_sets = chain.from_iterable(set_of_ds_sets)
        batches = chain.from_iterable(ds_sets)
        for batch in batches:
            batch.deallocate()
        self.cached_batches.clear()

    @property
    def model_exists(self) -> bool:
        """Return whether the executor has a model.

        :return: ``True`` if the model has been trained or loaded

        """
        return self._model is not None

    @property
    def model(self) -> BaseNetworkModule:
        """Get the PyTorch module that is used for training and test.

        """
        if self._model is None:
            raise ModelError('No model, is populated; use \'load\'')
        return self._model

    @model.setter
    def model(self, model: BaseNetworkModule):
        """Set the PyTorch module that is used for training and test.

        """
        self._set_model(model, False, True)

    def _set_model(self, model: BaseNetworkModule,
                   take_owner: bool, deallocate: bool):
        if logger.isEnabledFor(level=logging.DEBUG):
            logger.debug(f'setting model: {type(model)}')
        if deallocate:
            self._deallocate_model()
        self._model = model
        self._dealloc_model = take_owner
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'setting dealloc model: {self._dealloc_model}')
        self._criterion_optimizer_scheduler.clear()

    def _get_or_create_model(self) -> BaseNetworkModule:
        if self._model is None:
            self._dealloc_model = True
            model = self._create_model()
            self._model = model
        else:
            model = self._model
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'created model as dealloc: {self._dealloc_model}')
        return model

    def _create_model(self) -> BaseNetworkModule:
        """Create the network model instance.

        """
        mng: ModelManager = self.model_manager
        model = mng._create_module(self.net_settings, self.debug)
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'created model on {model.device} ' +
                        f'with {self.torch_config}')
        return model

    def _create_model_result(self) -> ModelResult:
        res = ModelResult(
            self.config, f'{self.model_name}: {ModelResult.get_num_runs()}',
            self.model_settings, self.net_settings,
            self.batch_stash.decoded_attributes)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'creating model result ({id(res)}): {self.model_name}')
        return res

    @property
    @persisted('_criterion_optimizer_scheduler')
    def criterion_optimizer_scheduler(self) -> \
            Tuple[nn.L1Loss, torch.optim.Optimizer, Any]:
        """Return the loss function and descent optimizer.

        """
        criterion = self._create_criterion()
        optimizer, scheduler = self._create_optimizer_scheduler()
        return criterion, optimizer, scheduler

    def _create_criterion(self) -> torch.optim.Optimizer:
        """Factory method to create the loss function and optimizer.

        """
        resolver = self.config_factory.class_resolver
        criterion_class_name = self.model_settings.criterion_class_name
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'criterion: {criterion_class_name}')
        criterion_class = resolver.find_class(criterion_class_name)
        criterion = criterion_class()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'criterion={criterion}')
        return criterion

    def _create_optimizer_scheduler(self) -> Tuple[nn.L1Loss, Any]:
        """Factory method to create the optimizer and the learning rate scheduler (is
        any).

        """
        model = self.model
        resolver = self.config_factory.class_resolver
        optimizer_class_name = self.model_settings.optimizer_class_name
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'optimizer: {optimizer_class_name}')
        optimizer_class = resolver.find_class(optimizer_class_name)
        if self.model_settings.optimizer_params is None:
            optimizer_params = {}
        else:
            optimizer_params = dict(self.model_settings.optimizer_params)
        optimizer_params['lr'] = self.model_settings.learning_rate
        if issubclass(optimizer_class, ModelResourceFactory):
            opt_call = optimizer_class()
            optimizer_params['model'] = model
            optimizer_params['executor'] = self
        else:
            opt_call = optimizer_class
        optimizer = opt_call(model.parameters(), **optimizer_params)
        scheduler_class_name = self.model_settings.scheduler_class_name
        if scheduler_class_name is not None:
            scheduler_class = resolver.find_class(scheduler_class_name)
            scheduler_params = self.model_settings.scheduler_params
            if scheduler_params is None:
                scheduler_params = {}
            else:
                scheduler_params = dict(scheduler_params)
            scheduler_params['optimizer'] = optimizer
            if issubclass(scheduler_class, ModelResourceFactory):
                # model resource factories are callable
                sch_call = scheduler_class()
                scheduler_params['executor'] = self
            else:
                sch_call = scheduler_class
            scheduler = sch_call(**scheduler_params)
        else:
            scheduler = None
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'optimizer={optimizer}')
        return optimizer, scheduler

    def get_model_parameter(self, name: str):
        """Return a parameter of the model, found in ``model_settings``.

        """
        return getattr(self.model_settings, name)

    def set_model_parameter(self, name: str, value: Any):
        """Safely set a parameter of the model, found in ``model_settings``.  This
        makes the corresponding update in the configuration, so that when it is
        restored (i.e for test) the parameters are consistent with the trained
        model.  The value is converted to a string as the configuration
        representation stores all data values as strings.

        *Important*: ``eval`` syntaxes are not supported, and probably not the
        kind of values you want to set a parameters with this interface anyway.

        :param name: the name of the value to set, which is the key in the
                     configuration file

        :param value: the value to set on the model and the configuration

        """
        self.config.set_option(
            name, str(value), section=self.model_settings.name)
        setattr(self.model_settings, name, value)

    def get_network_parameter(self, name: str):
        """Return a parameter of the network, found in ``network_settings``.

        """
        return getattr(self.net_settings, name)

    def set_network_parameter(self, name: str, value: Any):
        """Safely set a parameter of the network, found in ``network_settings``.  This
        makes the corresponding update in the configuration, so that when it is
        restored (i.e for test) the parameters are consistent with the trained
        network.  The value is converted to a string as the configuration
        representation stores all data values as strings.

        *Important*: ``eval`` syntaxes are not supported, and probably not the
        kind of values you want to set a parameters with this interface anyway.

        :param name: the name of the value to set, which is the key in the
                     configuration file

        :param value: the value to set on the network and the configuration

        """
        self.config.set_option(
            name, str(value), section=self.net_settings.name)
        setattr(self.net_settings, name, value)

    def _to_iter(self, ds):
        ds_iter = ds
        if isinstance(ds_iter, Stash):
            ds_iter = ds_iter.values()
        return ds_iter

    def _gc(self, level: int):
        """Invoke the Python garbage collector if ``level`` is high enough.  The
        *lower* the value of ``level``, the more often it will be run during
        training, testing and validation.

        :param level: if priority of the need to collect--the lower the more
                      its needed

        """
        if level <= self.model_settings.gc_level:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('garbage collecting')
            with time('garbage collected', logging.DEBUG):
                gc.collect()

    def _train(self, train: List[Batch], valid: List[Batch]):
        """Train the network model and record validation and training losses.  Every
        time the validation loss shrinks, the model is saved to disk.

        """
        n_epochs = self.model_settings.epochs
        # create network model, loss and optimization functions
        model = self._get_or_create_model()
        model = self.torch_config.to(model)
        self._model = model
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'training model {type(model)} on {model.device} ' +
                        f'for {n_epochs} epochs using ' +
                        f'learning rate {self.model_settings.learning_rate}')
        criterion, optimizer, scheduler = self.criterion_optimizer_scheduler
        # create a second module manager for after epoch results
        if self.intermediate_results_path is not None:
            model_path = self.intermediate_results_path
            intermediate_manager = self._create_result_manager(model_path)
            intermediate_manager.file_pattern = '{prefix}.{ext}'
        else:
            intermediate_manager = None
        train_manager = self.train_manager
        action = UpdateAction.ITERATE_EPOCH
        # set up graphical progress bar
        exec_logger = logging.getLogger(__name__)
        if self.progress_bar and \
            (exec_logger.level == 0 or
             exec_logger.level > logging.INFO) and \
            (progress_logger.level == 0 or
             progress_logger.level > logging.INFO):
            pbar = tqdm(total=n_epochs, ncols=self.progress_bar_cols)
        else:
            pbar = None

        train_manager.start(optimizer, scheduler, n_epochs, pbar)
        self.model_result.train.start()
        self.model_result.validation.start()

        # epochs loop
        while action != UpdateAction.STOP:
            epoch = train_manager.current_epoch
            train_epoch_result = EpochResult(epoch, DatasetSplitType.train)
            valid_epoch_result = EpochResult(epoch, DatasetSplitType.validation)

            if progress_logger.isEnabledFor(logging.INFO):
                progress_logger.debug(f'training on epoch: {epoch}')

            self.model_result.train.append(train_epoch_result)
            self.model_result.validation.append(valid_epoch_result)

            # train ----
            # prep model for training and train
            model.train()
            train_epoch_result.start()
            for batch in self._to_iter(train):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'training on batch: {batch.id}')
                with time('trained batch', level=logging.DEBUG):
                    self.batch_iterator.iterate(
                        model, optimizer, criterion, batch,
                        train_epoch_result, DatasetSplitType.train)
                self._gc(3)
            train_epoch_result.end()

            self._gc(2)

            # validate ----
            # prep model for evaluation and evaluate
            ave_valid_loss = 0
            model.eval()
            valid_epoch_result.start()
            for batch in self._to_iter(valid):
                # forward pass: compute predicted outputs by passing inputs
                # to the model
                with torch.no_grad():
                    loss = self.batch_iterator.iterate(
                        model, optimizer, criterion, batch,
                        valid_epoch_result, DatasetSplitType.validation)
                    ave_valid_loss += (loss.item() * batch.size())
                self._gc(3)
            valid_epoch_result.end()
            ave_valid_loss = ave_valid_loss / len(valid)

            self._gc(2)

            valid_loss_min, decreased = train_manager.update_loss(
                valid_epoch_result, train_epoch_result, ave_valid_loss)

            if decreased:
                self.model_manager._save_executor(self)
                if intermediate_manager is not None:
                    inter_res = self.model_result.get_intermediate()
                    intermediate_manager.save_text_result(inter_res)
                    intermediate_manager.save_plot_result(inter_res)

            # look for indication of update or early stopping
            status = train_manager.get_status()
            action = status.action

        val_losses = train_manager.validation_loss_decreases
        if logger.isEnabledFor(logging.INFO):
            logger.info('final minimum validation ' +
                        f'loss: {train_manager.valid_loss_min}, ' +
                        f'{val_losses} decreases')

        if val_losses == 0:
            logger.warn('no validation loss decreases encountered, ' +
                        'so there was no model saved; model can not be tested')

        self.model_result.train.end()
        self.model_result.validation.end()
        self.model_manager._save_final_trained_results(self)

    def _test(self, batches: List[Batch]):
        """Test the model on the test set.  If a model is not given, it is unpersisted
        from the file system.

        """
        # create the loss and optimization functions
        criterion, optimizer, scheduler = self.criterion_optimizer_scheduler
        model = self.torch_config.to(self.model)
        # track epoch progress
        test_epoch_result = EpochResult(0, DatasetSplitType.test)

        if logger.isEnabledFor(logging.INFO):
            logger.info(f'testing model {type(model)} on {model.device}')

        # in for some reason the model was trained but not tested, we'll load
        # from the model file, which will have no train results (bad idea)
        if self.model_result is None:
            self.model_result = self._create_model_result()

        self.model_result.reset(DatasetSplitType.test)
        self.model_result.test.start()
        self.model_result.test.append(test_epoch_result)

        # prep model for evaluation
        model.eval()
        # run the model on test data
        test_epoch_result.start()
        for batch in self._to_iter(batches):
            # forward pass: compute predicted outputs by passing inputs
            # to the model
            with torch.no_grad():
                self.batch_iterator.iterate(
                    model, optimizer, criterion, batch,
                    test_epoch_result, DatasetSplitType.test)
            self._gc(3)
        test_epoch_result.end()

        self._gc(2)

        self.model_result.test.end()

    def _preproces_training(self, ds_train: Tuple[Batch]):
        """Preprocess the training set, which for this method implementation, includes
        a shuffle if configured in the model settings.

        """
        if self.model_settings.shuffle_training:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('shuffling training dataset')
            # data sets are ordered with training as the first
            rand.shuffle(ds_train)

    def _calc_batch_limit(self, src: Stash,
                          batch_limit: Union[int, float]) -> int:
        if batch_limit <= 0:
            raise ModelError(f'Batch limit must be positive: {batch_limit}')
        if isinstance(batch_limit, float):
            if batch_limit > 1.0:
                raise ModelError('Batch limit must be less than 1 ' +
                                 f'when a float: {batch_limit}')
            vlim = round(len(src) * batch_limit)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('batch limit calculated as a percentage: ' +
                             f'{vlim} = {len(src)} * {batch_limit}')
        else:
            vlim = batch_limit
        if isinstance(src, SplitStashContainer):
            desc = f' for {src.split_name}'
        else:
            desc = ''
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'using batch limit: {vlim}{desc}')
        return vlim

    def _prepare_datasets(self, batch_limit: Union[int, float],
                          to_deallocate: List[Batch],
                          ds_src: List[Stash]) -> List[List[Batch]]:
        """Return batches for each data set.  The batches are returned per dataset as
        given in :meth:`_get_dataset_splits`.

        Return:
          [(training batch 1..N), (validation batch 1..N), (test batch 1..N)]

        """
        biter = self.model_settings.batch_iteration
        cnt = 0

        if logger.isEnabledFor(logging.INFO):
            logger.info(f'preparing datasets using iteration: {biter}')

        if biter == 'gpu':
            ds_dst = []
            for src in ds_src:
                vlim = self._calc_batch_limit(src, batch_limit)
                cpu_batches = tuple(it.islice(src.values(), vlim))
                gpu_batches = list(map(lambda b: b.to(), cpu_batches))
                cnt += len(gpu_batches)
                # the `to` call returns the same instance if the tensor is
                # already on the GPU, so only deallocate batches copied over
                for cpu_batch, gpu_batch in zip(cpu_batches, gpu_batches):
                    if cpu_batch is not gpu_batch:
                        to_deallocate.append(cpu_batch)
                if not self.model_settings.cache_batches:
                    to_deallocate.extend(gpu_batches)
                ds_dst.append(gpu_batches)
        elif biter == 'cpu':
            ds_dst = []
            for src in ds_src:
                vlim = self._calc_batch_limit(src, batch_limit)
                batches = list(it.islice(src.values(), vlim))
                cnt += len(batches)
                if not self.model_settings.cache_batches:
                    to_deallocate.extend(batches)
                ds_dst.append(batches)
        elif biter == 'buffered':
            ds_dst = ds_src
            cnt = '?'
        else:
            raise ModelError(f'No such batch iteration method: {biter}')

        self._preproces_training(ds_dst[0])

        return cnt, ds_dst

    def _execute(self, sets_name: str, description: str,
                 func: Callable, ds_src: tuple) -> bool:
        """Either train or test the model based on method ``func``.

        :param sets_name: the name of the data sets, which ``train`` or
                          ``test``

        :param func: the method to call to do the training or testing

        :param ds_src: a tuple of datasets in a form such as ``(train,
                       validation, test)`` (see :meth:`_get_dataset_splits`)

        :return: ``True`` if training/testing was successful, otherwise
                 `the an exception occured or early bail

        """
        to_deallocate: List[Batch] = []
        ds_dst: List[List[Batch]] = None
        batch_limit = self.model_settings.batch_limit
        biter = self.model_settings.batch_iteration

        if self.model_settings.cache_batches and biter == 'buffered':
            raise ModelError('Can not cache batches for batch ' +
                             'iteration setting \'buffered\'')

        if logger.isEnabledFor(logging.INFO):
            logger.info(f'batch iteration: {biter}, limit: {batch_limit}' +
                        f', caching: {self.model_settings.cache_batches}'
                        f', cached: {len(self.cached_batches)}')

        self._gc(1)

        ds_dst = self.cached_batches.get(sets_name)
        if ds_dst is None:
            cnt = 0
            with time('loaded {cnt} batches'):
                cnt, ds_dst = self._prepare_datasets(
                    batch_limit, to_deallocate, ds_src)
            if self.model_settings.cache_batches:
                self.cached_batches[sets_name] = ds_dst

        if logger.isEnabledFor(logging.INFO):
            logger.info('train/test sets: ' +
                        f'{" ".join(map(lambda l: str(len(l)), ds_dst))}')

        try:
            with time(f'executed {sets_name}'):
                func(*ds_dst)
            if description is not None:
                res_name = f'{self.model_result.index}: {description}'
                self.model_result.name = res_name
            return True
        except EarlyBailError as e:
            logger.warning(f'<{e}>')
            self.reset()
            return False
        finally:
            self._train_manager.clear()
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'deallocating {len(to_deallocate)} batches')
            for batch in to_deallocate:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'deallocating: {batch}')
                batch.deallocate()
            self._gc(1)
            self.torch_config.empty_cache()

    def _get_dataset_splits(self) -> List[BatchStash]:
        """Return a stash, one for each respective data set tracked by this executor.

        """
        splits = self.dataset_stash.splits
        return tuple(map(lambda n: splits[n], self.dataset_split_names))

    def train(self, description: str = None) -> ModelResult:
        """Train the model.

        """
        self.model_result = self._create_model_result()
        train, valid, _ = self._get_dataset_splits()
        self._execute('train', description, self._train, (train, valid))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'trained model result: {self.model_result}')
        return self.model_result

    def test(self, description: str = None) -> ModelResult:
        """Test the model.

        """
        train, valid, test = self._get_dataset_splits()
        if self.model_result is None:
            logger.warning('no results found--loading')
            self.model_result = self.result_manager.load()
        self._execute('test', description, self._test, (test,))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'tested model result: {self.model_result}')
        return self.model_result

    def train_production(self, description: str = None) -> ModelResult:
        """Train and test the model on the training and test datasets.  This is used
        for a "production" model that is used for some purpose other than
        evaluation.

        """
        self.model_result = self._create_model_result()
        train, valid, test = self._get_dataset_splits()
        train = UnionStash((train, test))
        self._execute('train production', description,
                      self._train, (train, valid))
        return self.model_result

    def predict(self, batches: List[Batch]) -> ModelResult:
        """Create predictions on ad-hoc data.

        :param batches: contains the data (X) on which to predict

        :return: the results of the predictions

        """
        self._test(batches)
        return self.model_result.test

    def _write_model(self, depth: int, writer: TextIOBase):
        model = self._get_or_create_model()
        sio = StringIO()
        sp = self._sp(depth + 1)
        nl = '\n'
        print(model, file=sio)
        self._write_line('model:', depth, writer)
        writer.write(nl.join(map(lambda s: sp + s, sio.getvalue().split(nl))))

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              include_settings: bool = False, include_model: bool = False):
        sp = self._sp(depth)
        writer.write(f'{sp}model: {self.model_name}\n')
        writer.write(f'{sp}feature splits:\n')
        self.feature_stash.write(depth + 1, writer)
        writer.write(f'{sp}batch splits:\n')
        self.dataset_stash.write(depth + 1, writer)
        if include_settings:
            self._write_line('network settings:', depth, writer)
            self._write_dict(self.net_settings.asdict(), depth + 1, writer)
            self._write_line('model settings:', depth, writer)
            self._write_dict(self.model_settings.asdict(), depth + 1, writer)
        if include_model:
            self._write_model(depth, writer)
