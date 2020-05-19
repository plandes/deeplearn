"""This file contains the network model and data that holds the results.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field, InitVar
from typing import List, Callable, Any, Tuple
import sys
import gc
import logging
import itertools as it
from pathlib import Path
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from zensols.util import time
from zensols.config import Configurable, ConfigFactory, Writable
from zensols.persist import Stash, persisted
from zensols.deeplearn import (
    TorchConfig,
    EarlyBailException,
    EpochResult,
    ModelResult,
    ModelSettings,
    ModelResultManager,
    NetworkSettings,
    DatasetSplitStash,
    BatchStash,
    Batch,
    BaseNetworkModule,
    PersistedWork,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelManager(object):
    path: Path
    config_factory: ConfigFactory
    model_executor_name: str = field(default=None)
    keep_last_state_dict: bool = field(default=False)

    @staticmethod
    def copy_state_dict(state_dict):
        return {k: state_dict[k].clone() for k in state_dict.keys()}

    def save_executor(self, executor: Any, model: BaseNetworkModule,
                      optimizer: torch.optim.Optimizer):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        state_dict = model.state_dict()
        if self.keep_last_state_dict:
            self.last_saved_state_dict = self.copy_state_dict(state_dict)
        checkpoint = {'config_factory': self.config_factory,
                      'model_executor': self.model_executor_name,
                      'model_result': executor.model_result,
                      'model_optim_state_dict': optimizer.state_dict(),
                      'model_state_dict': state_dict}
        torch.save(checkpoint, str(self.path))
        logger.info(f'saved model to {self.path}')

    def update_results(self, executor):
        logger.debug(f'updating results: {self.path}')
        checkpoint = torch.load(str(self.path))
        checkpoint['model_result'] = executor.model_result
        torch.save(checkpoint, str(self.path))
        logger.info(f'saved results to {self.path}')

    def load_state_dict(self):
        checkpoint = torch.load(str(self.path))
        return checkpoint['model_state_dict']

    def load_model(self, net_settings: NetworkSettings,
                   checkpoint: dict = None):
        if checkpoint is None:
            logger.debug(f'loading model from: {self.path}')
            checkpoint = torch.load(str(self.path))
        model: BaseNetworkModule = self.create_module(net_settings)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def load_executor(self):
        """Load the model the last saved model from the disk.

        """
        logger.debug(f'loading model from: {self.path}')
        checkpoint = torch.load(str(self.path))
        logger.debug(f'loaded: {checkpoint.__class__}')
        config_factory = checkpoint['config_factory']
        logger.debug(f'loading config factory: {config_factory}')
        # ModelExecutor
        executor = config_factory.instance(checkpoint['model_executor'])
        model = self.load_model(executor.net_settings, checkpoint)
        executor.model = model
        executor.model_result = checkpoint['model_result']
        optimizer = executor.criterion_optimizer[1]
        optimizer.load_state_dict(checkpoint['model_optim_state_dict'])
        logger.info(f'loaded model from {executor.model_settings.path} ' +
                    f'on device {model.device}')
        return executor

    def create_module(self, net_settings: NetworkSettings) -> BaseNetworkModule:
        """Create the network model instance.

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


@dataclass
class ModelExecutor(Writable):
    """This class creates and uses a network to train, validate and test the model.

    :param config_factory: the configuration factory that created this instance

    :param config: the configuration used in the configuration factory to
                   create this instance

    :param net_settings: the settings used to configure the network

    :param model_name: a human readable name for the model

    :param model_settings: the configuration of the model

    :param net_settings: the configuration of the model's network

    :param dataset_stash: the split data set stash that contains the
                         ``BatchStash``, which contains the batches on which to
                         train and test

    :param dataset_split_names: the list of split names in the
                                ``dataset_stash`` in the order: train,
                                validation, test (see ``_get_dataset_splits``)

    :param result_path: if not ``None``, a path to a directory where the
                        results are to be dumped; the directory will be created
                        if it doesn't exist when the results are generated

    :param progress_bar: create text based progress bar if ``True``

    """
    config_factory: ConfigFactory
    config: Configurable
    name: str
    model_name: str
    model_settings: ModelSettings
    net_settings: NetworkSettings
    dataset_stash: DatasetSplitStash
    dataset_split_names: List[str]
    result_path: Path = field(default=None)
    progress_bar: bool = field(default=False)
    model: InitVar[BaseNetworkModule] = field(default=None)

    def __post_init__(self, model: BaseNetworkModule):
        self._model = model
        self.model_result: ModelResult = None
        self.batch_stash.delegate_attr: bool = True
        self._criterion_optimizer = PersistedWork('_criterion_optimizer', self)
        self._result_manager = PersistedWork('_result_manager', self)

    @property
    def batch_stash(self):
        return self.dataset_stash.split_container

    @property
    def feature_stash(self) -> Stash:
        """Return the stash used to generate the feature, which is not to be confused
        with the batch source stash``batch_stash``.

        """
        return self.batch_stash.split_stash_container

    @property
    def torch_config(self) -> TorchConfig:
        return self.batch_stash.model_torch_config

    @property
    @persisted('_result_manager')
    def result_manager(self) -> ModelResultManager:
        if self.result_path is not None:
            return ModelResultManager(
                name=self.model_name, path=self.result_path)

    @property
    @persisted('_model_manager')
    def model_manager(self):
        return ModelManager(
            self.model_settings.path, self.config_factory, self.name)

    def load_model(self):
        self.reset()
        self._model = self.model_manager.load_model(self.net_settings)

    @property
    def model(self) -> BaseNetworkModule:
        if self._model is None:
            raise ValueError('no model, is populated; use \'load_model\'')
        return self._model

    @model.setter
    def model(self, model: BaseNetworkModule):
        self._model = model
        self._criterion_optimizer.clear()

    def create_model(self) -> BaseNetworkModule:
        """Create the network model instance.

        """
        model = self.model_manager.create_module(self.net_settings)
        logger.info(f'create model on {model.device} with {self.torch_config}')
        return model

    @property
    @persisted('_criterion_optimizer')
    def criterion_optimizer(self) -> Tuple[nn.L1Loss, torch.optim.Optimizer]:
        """Return the loss function and descent optimizer.

        """
        return self._create_criterion_optimizer()

    def _create_criterion_optimizer(self) -> Tuple[nn.L1Loss, torch.optim.Optimizer]:
        """Factory method to create the loss function and optimizer.

        """
        model = self.model
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.model_settings.learning_rate)
        return criterion, optimizer

    def reset(self):
        self._criterion_optimizer.clear()
        self._result_manager.clear()
        self._model = None

    def _decode_outcomes(self, outcomes: np.ndarray) -> np.ndarray:
        # get the indexes of the max value across labels and outcomes
        return outcomes.argmax(1)

    def _train_batch(self, model: BaseNetworkModule, optimizer, criterion,
                     batch: Batch, epoch_result: EpochResult,
                     split_type: str):
        """Train on a batch.  This uses the back propogation algorithm on training and
        does a simple feed forward on validation and testing.

        """
        logger.debug(f'train/validate on {split_type}: batch={batch}')
        batch = batch.to()
        labels = batch.get_labels()
        label_shapes = labels.shape
        if split_type == 'train':
            optimizer.zero_grad()
        # forward pass, get our log probs
        output = model(batch)
        if output is None:
            raise ValueError('model output')
        # calculate the loss with the logps and the labels
        labels = labels.float()
        loss = criterion(output, labels)
        if split_type == 'train':
            # invoke back propogation on the network
            loss.backward()
            # take an update step and update the new weights
            optimizer.step()
        labels = self._decode_outcomes(labels)
        output = self._decode_outcomes(output)
        epoch_result.update(batch, loss, labels, output, label_shapes)

    def _to_iter(self, ds):
        ds_iter = ds
        if isinstance(ds_iter, Stash):
            ds_iter = ds_iter.values()
        return ds_iter

    def _train(self, train: List[Batch], valid: List[Batch]):
        """Train the network model and record validation and training losses.  Every
        time the validation loss shrinks, the model is saved to disk.

        """
        # create network model, loss and optimization functions
        model = self.create_model()
        model = self.torch_config.to(model)
        self.model = model
        criterion, optimizer = self.criterion_optimizer

        # set initial "min" to infinity
        valid_loss_min = np.Inf

        # set up graphical progress bar
        pbar = range(self.model_settings.epochs)
        progress_bar = self.progress_bar and \
            (logger.level == 0 or logger.level > logging.INFO)
        if progress_bar:
            pbar = tqdm(pbar, ncols=79)

        logger.info(f'training model {model} on {model.device}')

        if self.model_settings.use_gc:
            logger.debug('garbage collecting')
            gc.collect()

        self.model_result.train.start()

        # loop over epochs
        for epoch in pbar:
            logger.debug(f'training on epoch: {epoch}')

            train_epoch_result = EpochResult(epoch, 'train')
            valid_epoch_result = EpochResult(epoch, 'validation')

            self.model_result.train.append(train_epoch_result)
            self.model_result.validation.append(valid_epoch_result)

            # prep model for training and train
            model.train()
            for batch in self._to_iter(train):
                logger.debug(f'training on batch: {batch.id}')
                with time('trained batch', level=logging.DEBUG):
                    self._train_batch(model, optimizer, criterion, batch,
                                      train_epoch_result, 'train')

            if self.model_settings.use_gc:
                logger.debug('garbage collecting')
                gc.collect()

            # prep model for evaluation and evaluate
            model.eval()
            for batch in self._to_iter(valid):
                # forward pass: compute predicted outputs by passing inputs
                # to the model
                with torch.no_grad():
                    self._train_batch(model, optimizer, criterion, batch,
                                      valid_epoch_result, 'validation')

            if self.model_settings.use_gc:
                logger.debug('garbage collecting')
                gc.collect()

            decreased = valid_epoch_result.loss <= valid_loss_min
            dec_str = '\\/' if decreased else '/\\'
            msg = (f'train: {train_epoch_result.loss:.3f}, ' +
                   f'valid: {valid_epoch_result.loss:.3f} {dec_str}')
            logger.debug(msg)
            if progress_bar:
                pbar.set_description(msg)
            else:
                logger.info(f'epoch: {epoch}, {msg}')

            # save model if validation loss has decreased
            if decreased:
                logger.info(f'validation loss decreased ' +
                            f'({valid_loss_min:.6f}' +
                            f'-> {valid_epoch_result.loss:.6f}); saving model')
                self.model_manager.save_executor(self, model, optimizer)
                self.model_result.validation_loss = valid_epoch_result.loss
                valid_loss_min = valid_epoch_result.loss
            else:
                logger.info(f'validation loss increased ' +
                            f'({valid_loss_min:.6f}' +
                            f'-> {valid_epoch_result.loss:.6f})')

        self.model_result.train.end()
        self.model_manager.update_results(self)
        self.model = model

    def _test(self, batches: List[Batch]):
        """Test the model on the test set.

        If a model is not given, it is unpersisted from the file system.

        """
        # create the loss and optimization functions
        criterion, optimizer = self.criterion_optimizer
        model = self.torch_config.to(self.model)
        # track epoch progress
        test_epoch_result = EpochResult(0, 'test')

        if 1:
            self.model_result.reset('test')
        else:
            if self.model_result.test.contains_results:
                raise ValueError(f'duplicating test of {self.model_result} ' +
                                 f'in {self.name}')
        self.model_result.test.start()
        self.model_result.test.append(test_epoch_result)

        # prep model for evaluation
        model.eval()
        # run the model on test data
        for batch in self._to_iter(batches):
            # forward pass: compute predicted outputs by passing inputs
            # to the model
            with torch.no_grad():
                self._train_batch(model, optimizer, criterion, batch,
                                  test_epoch_result, 'test')

        self.model_result.test.end()

    def _train_or_test(self, func: Callable, ds_src: tuple):
        """Either train or test the model based on method ``func``.

        :return: ``True`` if the training ended successfully

        """
        batch_limit = self.model_settings.batch_limit
        logger.debug(f'batch limit: {batch_limit}')

        gc.collect()

        biter = self.model_settings.batch_iteration
        if biter == 'gpu':
            ds_dst = []
            for src in ds_src:
                batches = map(lambda b: b.to(), src.values())
                ds_dst.append(tuple(it.islice(batches, batch_limit)))
        elif biter == 'cpu':
            ds_dst = []
            for src in ds_src:
                ds_dst.append(tuple(it.islice(src.values(), batch_limit)))
        elif biter == 'buffer':
            ds_dst = ds_src
        else:
            raise ValueError(f'no such batch iteration method: {biter}')

        logger.info('train [,test] sets: ' +
                    f'{" ".join(map(lambda l: str(len(l)), ds_dst))}')

        try:
            func(*ds_dst)
            return self.model_result
        except EarlyBailException as e:
            logger.warning(f'<{e}>')
            self.reset()
            return
        finally:
            if ds_dst is not None:
                del ds_dst

    def _get_dataset_splits(self) -> List[BatchStash]:
        splits = self.dataset_stash.splits
        return tuple(map(lambda n: splits[n], self.dataset_split_names))

    def _assert_model_result(self, force=False):
        if self.model_result is None or force:
            self.model_result = ModelResult(
                self.config, self.model_name,
                self.model_settings, self.net_settings)

    def train(self) -> ModelResult:
        """Train the model.

        """
        self._assert_model_result(True)
        train, valid, test = self._get_dataset_splits()
        self._train_or_test(self._train, (train, valid))
        return self.model_result

    def test(self) -> ModelResult:
        """Test the model.

        """
        train, valid, test = self._get_dataset_splits()
        self._train_or_test(self._test, (test,))
        if self.result_manager is not None:
            self.result_manager.dump(self.model_result)
        return self.model_result

    def write(self, depth: int = 0, writer=sys.stdout):
        sp = self._sp(depth)
        writer.write(f'{sp}feature splits:\n')
        self.feature_stash.write(depth + 1, writer)
        writer.write(f'{sp}batch splits:\n')
        self.dataset_stash.write(depth + 1, writer)
