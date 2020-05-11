"""This file contains the network model and data that holds the results.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field
from typing import List, Callable
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
)

logger = logging.getLogger(__name__)


@dataclass
class ModelManager(Writable):
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
    model_name: str
    model_settings: ModelSettings
    net_settings: NetworkSettings
    dataset_stash: DatasetSplitStash
    dataset_split_names: List[str]
    result_path: Path = field(default=None)
    progress_bar: bool = field(default=False)

    def __post_init__(self):
        self.model_result: ModelResult = None
        # if 'train' not in self.dataset_split_names:
        #     raise ValueError(f'at least one split must be named "train"')
        # allow attribute dispatch to actual BatchStash as this instance is a
        # DatasetSplitStash
        self.batch_stash.delegate_attr: bool = True

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
            return ModelResultManager(self.model_name, self.result_path)

    def save_model(self, model: nn.Module):
        path = self.model_settings.path
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {'net_settings': self.net_settings,
                      'model_state_dict': model.state_dict()}
        torch.save(checkpoint, str(path))
        logger.info(f'saved model to {path}')

    def load_model(self) -> nn.Module:
        """Load the model the last saved model from the disk.

        """
        checkpoint = torch.load(self.model_settings.path)
        model = self.create_model(checkpoint['net_settings'])
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model = self.torch_config.to(model)
        logger.info(f'loaded model from {self.model_settings.path} ' +
                    f'on device {model.device}')
        return model

    def create_model(self, net_settings: NetworkSettings = None) -> nn.Module:
        """Create the network model instance.

        """
        net_settings = self.net_settings if net_settings is None else net_settings
        cls_name = net_settings.get_module_class_name()
        resolver = self.config_factory.class_resolver
        initial_reload = resolver.reload
        try:
            resolver.reload = net_settings.debug
            cls = resolver.find_class(cls_name)
        finally:
            resolver.reload = initial_reload
        model = cls(self.net_settings)
        model = self.torch_config.to(model)
        logger.info(f'create model on {model.device} with {self.torch_config}')
        return model

    def get_criterion_optimizer(self, model: nn.Module):
        """Return the loss function and descent optimizer.

        """
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.model_settings.learning_rate)
        return criterion, optimizer

    def _decode_outcomes(self, outcomes: np.ndarray) -> np.ndarray:
        # get the indexes of the max value across labels and outcomes
        return outcomes.argmax(1)

    def _train_batch(self, model: nn.Module, optimizer, criterion,
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
        model = self.create_model(self.net_settings)
        criterion, optimizer = self.get_criterion_optimizer(model)

        # set initial "min" to infinity
        valid_loss_min = np.Inf

        # set up graphical progress bar
        pbar = range(self.model_settings.epochs)
        progress_bar = self.progress_bar and \
            (logger.level == 0 or logger.level > logging.INFO)
        if progress_bar:
            pbar = tqdm(pbar, ncols=79)

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

            # prep model for training
            model.train()

            # train the model
            for batch in self._to_iter(train):
                logger.debug(f'training on batch: {batch.id}')
                with time('trained batch', level=logging.DEBUG):
                    self._train_batch(model, optimizer, criterion, batch,
                                      train_epoch_result, 'train')

            if self.model_settings.use_gc:
                logger.debug('garbage collecting')
                gc.collect()

            # prep model for evaluation
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
                self.save_model(model)
                self.model_result.validation_loss = valid_epoch_result.loss
                valid_loss_min = valid_epoch_result.loss
            else:
                logger.info(f'validation loss increased ' +
                            f'({valid_loss_min:.6f}' +
                            f'-> {valid_epoch_result.loss:.6f})')

        self.model_result.train.end()

        # save the model for testing later
        self.model = model

    def _test(self, batches: List[Batch]):
        """Test the model on the test set.

        If a model is not given, it is unpersisted from the file system.

        """
        if self.use_last:
            model = self.model
        else:
            model = self.load_model()
        # create the loss and optimization functions
        criterion, optimizer = self.get_criterion_optimizer(model)
        # track epoch progress
        test_epoch_result = EpochResult(0, 'test')

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
            return False
        finally:
            if ds_dst is not None:
                del ds_dst

    def _get_dataset_splits(self) -> List[BatchStash]:
        splits = self.dataset_stash.splits
        return tuple(map(lambda n: splits[n], self.dataset_split_names))

    def train(self) -> ModelResult:
        """Train the model.

        """
        self.model_result = ModelResult(
            self.config, self.model_name,
            self.model_settings, self.net_settings)
        train, valid, test = self._get_dataset_splits()
        self._train_or_test(self._train, (train, valid))
        return self.model_result

    def test(self, use_last: bool = False) -> ModelResult:
        """Test the model.

        """
        train, valid, test = self._get_dataset_splits()
        self.use_last = use_last
        self._train_or_test(self._test, (test,))
        if self.result_manager is not None:
            self.result_manager.dump(self.model_result)
        return self.model_result

    def write(self, depth: int = 0, writer=sys.stdout):
        sp = self._sp(depth)
        writer.write(f'{sp}feature splits:\n')
        self.feature_stash.write(depth + 1, writer)
        # writer.write('----\n')
        writer.write(f'{sp}batch splits:\n')
        self.dataset_stash.write(depth + 1, writer)
