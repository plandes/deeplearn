"""This file contains the network model and data that holds the results.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass
from typing import List, Callable
import gc
import itertools as it
import logging
from itertools import chain
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from zensols.util import time
from zensols.config import Configurable, ConfigFactory
from zensols.persist import Stash
from zensols.deeplearn import (
    TorchConfig,
    EarlyBailException,
    EpochResult,
    ModelResult,
    ModelSettings,
    NetworkSettings,
    BatchStash,
    Batch,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelManager(object):
    """This class creates and uses a network to train, validate and test the model.

    :param net_settings: the settings used to configure the network
    :param debug: if ``True``, raise an error on the first forward pass

    """
    config_factory: ConfigFactory
    config: Configurable
    model_settings: ModelSettings
    net_settings: NetworkSettings
    batch_stash: BatchStash
    dataset_split_names: List[str]

    def __post_init__(self):
        # allow attribute dispatch to actual BatchStash as this instance is a
        # DatasetSplitStash
        self.batch_stash.delegate_attr = True

    @property
    def torch_config(self) -> TorchConfig:
        return self.batch_stash.model_torch_config

    def save_model(self, model: nn.Module):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {'model_settings': self.net_settings,
                      'model_state_dict': model.state_dict()}
        torch.save(checkpoint, str(self.path))

    def load_model(self) -> nn.Module:
        """Load the model the last saved model from the disk.

        """
        logger.info(f'loading model from {self.model_settings.path}')
        checkpoint = torch.load(self.model_settings.path)
        model = self._create_model(checkpoint['net_settings'])
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model = self.torch_config.to(model)
        return model

    def _create_model(self, net_settings: NetworkSettings) -> nn.Module:
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
        model = cls(self.net_settings)
        model = self.torch_config.to(model)
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

    def _train(self, train: List[Batch], valid: List[Batch],
               model_result: ModelResult):
        """Train the network model and record validation and training losses.  Every
        time the validation loss shrinks, the model is saved to disk.

        """
        # create network model, loss and optimization functions
        model = self._create_model(self.net_settings)
        criterion, optimizer = self.get_criterion_optimizer(model)

        # set initial "min" to infinity
        valid_loss_min = np.Inf

        # set up graphical progress bar
        pbar = range(self.model_settings.epochs)
        if self.model_settings.console:
            pbar = tqdm(pbar, ncols=79)

        if self.model_settings.use_gc:
            logger.debug('garbage collecting')
            gc.collect()

        # loop over epochs
        for epoch in pbar:
            logger.debug(f'training on epoch: {epoch}')

            train_epoch_result = EpochResult(epoch, 'train')
            valid_epoch_result = EpochResult(epoch, 'validation')

            model_result.train.append(train_epoch_result)
            model_result.validation.append(valid_epoch_result)

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
            if self.model_settings.console:
                pbar.set_description(msg)
            else:
                logger.info(f'epoch: {epoch}, {msg}')

            # save model if validation loss has decreased
            if decreased:
                logger.info(f'validation loss decreased ' +
                            f'({valid_loss_min:.6f}' +
                            f'-> {valid_epoch_result.loss:.6f}); saving model')
                self.save_model(model)
                model_result.validation_loss = valid_epoch_result.loss
                valid_loss_min = valid_epoch_result.loss
            else:
                logger.info(f'validation loss increased ' +
                            f'({valid_loss_min:.6f}' +
                            f'-> {valid_epoch_result.loss:.6f})')
        # save the model for testing later
        self.model = model

    def _test(self, batches: List[Batch], model_result: ModelResult):
        """Test the model on the test set.

        If a model is not given, it is unpersisted from the file system.

        """
        model = self.model
        # create the loss and optimization functions
        criterion, optimizer = self.get_criterion_optimizer(model)
        # track epoch progress
        test_epoch_result = EpochResult(0, 'test')
        model_result.test.append(test_epoch_result)

        # prep model for evaluation
        model.eval()
        # run the model on test data
        for batch in self._to_iter(batches):
            # forward pass: compute predicted outputs by passing inputs
            # to the model
            with torch.no_grad():
                self._train_batch(model, optimizer, criterion, batch,
                                  test_epoch_result, 'test')

    def _train_or_test(self, func: Callable, ds_src: tuple) -> ModelResult:
        """Either train or test the model based on method ``func``.

        :return: ``True`` if the training ended successfully

        """
        model_result = ModelResult(
            self.config, self.model_settings, self.net_settings)
        batch_limit = self.model_settings.batch_limit
        logger.debug(f'batch limit: {batch_limit}')

        gc.collect()

        biter = self.model_settings.batch_iteration
        if biter == 'gpu':
            ds_dst = ([], [])
            for src, dst in zip(ds_src, ds_dst):
                for batch in it.islice(src.values(), batch_limit):
                    dst.append(batch.to())
        elif biter == 'list':
            ds_dst = []
            for src in ds_src:
                ds_dst.append(tuple(it.islice(src.values(), batch_limit)))
        elif biter == 'iter':
            ds_dst = ds_src
        else:
            raise ValueError('no such batch iteration method: {biter}')
        try:
            func(*ds_dst, model_result=model_result)
            return model_result
        except EarlyBailException as e:
            logger.warning(f'<{e}>')
            return False
        finally:
            if biter == 'cuda':
                for dst in chain.from_iterable(ds_dst):
                    batch.deallocate()

    def _get_dataset_splits(self) -> List[BatchStash]:
        splits = self.batch_stash.splits
        return tuple(map(lambda n: splits[n], self.dataset_split_names))

    def train(self) -> bool:
        """Train the model.

        :return: ``True`` if the training ended successfully

        """
        train, valid, test = self._get_dataset_splits()
        return self._train_or_test(self._train, (train, valid))

    def test(self) -> bool:
        """Test the model.

        :return: ``True`` if the training ended successfully

        """
        train, valid, test = self._get_dataset_splits()
        return self._train_or_test(self._test, (test,))
