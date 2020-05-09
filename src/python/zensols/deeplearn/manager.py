"""This file contains the network model and data that holds the results.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass
import gc
import itertools as it
import logging
from itertools import chain
from pathlib import Path
from types import FunctionType
from typing import List
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from zensols.util import time
from zensols.persist import Stash
from zensols.deeplearn import (
    EarlyBailException,
    EpochResult,
    NetworkModelResult,
    BatchStash,
    Batch,
)

logger = logging.getLogger(__name__)


@dataclass
class NetworkModelManager(object):
    """This class creates and uses a network to train, validate and test the model.

    """
    batch_stash: BatchStash
    model_result: NetworkModelResult

    def __post_init__(self):
        self.ccnf = self.model_result.net_settings.cuda_config
        self.cfs = self.model_result.cl_settings
        self.net_settings = self.model_result.net_settings

    @property
    def model_path(self) -> Path:
        """The model file used to persist for this configuration.

        """
        return Path(self.cfs.model_path_format.format(
            **{'r': self.model_result}))

    def save_model(self, model) -> Path:
        path = self.model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {'net_settings': self.net_settings,
                      'model_state_dict': model.state_dict()}
        torch.save(checkpoint, str(path))
        return path

    def load_model(self) -> nn.Module:
        """Load the model the last saved model from the disk.

        """
        model_path = self.model_path
        logger.info(f'loading model from {model_path}')
        checkpoint = torch.load(model_path)
        model = self._create_model(checkpoint['net_settings'])
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        return model

    def _create_model(self, net_settings=None) -> nn.Module:
        """Create the network model instance.

        """
        cls = self.cfs.net_class
        if self.net_settings.debug:
            from zensols.actioncli import ClassImporter
            cname = f'{cls.__module__}.{cls.__name__}'
            cls = ClassImporter(cname).get_module_class()[1]
        if net_settings is None:
            net_settings = self.net_settings
        model = cls(net_settings)
        model = self.ccnf.to(model)
        return model

    def get_criterion_optimizer(self, model):
        """Return the loss function and descent optimizer.

        """
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.cfs.learning_rate)
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

    def _train(self, train: List[Batch], valid: List[Batch]) -> bool:
        """Train the network model and record validation and training losses.  Every
        time the validation loss shrinks, the model is saved to disk.

        """
        # create network model, loss and optimization functions
        model = self._create_model()
        criterion, optimizer = self.get_criterion_optimizer(model)

        # set initial "min" to infinity
        valid_loss_min = np.Inf

        # set up graphical progress bar
        pbar = range(self.cfs.epochs)
        if self.cfs.console:
            pbar = tqdm(pbar, ncols=79)

        if self.cfs.use_gc:
            logger.debug('garbage collecting')
            gc.collect()

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

            if self.cfs.use_gc:
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

            if self.cfs.use_gc:
                logger.debug('garbage collecting')
                gc.collect()

            decreased = valid_epoch_result.loss <= valid_loss_min
            dec_str = '\\/' if decreased else '/\\'
            msg = (f'train: {train_epoch_result.loss:.3f}, ' +
                   f'valid: {valid_epoch_result.loss:.3f} {dec_str}')
            logger.debug(msg)
            if self.cfs.console:
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
        self.model = model
        return True

    def _test(self, batches: List[Batch]) -> bool:
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
        return True

    def _train_or_test(self, func: FunctionType, ds_src: tuple) -> bool:
        """Either train or test the model based on method ``func``.

        :return: ``True`` if the training ended successfully

        """

        batch_limit = self.cfs.batch_limit
        logger.debug(f'batch limit: {batch_limit}')

        gc.collect()

        biter = self.cfs.batch_iteration
        if biter == 'cuda':
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
            return func(*ds_dst)
        except EarlyBailException as e:
            logger.warning(f'<{e}>')
            return False
        finally:
            if biter == 'cuda':
                for dst in chain.from_iterable(ds_dst):
                    batch.deallocate()

    def train(self) -> bool:
        """Train the model.

        :return: ``True`` if the training ended successfully

        """
        # create the datasets
        train, valid, test = self.batch_stash.stashes
        return self._train_or_test(self._train, (train, valid))

    def test(self, use_last=False) -> bool:
        """Test the model.

        :return: ``True`` if the training ended successfully

        """
        # create the datasets
        train, valid, test = self.batch_stash.stashes
        self.use_last = use_last
        return self._train_or_test(self._test, (test,))

    def unpersist_results(self) -> NetworkModelResult:
        return self.model_result.load()
