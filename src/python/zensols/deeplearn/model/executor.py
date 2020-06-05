"""This file contains the network model and data that holds the results.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field, InitVar
from typing import List, Callable, Tuple, Any
import sys
import gc
import logging
import copy as cp
import itertools as it
from itertools import chain
from pathlib import Path
import numpy as np
import torch
from torch import nn
import pandas as pd
from tqdm import tqdm
from zensols.util import time
from zensols.config import Configurable, ConfigFactory, Writable
from zensols.persist import (
    persisted, PersistedWork, PersistableContainer,
    Stash
)
from zensols.dataset import DatasetSplitStash
from zensols.deeplearn import TorchConfig, EarlyBailException, NetworkSettings
from zensols.deeplearn.result import (
    EpochResult,
    ModelResult,
    ModelSettings,
    ModelResultManager,
    PredictionsDataFrameFactory,
)
from zensols.deeplearn.batch import (
    BatchStash,
    DataPoint,
    Batch,
)
from . import BaseNetworkModule, ModelManager

logger = logging.getLogger(__name__)


@dataclass
class ModelExecutor(PersistableContainer, Writable):
    """This class creates and uses a network to train, validate and test the model.
    This class is either configured using a
    :class:`zensols.config.ConfigFactory` or is unpickled with
    :class:`zensols.deeplearn.model.ModelManager`.  If the later, it's from a
    previously trained (and possibly tested) state.

    Typically, after creating a nascent instance, :meth:`train` is called to
    train the model.  This returns the results, but the results are also
    available via the :class:`ResultManager` using the
    :py:attr:`~model_manager` property.  To load previous results, use
    ``executor.result_manager.load()``.

    During training, the training set is used to train the weights of the model
    provided by the executor in the :py:attr:`~model_settings`, then validated
    using the validation set.  When the validation loss is minimized, the
    following is saved to disk:

        * Settings: :py:attr:`~net_settings`, :py:attr:`~model_settings`,
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

    :param progress_bar: create text/ASCII based progress bar if ``True``

    :param progress_bar_cols: the number of console columns to use for the
                              text/ASCII based progress bar

    :param model: the base module to use if any; **imporant**: it's better to
                  not specify the model to allow the model manager to dictate
                  the model lifecycle

    :see: :class:`zensols.deeplearn.model.ModelExecutor`
    :see: :class:`zensols.deeplearn.model.NetworkSettings`
    :see: :class:`zensols.deeplearn.model.ModelSettings`

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
    progress_bar_cols: int = field(default=79)
    model: InitVar[BaseNetworkModule] = field(default=None)

    def __post_init__(self, model: BaseNetworkModule):
        if not isinstance(self.dataset_stash, DatasetSplitStash) and False:
            raise ValueError('expecting type DatasetSplitStash but ' +
                             f'got {self.dataset_stash.__class__}')
        self._model = model
        self.model_result: ModelResult = None
        self.batch_stash.delegate_attr: bool = True
        self._criterion_optimizer = PersistedWork('_criterion_optimizer', self)
        self._result_manager = PersistedWork('_result_manager', self)

    @property
    def batch_stash(self) -> DatasetSplitStash:
        """Return the stash used to obtain the data for training and testing.  This
        stash should have a training, validation and test splits.  The names of
        these splits are given in the ``dataset_split_names``.

        """
        return self.dataset_stash.split_container

    @property
    def feature_stash(self) -> Stash:
        """Return the stash used to generate the feature, which is not to be confused
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
            return ModelResultManager(
                name=self.model_name, path=self.result_path)

    @property
    @persisted('_model_manager')
    def model_manager(self):
        """Return the manager used for controlling the lifecycle of the model.

        """
        return ModelManager(
            self.model_settings.path, self.config_factory, self.name)

    def load(self):
        """Load the state of the model from the last time it was trained (if it was
        trained).  The state of the model is then ready for testing.

        When the model is trained, that state is not saved the state loaded
        with this model.  Instead, the results are then saved off with the
        ``ModelResultManager``.

        """
        executor = self.model_manager.load_executor()
        self.__dict__ = executor.__dict__

    def reset(self):
        """Clear all results and trained state.

        *Note:* that this nulls out the :py:attrib:`~model` if any was given in
        the initializer (see class docs).

        """
        self._model = None
        self._get_persistable_metadata().clear()
        self.config_factory = cp.copy(self.config_factory)
        self.config_factory.clear()
        executor = self.config_factory.instance(self.name)
        self.model_settings = executor.model_settings
        self.net_settings = executor.net_settings

    @property
    def model(self) -> BaseNetworkModule:
        """Get the PyTorch module that is used for training and test.

        """
        if self._model is None:
            raise ValueError('no model, is populated; use \'load\'')
        return self._model

    @model.setter
    def model(self, model: BaseNetworkModule):
        """Set the PyTorch module that is used for training and test.

        """
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

    def _create_criterion_optimizer(self) -> \
            Tuple[nn.L1Loss, torch.optim.Optimizer]:
        """Factory method to create the loss function and optimizer.

        """
        model = self.model
        resolver = self.config_factory.class_resolver
        criterion_class_name = self.model_settings.criterion_class_name
        logger.debug(f'criterion: {criterion_class_name}')
        criterion_class = resolver.find_class(criterion_class_name)
        criterion = criterion_class()
        optimizer_class_name = self.model_settings.optimizer_class_name
        logger.debug(f'optimizer: {optimizer_class_name}')
        optimizer_class = resolver.find_class(optimizer_class_name)
        optimizer = optimizer_class(
            model.parameters(),
            lr=self.model_settings.learning_rate)
        logger.debug(f'criterion={criterion}, optimizer={optimizer}')
        return criterion, optimizer

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

    def _decode_outcomes(self, outcomes: torch.Tensor) -> torch.Tensor:
        """Transform the model output in to a result to be added to the
        ``EpochResult``, which composes a ``ModelResult``.

        """
        # get the indexes of the max value across labels and outcomes
        return outcomes.argmax(1)

    def _train_batch(self, model: BaseNetworkModule, optimizer, criterion,
                     batch: Batch, epoch_result: EpochResult,
                     split_type: str):
        """Train on a batch.  This uses the back propogation algorithm on training and
        does a simple feed forward on validation and testing.

        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'train/validate on {split_type}: ' +
                         f'batch={batch} ({id(batch)})')
        batch = batch.to()
        labels = batch.get_labels()
        label_shapes = labels.shape
        if split_type == ModelResult.TRAIN_DS_NAME:
            optimizer.zero_grad()
        # forward pass, get our log probs
        output = model(batch)
        if output is None:
            raise ValueError('null model output')
        if not self.model_settings.nominal_labels:
            labels = labels.float()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'input: labels={labels.shape} (labels.dtype), ' +
                         f'output={output.shape} (output.dtype)')
        # calculate the loss with the logps and the labels
        loss = criterion(output, labels)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'split: {split_type}, loss: {loss}')
        if split_type == ModelResult.TRAIN_DS_NAME:
            # invoke back propogation on the network
            loss.backward()
            # take an update step and update the new weights
            optimizer.step()
        if not self.model_settings.nominal_labels:
            labels = self._decode_outcomes(labels)
        output = self._decode_outcomes(output)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'input: labels={labels.shape} (labels.dtype), ' +
                         f'output={output.shape} (output.dtype)')
        epoch_result.update(batch, loss, labels, output, label_shapes)
        return loss

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
            pbar = tqdm(pbar, ncols=self.progress_bar_cols)

        logger.info(f'training model {model} on {model.device}')

        if self.model_settings.use_gc:
            logger.debug('garbage collecting')
            gc.collect()

        self.model_result.train.start()

        # loop over epochs
        for epoch in pbar:
            logger.debug(f'training on epoch: {epoch}')

            train_epoch_result = EpochResult(epoch, ModelResult.TRAIN_DS_NAME)
            valid_epoch_result = EpochResult(epoch, ModelResult.VALIDATION_DS_NAME)

            self.model_result.train.append(train_epoch_result)
            self.model_result.validation.append(valid_epoch_result)

            # train ----
            # prep model for training and train
            model.train()
            for batch in self._to_iter(train):
                logger.debug(f'training on batch: {batch.id}')
                with time('trained batch', level=logging.DEBUG):
                    self._train_batch(
                        model, optimizer, criterion, batch,
                        train_epoch_result, ModelResult.TRAIN_DS_NAME)

            if self.model_settings.use_gc:
                logger.debug('garbage collecting')
                gc.collect()

            # validate ----
            # prep model for evaluation and evaluate
            vloss = 0
            model.eval()
            for batch in self._to_iter(valid):
                # forward pass: compute predicted outputs by passing inputs
                # to the model
                with torch.no_grad():
                    loss = self._train_batch(
                        model, optimizer, criterion, batch,
                        valid_epoch_result, ModelResult.VALIDATION_DS_NAME)
                    vloss += (loss.item() * batch.size())
            vloss = vloss / len(valid)

            if self.model_settings.use_gc:
                logger.debug('garbage collecting')
                gc.collect()

            valid_loss = valid_epoch_result.ave_loss

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'vloss / valid_loss {vloss}/{valid_loss}, ' +
                             f'valid size: {len(valid)}, ' +
                             f'losses: {len(valid_epoch_result.losses)}')

            decreased = valid_loss <= valid_loss_min
            dec_str = '\\/' if decreased else '/\\'
            assert abs(vloss - valid_loss) < 1e-10
            msg = (f'train: {train_epoch_result.ave_loss:.3f}|' +
                   f'valid: {valid_loss:.3f}/{valid_loss_min:.3f} {dec_str}')
            if progress_bar:
                logger.debug(msg)
                pbar.set_description(msg)
            else:
                logger.info(f'epoch: {epoch}, {msg}')

            # save model if validation loss has decreased
            if decreased:
                logger.info('validation loss decreased ' +
                            f'({valid_loss_min:.6f}' +
                            f'-> {valid_loss:.6f}); saving model')
                self.model_manager.save_executor(self)
                valid_loss_min = valid_loss
            else:
                logger.info('validation loss increased ' +
                            f'({valid_loss_min:.6f}' +
                            f'-> {valid_loss:.6f})')

        logger.info(f'final validation min loss: {valid_loss_min}')
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
        test_epoch_result = EpochResult(0, ModelResult.TEST_DS_NAME)

        if 1:
            self.model_result.reset(ModelResult.TEST_DS_NAME)
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
                                  test_epoch_result, ModelResult.TEST_DS_NAME)

        self.model_result.test.end()

    def _train_or_test(self, func: Callable, ds_src: tuple):
        """Either train or test the model based on method ``func``.

        :return: ``True`` if the training ended successfully

        """
        batch_limit = self.model_settings.batch_limit
        biter = self.model_settings.batch_iteration
        logger.debug(f'batch limit: {batch_limit} using iteration: {biter}')

        if self.model_settings.use_gc:
            logger.debug('garbage collecting')
            gc.collect()

        ds_dst = None
        to_deallocate = []
        with time('loaded {cnt} batches'):
            cnt = 0
            if biter == 'gpu':
                ds_dst = []
                for src in ds_src:
                    vals = tuple(it.islice(src.values(), batch_limit))
                    to_deallocate.extend(vals)
                    batches = tuple(map(lambda b: b.to(), vals))
                    to_deallocate.extend(batches)
                    cnt += len(batches)
                    ds_dst.append(batches)
            elif biter == 'cpu':
                ds_dst = []
                for src in ds_src:
                    batches = tuple(it.islice(src.values(), batch_limit))
                    to_deallocate.extend(batches)
                    cnt += len(batches)
                    ds_dst.append(batches)
            elif biter == 'buffered':
                ds_dst = ds_src
                cnt = '?'
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
            logger.debug('deallocating batches')
            for batch in to_deallocate:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'deallocating: {batch}')
                batch.deallocate()
            if self.model_settings.use_gc:
                logger.debug('garbage collecting')
                gc.collect()

    def _get_dataset_splits(self) -> List[BatchStash]:
        """Return a stash, one for each respective data set tracked by this executor.

        """
        splits = self.dataset_stash.splits
        return tuple(map(lambda n: splits[n], self.dataset_split_names))

    def _assert_model_result(self, force=False):
        """Create the :class:`zensols.deeplearn.result.ModelResult` container class (if
        it doesn't already) that will be used to contains the results

        """
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

    def get_predictions(self, column_names: List[str] = None,
                        transform: Callable[[DataPoint], tuple] = None,
                        name: str = None) -> pd.DataFrame:
        """Generate Pandas dataframe containing all predictinos from the test data set.

        :param column_names: the list of string column names for each data item
                             the list returned from ``data_point_transform`` to
                             be added to the results for each label/prediction

        :param transform: a function that returns a tuple, each with an element
                          respective of ``column_names`` to be added to the
                          results for each label/prediction; if ``None`` (the
                          default), ``str`` used

        :param name: the key of the previously saved results to fetch the
                     results, or ``None`` (the default) to get the last result
                     set saved

        """
        if name is None and self.model_result is not None and \
           self.model_result.test.contains_results:
            logger.info('using current results')
            res = self.model_result
        else:
            logger.info(f'loading results from {name}')
            res = self.result_manager.load(name)
        if not res.test.contains_results:
            raise ValueError('no test results found')
        res: EpochResult = res.test.results[0]
        df_fac = PredictionsDataFrameFactory(
            res, self.batch_stash, column_names, transform)
        return df_fac.dataframe

    def write(self, depth: int = 0, writer=sys.stdout):
        sp = self._sp(depth)
        writer.write(f'{sp}feature splits:\n')
        self.feature_stash.write(depth + 1, writer)
        writer.write(f'{sp}batch splits:\n')
        self.dataset_stash.write(depth + 1, writer)
