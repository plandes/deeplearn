"""Contains a class to assist in the batch loop during training, validation and
testing.

"""
__author__ = 'Paul Landes'

from typing import Any
from dataclasses import dataclass, InitVar
import logging
from logging import Logger
import torch
from torch import Tensor
from zensols.deeplearn import EarlyBailException
from zensols.deeplearn.result import (
    EpochResult,
    ModelResult,
)
from zensols.deeplearn.batch import Batch, MetadataNetworkSettings
from . import BaseNetworkModule, ScoredNetworkModule


@dataclass
class BatchIterator(object):
    """This class assists in the batch loop during training, validation and
    testing.  Any special handling of a model related to its loss function can
    be overridden in this class.

    :params executor: the owning executor

    :params logger: the status logger from the executor

    """

    executor: InitVar[Any]
    logger: Logger

    def __post_init__(self, executor: Any):
        self.model_settings = executor.model_settings
        self.net_settings = executor.net_settings
        self.torch_config = executor.torch_config

    def _decode_outcomes(self, outcomes: Tensor) -> Tensor:
        """Transform the model output (and optionally the labels) that will be added to
        the ``EpochResult``, which composes a ``ModelResult``.

        This implementation returns :py:meth:~`Tensor.argmax`, which are
        the indexes of the max value across columns.

        """
        logger = self.logger
        reduce_outcomes = self.model_settings.reduce_outcomes
        # get the indexes of the max value across labels and outcomes (for the
        # descrete classification case)
        if reduce_outcomes == 'argmax':
            res = outcomes.argmax(dim=-1)
        # softmax over each outcome
        elif reduce_outcomes == 'softmax':
            res = outcomes.softmax(dim=-1)
        elif reduce_outcomes == 'none':
            # leave when nothing, prediction/regression measure is used
            res = outcomes
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'argmax outcomes: {outcomes.shape} -> {res.shape}')
        return res

    def _encode_labels(self, labels: Tensor) -> Tensor:
        logger = self.logger
        if not self.model_settings.nominal_labels:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'labels type: {labels.dtype}')
            labels = self.torch_config.to_type(labels)
        return labels

    def _debug_output(self, msg: str, labels: Tensor, output: Tensor):
        logger = self.logger
        if isinstance(self.debug, int) and self.debug > 1 and \
           logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'{msg}:')
            logger.debug(f'labels: {labels.shape} ({labels.dtype})')
            if isinstance(self.debug, int) and self.debug > 1:
                logger.debug(f'label values:\n{labels}')
            if output is None:
                logger.debug('output: <none>')
            else:
                logger.debug(f'output: {output.shape} ({output.dtype})')
                if isinstance(self.debug, int) and self.debug > 1:
                    logger.debug(f'\n{output}')

    def _execute(self, model: BaseNetworkModule, optimizer, criterion,
                 batch: Batch, labels, split_type: str):
        logger = self.logger

        # forward pass, get our log probs
        output = model(batch)
        if output is None:
            raise ValueError('null model output')

        labels = self._encode_labels(labels)
        self._debug_output('input', labels, output)

        # calculate the loss with the logps and the labels
        loss = criterion(output, labels)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'split: {split_type}, loss: {loss}')

        if split_type == ModelResult.TRAIN_DS_NAME:
            # invoke back propogation on the network
            loss.backward()
            # take an update step and update the new weights
            optimizer.step()

        self._debug_output('output', labels, output)

        if not self.model_settings.nominal_labels:
            labels = self._decode_outcomes(labels)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'label nom decoded: {labels.shape}')

        output = self._decode_outcomes(output)

        return loss, labels, output

    def iterate(self, model: BaseNetworkModule, optimizer, criterion,
                batch: Batch, epoch_result: EpochResult, split_type: str):
        """Train, validate or test on a batch.  This uses the back propogation
        algorithm on training and does a simple feed forward on validation and
        testing.

        """
        logger = self.logger
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'train/validate on {split_type}: ' +
                         f'batch={batch} ({id(batch)})')
            logger.debug(f'model on device: {model.device}')
        batch = batch.to()
        labels = None
        output = None
        try:
            if self.debug:
                if isinstance(self.net_settings, MetadataNetworkSettings):
                    meta = self.net_settings.batch_metadata_factory()
                    meta.write()
                batch.write()

            labels = batch.get_labels()
            label_shapes = labels.shape
            if split_type == ModelResult.TRAIN_DS_NAME:
                optimizer.zero_grad()

            loss, labels, output = self._execute(
                model, optimizer, criterion, batch, labels, split_type)
            self._debug_output('decode', labels, output)

            if self.debug:
                raise EarlyBailException()

            epoch_result.update(batch, loss, labels, output, label_shapes)

            return loss

        finally:
            biter = self.model_settings.batch_iteration
            cb = self.model_settings.cache_batches
            if (biter == 'cpu' and not cb) or biter == 'buffered':
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'deallocating batch: {batch}')
                batch.deallocate()
            if labels is not None:
                del labels
            if output is not None:
                del output


@dataclass
class ScoredBatchIterator(BatchIterator):
    """Expects outputs as a list of lists of labels of indexes.  Examples of
    use cases include CRFs (e.g. BiLSTM/CRFs).

    """
    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        self.cnt = 0

    def _execute(self, model: ScoredNetworkModule, optimizer, criterion,
                 batch: Batch, labels, split_type: str):
        logger = self.logger

        # forward pass, get our log probs
        if split_type == ModelResult.TRAIN_DS_NAME:
            output = None
            loss = model(batch)
        else:
            output, score = model.score(batch)
            if split_type == ModelResult.TEST_DS_NAME:
                # we don't need the loss for testing
                loss = self.torch_config.singleton([0], dtype=torch.float32)
            else:
                loss = model.get_loss(batch)
            if output is None:
                raise ValueError('null model output')

        labels = self._encode_labels(labels)
        self._debug_output('input', labels, output)

        if split_type == ModelResult.TRAIN_DS_NAME:
            # invoke back propogation on the network
            loss.backward()
            # take an update step and update the new weights
            optimizer.step()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'split: {split_type}, loss: {loss}')

        self._debug_output('output', labels, output)

        if not self.model_settings.nominal_labels:
            labels = self._decode_outcomes(labels)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'label nom decoded: {labels.shape}')

        if output is not None:
            outs = []
            labs = []
            for rix, bout in enumerate(output):
                blen = len(bout)
                outs.append(torch.tensor(bout, dtype=labels.dtype))
                labs.append(labels[rix, :blen].cpu())
            output = torch.cat(outs, 0)
            labels = torch.cat(labs, 0)

        return loss, labels, output
