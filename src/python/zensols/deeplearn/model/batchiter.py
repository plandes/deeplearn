"""Contains a class to assist in the batch loop during training, validation and
testing.

"""
__author__ = 'Paul Landes'

from typing import Any
from dataclasses import dataclass, InitVar, field
import logging
from logging import Logger
import torch
from torch import Tensor
from zensols.deeplearn import ModelError, EarlyBailError, DatasetSplitType
from zensols.deeplearn.result import EpochResult
from zensols.deeplearn.batch import Batch, MetadataNetworkSettings
from . import (
    BaseNetworkModule,
    ScoredNetworkModule, ScoredNetworkContext, ScoredNetworkOutput
)


@dataclass
class BatchIterator(object):
    """This class assists in the batch loop during training, validation and
    testing.  Any special handling of a model related to its loss function can
    be overridden in this class.

    .. document private functions
    .. automethod:: _decode_outcomes

    """

    executor: InitVar[Any] = field()
    """The owning executor."""

    logger: Logger = field()
    """The status logger from the executor."""

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
                 batch: Batch, labels: Tensor, split_type: DatasetSplitType):
        logger = self.logger

        # forward pass, get our log probs
        output = model(batch)
        if output is None:
            raise ModelError('Null model output')

        labels = self._encode_labels(labels)
        self._debug_output('input', labels, output)

        # calculate the loss with the logps and the labels
        loss = criterion(output, labels)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'split: {split_type}, loss: {loss}')

        if split_type == DatasetSplitType.train:
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
                batch: Batch, epoch_result: EpochResult,
                split_type: DatasetSplitType):
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
            if split_type == DatasetSplitType.train:
                optimizer.zero_grad()

            loss, labels, output = self._execute(
                model, optimizer, criterion, batch, labels, split_type)
            self._debug_output('decode', labels, output)

            if self.debug:
                raise EarlyBailError()

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
                 batch: Batch, labels: Tensor, split_type: DatasetSplitType):
        logger = self.logger
        cctx = ScoredNetworkContext(split_type, criterion)
        sout: ScoredNetworkOutput = model(batch, cctx)
        preds: Tensor = sout.predictions
        loss: Tensor = sout.loss

        if logger.isEnabledFor(logging.DEBUG):
            pshape = '<none>' if preds is None else preds.shape
            logger.debug(f'output: {sout}, pred shape: {pshape}')

        labels = self._encode_labels(labels)
        self._debug_output('after forward', labels, preds)

        if split_type == DatasetSplitType.train:
            # invoke back propogation on the network
            loss.backward()
            # take an update step and update the new weights
            optimizer.step()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'split: {split_type}, loss: {loss}')

        # if not self.model_settings.nominal_labels:
        #     labels = self._decode_outcomes(labels)
        #     if logger.isEnabledFor(logging.DEBUG):
        #         logger.debug(f'label nom decoded: {labels.shape}')

        # self._debug_output('after decode', labels, preds)

        # if preds is not None:
        #     outs = []
        #     labs = []
        #     for rix, bout in enumerate(preds):
        #         blen = len(bout)
        #         outs.append(torch.tensor(bout, dtype=labels.dtype))
        #         labs.append(labels[rix, :blen].cpu())
        #         if logger.isEnabledFor(logging.DEBUG):
        #             logger.debug(f'row: {rix}, len: {blen}, out/lab')
        #     preds = torch.stack(outs)
        #     labels = torch.stack(labs)
        #     print(f'out/lab: {preds.shape}/{labels.shape}')

        # labels = labels.flatten()
        # preds = preds.flatten()

        return loss, labels, preds
