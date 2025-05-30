"""Contains a class to assist in the batch loop during training, validation and
testing.

"""
__author__ = 'Paul Landes'

from typing import Any, Tuple, Dict, Callable
from dataclasses import dataclass, InitVar, field
import logging
from logging import Logger
from torch import Tensor
from torch import nn
from torch.optim import Optimizer
from zensols.deeplearn import (
    ModelError, EarlyBailError, DatasetSplitType, TorchConfig,
    ModelSettings, NetworkSettings
)
from zensols.introspect import ClassImporter
from zensols.deeplearn.result import EpochResult
from zensols.deeplearn.batch import Batch, MetadataNetworkSettings
from . import BaseNetworkModule


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
        self.model_settings: ModelSettings = executor.model_settings
        self.net_settings: NetworkSettings = executor.net_settings
        self.torch_config: TorchConfig = executor.torch_config
        self.outcome_reducer: Callable = self._get_outcome_reducer()

    def _get_outcome_reducer(self) -> Callable:
        def argmax_reduce(outcomes: Tensor) -> Tensor:
            return outcomes.argmax(dim=-1)

        def softmax_reduce(outcomes: Tensor) -> Tensor:
            return outcomes.softmax(dim=-1)

        def identity_reduce(outcomes: Tensor) -> Tensor:
            return outcomes

        reduce_outcomes: str = self.model_settings.reduce_outcomes
        reducer: Callable = None
        # get the indexes of the max value across labels and outcomes (for the
        # descrete classification case)
        if reduce_outcomes == 'argmax':
            reducer = argmax_reduce
        # softmax over each outcome
        elif reduce_outcomes == 'softmax':
            reducer = softmax_reduce
        # leave when nothing, prediction/regression measure is used
        elif reduce_outcomes == 'none':
            reducer = identity_reduce
        else:
            # custom reducer, such as .multilabel.MultiLabelOutcomeReducer
            ci = ClassImporter(reduce_outcomes, False)
            reducer = ci.instance(model_settings=self.model_settings)
        return reducer

    def _decode_outcomes(self, outcomes: Tensor) -> Tensor:
        """Transform the model output (and optionally the labels) that will be
        added to the ``EpochResult``, which composes a ``ModelResult``.

        This implementation returns :py:meth:~`Tensor.argmax`, which are
        the indexes of the max value across columns.

        """
        logger: Logger = self.logger
        reduced: Tensor = self.outcome_reducer(outcomes)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'reduced outcomes: {outcomes.shape}->{reduced.shape}')
        return reduced

    def _encode_output(self, output: Tensor) -> Tensor:
        """Any massaging of the model output that might be necessary.  See
        :meth:`_encode_labels`.

        """
        return output

    def _encode_labels(self, labels: Tensor) -> Tensor:
        """Encode labels to be in the same form and on the same CUDA device as
        the batch data.  This base class implementation only copies to the GPU.

        :param labels: labels paired with the training and validation datasets

        :return: labels to be used in the loss function

        """
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
            shape = None if labels is None else labels.shape
            dtype = None if labels is None else labels.dtype
            logger.debug(f'labels: {shape} ({dtype})')
            if isinstance(self.debug, int) and self.debug > 1:
                logger.debug(f'label values:\n{labels}')
            if output is None:
                logger.debug('output: <none>')
            else:
                logger.debug(f'output: {output.shape} ({output.dtype})')
                if isinstance(self.debug, int) and self.debug > 1:
                    logger.debug(f'\n{output}')

    def iterate(self, model: BaseNetworkModule, optimizer: Optimizer,
                criterion, batch: Batch, epoch_result: EpochResult,
                split_type: DatasetSplitType) -> Tensor:
        """Train, validate or test on a batch.  This uses the back propogation
        algorithm on training and does a simple feed forward on validation and
        testing.

        One call of this method represents a single batch iteration

        :param model: the model to excercise

        :param optimizer: the optimization algorithm (i.e. adam) to iterate

        :param criterion: the loss function (i.e. cross entropy loss) used for
                          the backward propogation step

        :param batch: contains the data to test, predict, and optionally the
                      labels for training and validation

        :param epoch_result: to be populated with the results of this epoch's
                             run

        :param split_type: indicates if we're training, validating or testing

        :return: the singleton tensor containing the loss

        """
        logger = self.logger
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'train/validate on {split_type}: ' +
                         f'batch={batch} ({id(batch)})')
            logger.debug(f'model on device: {model.device}')
        # copy batch to GPU if configured to do so
        batch: Batch = batch.to()
        outcomes: Tensor = None
        output: Tensor = None
        try:
            if self.debug:
                # write a batch sample when debugging; maybe make this a hook
                if isinstance(self.net_settings, MetadataNetworkSettings):
                    meta = self.net_settings.batch_metadata
                    meta.write()
                batch.write()

            # when training, reset gradients for the next epoch
            if split_type == DatasetSplitType.train:
                optimizer.zero_grad()

            # execute an the epoch
            loss, labels, outcomes, output = self._execute(
                model, optimizer, criterion, batch, split_type)
            self._debug_output('decode', labels, outcomes)

            # if debugging the model, raise the exception to interrupt the
            # flow, which is caught in ModelExecutor._execute
            if self.debug:
                raise EarlyBailError()

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('outcomes shape: {outcomes.shape}')

            # add results for performance metrics, predictions output, etc
            epoch_result.update(batch, loss, labels, outcomes, output)

            return loss
        finally:
            # clean up and GPU memeory deallocation
            biter = self.model_settings.batch_iteration
            cb = self.model_settings.cache_batches
            if (biter == 'cpu' and not cb) or biter == 'buffered':
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'deallocating batch: {batch}')
                batch.deallocate()

    def _step(self, loss: Tensor, split_type: DatasetSplitType,
              optimizer, model: BaseNetworkModule):
        """Iterate over the error surface."""
        # when training, backpropogate and step
        if split_type == DatasetSplitType.train:
            clip_thresh: float = self.model_settings.clip_gradient_threshold
            clip_params: Dict[str, Any] = \
                self.model_settings.scale_gradient_params
            # invoke back propogation on the network
            loss.backward()
            # clip the gradient
            if clip_thresh is not None:
                nn.utils.clip_grad_value_(model.parameters(), clip_thresh)
            # scale the gradient
            if clip_params is not None:
                nn.utils.clip_grad_norm_(model.parameters(), **clip_params)
            # take an update step and update the new weights
            optimizer.step()

    def _execute(self, model: BaseNetworkModule, optimizer: Optimizer,
                 criterion, batch: Batch, split_type: DatasetSplitType) -> \
            Tuple[Tensor]:
        """Execute one epoch of training, testing, validation or prediction.

        :param model: the model to excercise

        :param optimizer: the optimization algorithm (i.e. adam) to iterate

        :param criterion: the loss function (i.e. cross entropy loss) used for
                          the backward propogation step

        :param batch: contains the data to test, predict, and optionally the
                      labels for training and validation

        :param split_type: indicates if we're training, validating or testing

        :return: a tuple of the loss, labels, outcomes, and the output
                 (i.e. logits); the outcomes are the decoded
                 (:meth:`_decode_outcomes`) output and represent some ready to
                 use data, like argmax'd classification nominal label integers

        """
        logger = self.logger
        labels: Tensor = batch.get_labels()
        # forward pass, get our output, which are usually the logits
        output: Tensor = model(batch)

        # sanity check
        if output is None:
            raise ModelError('Null model output')

        # check for sane state with labels, and munge if necessary
        if labels is None:
            # sanity check
            if split_type != DatasetSplitType.test:
                raise ModelError('Expecting no split type on prediction, ' +
                                 f'but got: {split_type}')
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('skipping loss calculation on prediction execute')
            loss = None
        else:
            # put labels in a form to be used by the loss function
            labels = self._encode_labels(labels)
            output = self._encode_output(output)
            self._debug_output('input', labels, output)

            # calculate the loss with the logps and the labels
            loss = criterion(output, labels)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'split: {split_type}, loss: {loss}')

        # iterate over the error surface
        self._step(loss, split_type, optimizer, model)
        self._debug_output('output', labels, output)

        # apply the same decoding on the labels as the output if necessary
        if labels is not None and not self.model_settings.nominal_labels:
            labels = self._decode_outcomes(labels)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'label nom decoded: {labels.shape}')

        outcomes = self._decode_outcomes(output)
        loss, labels, outcomes, output = self.torch_config.to_cpu_deallocate(
            loss, labels, outcomes, output)
        return loss, labels, outcomes, output
