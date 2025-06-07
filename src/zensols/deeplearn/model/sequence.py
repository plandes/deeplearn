"""Sequence modules for sequence models.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import List, Union, Tuple
from dataclasses import dataclass, field
from abc import abstractmethod
import logging
import torch
from torch import Tensor
from torch import nn
from torch.optim import Optimizer
from zensols.persist import Deallocatable
from zensols.deeplearn import DatasetSplitType, ModelError
from zensols.deeplearn.batch import Batch
from . import BaseNetworkModule, BatchIterator

logger = logging.getLogger(__name__)


@dataclass
class SequenceNetworkContext(object):
    """The forward context for the :class:`.SequenceNetworkModule`.  This is
    used in :meth:`.SequenceNetworkModule._forward` to provide the module
    additional information needed to score the model and produce the loss.

    """
    split_type: DatasetSplitType = field()
    """The split type, which informs the module when decoding to produce outputs
    or using the forward pass to prod.

    :see: :meth:`.SequenceNetworkModule._forward`

    """
    criterion: nn.Module = field()
    """The criterion used to create the loss.  This is provided for modules that
    produce the loss in the forward phase with the
    `:meth:`torch.nn.module.forward`` method.

    """


class SequenceNetworkOutput(Deallocatable):
    """The output from :clas:`.SequenceNetworkModule` modules.

    """
    def __init__(self, predictions: Union[List[List[int]], Tensor],
                 loss: Tensor = None,
                 score: Tensor = None,
                 labels: Union[List[List[int]], Tensor] = None,
                 outputs: Tensor = None):
        """Initialize the output of a sequence NN.

        :param predictions: list of list predictions to convert in to a 1-D
                            tensor if given and not already a tensor; if a
                            tensor, the shape must also be 1-D

        :param loss: the loss tensor

        :param score: the score given by the CRF's Verterbi algorithm

        :param labels: list of list gold labels to convert in to a 1-D tensor
                       if given and not already a tensor

        :param outputs: the logits from the model

        """
        if predictions is not None and not isinstance(predictions, Tensor):
            # shape: 1D
            self.predictions = self._to_tensor(predictions)
        else:
            self.predictions = predictions
        if labels is not None and not isinstance(labels, Tensor):
            self.labels = self._to_tensor(labels)
        else:
            self.labels = labels
        self.loss = loss
        self.score = score
        self.outputs = outputs

    def _to_tensor(self, lists: List[List[int]]) -> Tensor:
        """Flatten a list of lists.

        :return: a 1-D tensor by flattening of the ``lists`` data

        """
        outs = []
        for lst in lists:
            outs.append(torch.tensor(lst, dtype=torch.int64))
        arr: Tensor = torch.cat(outs, dim=0)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'output shape: {arr.shape}')
        return arr

    def righsize_labels(self, preds: List[List[int]]):
        """Convert the :obj:`labels` tensor as a 1-D tensor.  This removes the
        padded values by iterating over ``preds`` using each sub list's for
        copying the gold label tensor to the new tensor.

        """
        labs = []
        labels = self.labels
        for rix, bout in enumerate(preds):
            blen: int = len(bout)
            labs.append(labels[rix, :blen].cpu())
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'row: {rix}, len: {blen}, out/lab')
        self.labels = torch.cat(labs, 0)

    def deallocate(self):
        for i in 'predictions loss score':
            if hasattr(self, i):
                delattr(self, i)

    def __str__(self) -> str:
        lbs, preds = self.labels, self.predictions
        lbs: str = None if lbs is None else str(len(lbs))
        preds: str = None if preds is None else str(len(preds))
        return (f'labels: {lbs}, predictions: {preds}, ' +
                f'loss: {self.loss}, score: {self.score}')


class SequenceNetworkModule(BaseNetworkModule):
    """A module that has a forward training pass and a separate *scoring* phase.
    Examples include layers with an ending linear CRF layer, such as a BiLSTM
    CRF.  This module has a ``decode`` method that returns a 2D list of integer
    label indexes of a nominal class.

    The context provides additional information needed to train, test and use
    the module.

    :see: :class:`zensols.deeplearn.layer.RecurrentCRFNetwork`

    .. document private functions
    .. automethod:: _forward

    """
    @abstractmethod
    def _forward(self, batch: Batch, context: SequenceNetworkContext) -> \
            SequenceNetworkOutput:
        """The forward pass, which either trains the model and creates the loss
        and/or decodes the output for testing and evaluation.

        :param batch: the batch to train, validate or test on

        :param context: contains the additional information needed for scoring
                        and decoding the sequence

        """
        pass


@dataclass
class SequenceBatchIterator(BatchIterator):
    """Expects outputs as a list of lists of labels of indexes.  Examples of
    use cases include CRFs (e.g. BiLSTM/CRFs).

    """
    def _execute(self, model: BaseNetworkModule, optimizer: Optimizer,
                 criterion, batch: Batch, split_type: DatasetSplitType) -> \
            Tuple[Tensor]:
        logger = self.logger
        cctx = SequenceNetworkContext(split_type, criterion)
        seq_out: SequenceNetworkOutput = model(batch, cctx)
        outcomes: Tensor = seq_out.predictions
        loss: Tensor = seq_out.loss

        if seq_out.labels is not None and seq_out.predictions is not None and \
           seq_out.labels.shape != seq_out.predictions.shape:
            raise ModelError(
                f'Label / prediction count mismatch: {seq_out}, batch: {batch}')

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'{batch.id}: output: {seq_out}')

        if seq_out.labels is not None:
            labels = seq_out.labels
        else:
            labels: Tensor = batch.get_labels()
            labels = self._encode_labels(labels)

        if logger.isEnabledFor(logging.DEBUG):
            if labels is not None:
                logger.debug(f'label shape: {labels.shape}')

        self._debug_output('after forward', labels, outcomes)

        # iterate over the error surface
        self._step(loss, split_type, optimizer, model)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'split: {split_type}, loss: {loss}')

        # transform the labels in the same manner as the predictions so tensor
        # shapes match
        if not self.model_settings.nominal_labels:
            labels = self._decode_outcomes(labels)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'label nom decoded: {labels.shape}')

        if outcomes is None and split_type != DatasetSplitType.train:
            raise ModelError('Expecting predictions for all splits except ' +
                             f'{DatasetSplitType.train} on {split_type}')

        if logger.isEnabledFor(logging.DEBUG):
            if outcomes is not None:
                logger.debug(f'outcomes: {outcomes.shape}')
            if labels is not None:
                logger.debug(f'labels: {labels.shape}')

        loss, labels, outcomes, outputs = self.torch_config.to_cpu_deallocate(
            loss, labels, outcomes, seq_out.outputs)
        return loss, labels, outcomes, outputs
