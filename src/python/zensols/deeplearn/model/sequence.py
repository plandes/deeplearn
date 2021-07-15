from __future__ import annotations
"""Sequence modules for sequence models.

"""
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
    """The forward context for the :class:`.SequenceNetworkModule`.  This is used
    in :meth:`.SequenceNetworkModule._forward` to provide the module additional
    information needed to score the model and produce the loss.

    """

    split_type: DatasetSplitType = field()
    """The split type, which informs the module when decoding to produce outputs or
    using the forward pass to prod.

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
                 labels: Union[List[List[int]]] = None,
                 outputs: Tensor = None):
        if predictions is not None and not isinstance(predictions, Tensor):
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
        outs = []
        for lst in lists:
            outs.append(torch.tensor(lst, dtype=torch.int64))
        arr: Tensor = torch.cat(outs, dim=0)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'output shape: {arr.shape}')
        return arr

    def righsize_labels(self, preds: List[List[int]]):
        labs = []
        labels = self.labels
        for rix, bout in enumerate(preds):
            blen = len(bout)
            labs.append(labels[rix, :blen].cpu())
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'row: {rix}, len: {blen}, out/lab')
        self.labels = torch.cat(labs, 0)

    def deallocate(self):
        for i in 'predictions loss score':
            if hasattr(self, i):
                delattr(self, i)


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
        """The forward pass, which either trains the model and creates the loss and/or
        decodes the output for testing and evaluation.

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
    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        self.cnt = 0

    def _execute(self, model: BaseNetworkModule, optimizer: Optimizer,
                 criterion, batch: Batch, split_type: DatasetSplitType) -> \
            Tuple[Tensor]:
        logger = self.logger
        cctx = SequenceNetworkContext(split_type, criterion)
        seq_out: SequenceNetworkOutput = model(batch, cctx)
        outcomes: Tensor = seq_out.predictions
        loss: Tensor = seq_out.loss

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

        if split_type == DatasetSplitType.train:
            # invoke back propogation on the network
            loss.backward()
            # take an update step and update the new weights
            optimizer.step()

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
