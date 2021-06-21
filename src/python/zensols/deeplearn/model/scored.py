from __future__ import annotations
"""Scored modules for sequence models.

"""
__author__ = 'Paul Landes'

from typing import List, Union
from dataclasses import dataclass, field
from abc import abstractmethod
import logging
import torch
from torch import Tensor
from torch import nn
from zensols.persist import Deallocatable
from zensols.deeplearn import DatasetSplitType
from zensols.deeplearn.batch import Batch
from . import BaseNetworkModule

logger = logging.getLogger(__name__)


@dataclass
class ScoredNetworkContext(object):
    """The forward context for the :class:`.ScoredNetworkModule`.  This is used in
    :meth:`.ScoredNetworkModule._forward` to provide the module additional
    information needed to score the model and produce the loss.

    """

    split_type: DatasetSplitType = field()
    """The split type, which informs the module when decoding to produce outputs or
    using the forward pass to prod.

    :see: :meth:`.ScoredNetworkModule._forward`

    """
    criterion: nn.Module = field()
    """The criterion used to create the loss.  This is provided for modules that
    produce the loss in the forward phase with the
    `:meth:`torch.nn.module.forward`` method.

    """


class ScoredNetworkOutput(Deallocatable):
    """The output from :clas:`.ScoredNetworkModule` modules.

    """
    def __init__(self, predictions: Union[List[List[int]], Tensor],
                 loss: Tensor = None,
                 score: Tensor = None,
                 labels: List[List[int]] = None):
        self.predictions = predictions
        self.labels = labels
        if not isinstance(predictions, Tensor) and predictions is not None:
            self.predictions = self._to_tensor(self.predictions)
        if not isinstance(labels, Tensor) and labels is not None:
            self.labels = self._to_tensor(self.labels)
        self.loss = loss
        self.score = score

    def _to_tensor(self, lists: List[List[int]]) -> Tensor:
        outs = []
        for lst in lists:
            outs.append(torch.tensor(lst, dtype=torch.int64))
        arr: Tensor = torch.cat(outs, dim=0)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'output shape: {arr.shape}')
        return arr

    def deallocate(self):
        for i in 'predictions loss score':
            if hasattr(self, i):
                delattr(self, i)


class ScoredNetworkModule(BaseNetworkModule):
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
    def _forward(self, batch: Batch, context: ScoredNetworkContext) -> \
            ScoredNetworkOutput:
        """The forward pass, which either trains the model and creates the loss and/or
        decodes the output for testing and evaluation.

        :param batch: the batch to train, validate or test on

        :param context: contains the additional information needed for scoring
                        and decoding the sequence

        """
        pass
