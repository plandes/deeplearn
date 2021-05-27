"""Scored modules for sequence models.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field
from abc import abstractmethod
from torch import Tensor
from torch import nn
from zensols.deeplearn import DatasetSplitType
from zensols.deeplearn.batch import Batch
from . import BaseNetworkModule


@dataclass
class ScoredNetworkContext(object):
    split_type: DatasetSplitType
    criterion: nn.Module = field()


@dataclass
class ScoredNetworkOutput(object):
    predictions: Tensor = field(repr=False)
    loss: Tensor
    score: Tensor


class ScoredNetworkModule(BaseNetworkModule):
    """A module that has a forward training pass and a separate **scoring** phase.
    Examples include layers with an ending linear CRF layer, such as a BiLSTM
    CRF.  This module has a ``decode`` method that returns a 2D list of integer
    label indexes of a nominal class.

    :see: :class:`zensols.deeplearn.layer.RecurrentCRFNetwork`

    """
    @abstractmethod
    def _forward(self, batch: Batch, context: ScoredNetworkContext) -> \
            ScoredNetworkOutput:
        pass

    # @abstractmethod
    # def _score(self, batch: Batch, context: ScoredNetworkContext) -> Tuple[Tensor, Tensor]:
    #     pass

    # def score(self, batch: Batch, context: ScoredNetworkContext) -> Tuple[Tensor, Tensor]:
    #     return self._score(batch, context)

    # def get_loss(self, batch: Batch, criterion) -> Tensor:
    #     if self.training:
    #         raise ModelError('Resolving a loss while training results ' +
    #                          'in the model training on the valdiation set--' +
    #                          'use model.eval() first')
    #     return self.forward(batch, criterion)
