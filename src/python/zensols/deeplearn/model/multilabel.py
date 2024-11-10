"""A class to reduce multilabel output.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field
import torch
from torch import Tensor
from torch import nn
from ..domain import ModelSettings


@dataclass
class MultiLabelOutcomeReducer(object):
    """Reduces mult-label classification predictions by clamping to ``[0, 1]``
    and converting the tensor to longs.

    """
    model_settings: ModelSettings = field()
    """Configures the model."""

    apply_softmax: bool = field(default=False)
    """Whether the application of the softmax is applied to the predictions
    before boxed between [0, 1] and rounded.

    """
    def __post_init__(self):
        if self.apply_softmax:
            self._softmax = nn.Softmax(dim=1)
        else:
            self._softmax = None

    def __call__(self, outcomes: Tensor) -> Tensor:
        if self._softmax is not None:
            outcomes = self._softmax(outcomes)
        return torch.clamp(outcomes, min=0, max=1).type(torch.LongTensor)
