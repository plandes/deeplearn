"""A class to reduce multilabel output.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field
import torch
from torch import Tensor
from ..domain import ModelSettings


@dataclass
class MultiLabelOutcomeReducer(object):
    """Reduces mult-label classification predictions by clamping to ``[0, 1]``,
    rounding, and converting the tensor to longs.

    """
    model_settings: ModelSettings = field()
    """Configures the model."""

    def __call__(self, outcomes: Tensor) -> Tensor:
        outcomes = torch.clamp(outcomes, min=0, max=1)
        outcomes.round_()
        return outcomes.type(torch.LongTensor)
