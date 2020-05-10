import logging
from dataclasses import dataclass, field
from typing import Any, List
from torch import nn
from zensols.deeplearn import (
    EarlyBailException,
    NetworkSettings,
    DeepLinearLayer,
    BaseNetworkModule,
)

logger = logging.getLogger(__name__)


@dataclass
class IrisNetworkSettings(NetworkSettings):
    """A utility container settings class for convulsion network models.

    """
    in_features: int
    out_features: int
    middle_features: List[Any] = field(default=None)

    def get_module_class_name(self) -> str:
        return __name__ + '.IrisNetwork'


class IrisNetwork(BaseNetworkModule):
    """A recurrent neural network model that is used to classify sentiment.

    """
    def __init__(self, net_settings: IrisNetworkSettings):
        super().__init__(net_settings)
        ns = net_settings
        self.fc = DeepLinearLayer(
            ns.in_features, ns.out_features, dropout=ns.dropout,
            middle_features=ns.middle_features, 
            activation_function=ns.activation_function)
        self.dropout = None if ns.dropout is None else nn.Dropout(ns.dropout)

    def _forward(self, batch):
        logger.debug(f'batch: {batch}')

        x = batch.get_flower_dimensions()
        self._shape_debug('input', x)

        x = self.fc(x)
        self._shape_debug('linear', x)

        return x
