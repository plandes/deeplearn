import logging
from dataclasses import dataclass, field
from typing import Any, List
from torch import nn
import torch.nn.functional as F
from zensols.persist import persisted
from zensols.deeplearn import (
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
    deeep_linear_activation: str = field(default=None)

    @property
    @persisted('_deeep_linear_activation_function')
    def deeep_linear_activation_function(self):
        return self.get_activation_function(self.deeep_linear_activation)

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
            activation_function=ns.deeep_linear_activation_function)

    def _forward(self, batch):
        logger.debug(f'batch: {batch}')

        x = batch.get_flower_dimensions()
        self._shape_debug('input', x)

        x = self.fc(x)
        self._shape_debug('linear', x)

        if self.net_settings.activation_function is not None:
            x = self.net_settings.activation_function(x)
            x = F.relu(x)

        return x
