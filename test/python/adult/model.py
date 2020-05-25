import logging
from dataclasses import dataclass, field
from typing import Any, List
from torch import nn
from zensols.persist import persisted
from zensols.deeplearn.dataframe import DataframeBatchStash
from zensols.deeplearn.model import BaseNetworkModule
from zensols.deeplearn import (
    NetworkSettings,
    DeepLinearLayer,
)

logger = logging.getLogger(__name__)


@dataclass
class AdultNetworkSettings(NetworkSettings):
    """A utility container settings class for convulsion network models.

    """
    dataframe_stash: DataframeBatchStash
    middle_features: List[Any]
    last_layer_features: int
    out_features: int
    deep_linear_activation: str = field(default=None)
    use_batch_norm: bool = field(default=False)
    input_dropout: float = field(default=None)

    @property
    def in_features(self) -> int:
        return self.dataframe_stash.flattened_features_shape[0]

    @property
    @persisted('_deep_linear_activation_function')
    def deep_linear_activation_function(self):
        return self.get_activation_function(self.deep_linear_activation)

    def get_module_class_name(self) -> str:
        return __name__ + '.AdultNetwork'


class AdultNetwork(BaseNetworkModule):
    """A recurrent neural network model that is used to classify sentiment.

    """
    def __init__(self, net_settings: AdultNetworkSettings):
        super().__init__(net_settings, logger)
        ns = net_settings
        self.fc = DeepLinearLayer(
            ns.in_features, ns.last_layer_features, dropout=ns.dropout,
            middle_features=ns.middle_features,
            activation_function=ns.deep_linear_activation_function)
        self.last_fc = nn.Linear(ns.last_layer_features, ns.out_features)
        if ns.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(ns.last_layer_features)
        if ns.input_dropout is not None:
            self.input_droput = nn.Dropout(ns.input_dropout)

    def _forward(self, batch):
        logger.debug(f'batch: {batch}')

        x = batch.get_features()
        self._shape_debug('input', x)

        if self.net_settings.input_dropout is not None:
            x = self.input_droput(x)
            self._shape_debug('input_dropout', x)

        x = self.fc(x)
        self._shape_debug('deep linear', x)

        if self.net_settings.use_batch_norm:
            x = self.batch_norm(x)
            self._shape_debug('batch norm', x)

        x = self.last_fc(x)
        self._shape_debug('last linear', x)

        if self.net_settings.activation_function is not None:
            x = self.net_settings.activation_function(x)

        return x
