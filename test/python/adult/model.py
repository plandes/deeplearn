import logging
from dataclasses import dataclass
from torch import nn
from zensols.deeplearn.model import BaseNetworkModule
from zensols.deeplearn.layer import DeepLinear, DeepLinearNetworkSettings
from zensols.deeplearn.dataframe import DataframeBatchStash

logger = logging.getLogger(__name__)


@dataclass
class AdultNetworkSettings(DeepLinearNetworkSettings):
    """A utility container settings class for convulsion network models.

    """
    def __init__(self, name: str, dataframe_stash: DataframeBatchStash,
                 label_features: int, *args,
                 use_batch_norm: bool = False, **kwargs):
        in_feats = dataframe_stash.flattened_features_shape[0]
        super().__init__(name, *args, in_features=in_feats, **kwargs)
        self.dataframe_stash = dataframe_stash
        self.label_features = label_features
        self.use_batch_norm = use_batch_norm

    def get_module_class_name(self) -> str:
        return __name__ + '.AdultNetwork'


class AdultNetwork(BaseNetworkModule):
    """A recurrent neural network model that is used to classify sentiment.

    """
    def __init__(self, net_settings: AdultNetworkSettings):
        super().__init__(net_settings, logger)
        ns = net_settings
        self.fc = DeepLinear(ns)
        if ns.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(ns.out_features)
        self.last_fc = nn.Linear(ns.out_features, ns.label_features)

    def _forward(self, batch):
        logger.debug(f'batch: {batch}')

        x = batch.get_features()
        self._shape_debug('input', x)

        x = self.fc(x)
        self._shape_debug('deep linear', x)

        if self.net_settings.use_batch_norm:
            x = self.batch_norm(x)
            self._shape_debug('batch norm', x)

        x = self.last_fc(x)
        self._shape_debug('last linear', x)

        return x
