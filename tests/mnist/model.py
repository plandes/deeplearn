from typing import Tuple
from dataclasses import dataclass
import logging
import torch
from torch import nn
import torch.nn.functional as F
from zensols.deeplearn.model import (
    BaseNetworkModule,
    NetworkSettings,
)
from zensols.deeplearn.batch import (
    DataPoint,
    Batch,
    BatchFeatureMapping,
    ManagerFeatureMapping,
    FieldFeatureMapping,
)

logger = logging.getLogger(__name__)


@dataclass
class MnistDataPoint(DataPoint):
    data_label: Tuple[torch.Tensor, torch.Tensor]

    @property
    def data(self):
        return self.data_label[0]

    @property
    def label(self):
        return self.data_label[1]


@dataclass
class MnistBatch(Batch):
    MAPPINGS = BatchFeatureMapping(
        'label',
        [ManagerFeatureMapping(
            'mnist_vectorizer_manager',
            (FieldFeatureMapping('label', 'identity', is_agg=True),
             FieldFeatureMapping('data', 'identity', is_agg=True)))])

    def _get_batch_feature_mappings(self) -> BatchFeatureMapping:
        return self.MAPPINGS

    def get_data(self) -> torch.Tensor:
        return self.attributes['data']


@dataclass
class MnistNetworkSettings(NetworkSettings):
    def get_module_class_name(self) -> str:
        return __name__ + '.MnistNetwork'


class MnistNetwork(BaseNetworkModule):
    def __init__(self, net_settings: MnistNetworkSettings):
        super(MnistNetwork, self).__init__(net_settings, logger)
        # number of hidden nodes in each layer (512)
        hidden_1 = 512
        hidden_2 = 512
        # linear layer (784 -> hidden_1)
        self.fc1 = nn.Linear(28 * 28, hidden_1)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(hidden_2, 10)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.2)

    def _forward(self, batch):
        logger.debug(f'batch: {batch}')

        x = batch.get_data()
        self._shape_debug('input', x)

        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        # add dropout layer
        x = self.dropout(x)
        # add output layer
        x = self.fc3(x)

        return x
