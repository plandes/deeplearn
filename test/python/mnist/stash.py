from typing import Tuple, Iterable, Dict, Set
from dataclasses import dataclass
import logging
import collections
import itertools as it
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from zensols.persist import persisted, OneShotFactoryStash
from zensols.dataset import SplitKeyContainer

logger = logging.getLogger(__name__)


@dataclass
class DataLoaderStash(OneShotFactoryStash, SplitKeyContainer):
    path: Path

    @property
    def worker(self) -> Iterable[Tuple[str, str, torch.Tensor]]:
        """Return an iterable of the data from the data loaders across all data sets.

        :return: an iterable of tuples of the form:
                 ``(id, dataset split, the data tensor, the label tensor)``

        """
        ds_name = 'train val test'.split()
        self._key_splits = collections.defaultdict(lambda: set())
        id = 0
        for name, ds in zip(ds_name, self.get_data_by_split()):
            for data, labels in ds:
                id += 1
                key = str(id)
                self._key_splits[name].add(key)
                yield (key, (data, labels))

    def _get_keys_by_split(self) -> Dict[str, Set[str]]:
        self.prime()
        return dict(self._key_splits)

    @persisted('_data_by_spilt', cache_global=True)
    def get_data_by_split(self) -> Tuple[Tuple[Tuple[torch.Tensor, torch.Tensor]]]:
        # number of subprocesses to use for data loading
        num_workers = 0
        # how many samples per batch to load
        batch_size = 1
        # percentage of training set to use as validation
        valid_size = 0.2

        # convert data to torch.FloatTensor
        transform = transforms.ToTensor()

        # choose the training and test datasets
        root = str(self.path)
        train_data = datasets.MNIST(root=root, train=True,
                                    download=True, transform=transform)
        test_data = datasets.MNIST(root=root, train=False,
                                   download=True, transform=transform)

        # obtain training indices that will be used for validation
        num_train = len(train_data)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # prepare data loaders
        train_loader = DataLoader(
            train_data, batch_size=batch_size,
            sampler=train_sampler, num_workers=num_workers)
        valid_loader = DataLoader(
            train_data, batch_size=batch_size, 
            sampler=valid_sampler, num_workers=num_workers)
        test_loader = DataLoader(
            test_data, batch_size=batch_size, 
            num_workers=num_workers)

        train = tuple(it.islice(train_loader, 18720))#936))
        valid = tuple(valid_loader)
        test = tuple(test_loader)

        return train, valid, test
