from typing import Tuple
import logging
import torch
from zensols.deeplearn import TorchConfig
from util import TargetTestCase

logger = logging.getLogger(__name__)
if 0:
    logging.basicConfig(level=logging.WARN)
    logger.setLevel(logging.DEBUG)


class TestMnistData(TargetTestCase):
    CONF_FILE = 'test-resources/mnist/mnist.conf'

    def test_datasets(self):
        tc = TorchConfig(False)
        fac = self.fac
        stash = fac('dataloader_stash')
        dataset = fac('mnist_batch_stash')
        dataset.delegate_attr = True
        ds_name = 'train val test'.split()
        batch_size = dataset.delegate.batch_size
        name: str
        ds: Tuple[Tuple[torch.Tensor, torch.Tensor]]
        for name, ds in zip(ds_name, stash.get_data_by_split()):
            ds_start = 0
            ds_stash = dataset.splits[name]
            ds_data = torch.cat(tuple(map(lambda x: x[0], ds)))
            ds_labels = torch.cat(tuple(map(lambda x: x[1], ds)))
            dpts = sum(map(lambda b: len(b.data_point_ids), ds_stash.values()))
            logger.info(f'name: stash size: {len(ds_stash)}, ' +
                        f'data set size: {len(ds)}, ' +
                        f'stash X batch_size: {len(ds_stash) * batch_size}, ' +
                        f'data/label shapes: {ds_data.shape}/{ds_labels.shape}, ' +
                        f'data points: {dpts}')
            assert len(ds) == len(ds_stash)
            assert dpts == ds_labels.shape[0]
            assert ds_labels.shape[0] == ds_data.shape[0]
            for id, batch in ds_stash:
                ds_end = ds_start + len(batch)
                dsb_labels = ds_labels[ds_start:ds_end]
                dsb_data = ds_data[ds_start:ds_end]
                ds_start = ds_end
                blabels = batch.get_labels()
                bdata = batch.get_data()
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'data point ids: {batch.data_point_ids}')
                    logger.debug(f'ds/batch labels: {dsb_labels}/{blabels}')
                assert (tc.equal(dsb_labels, blabels))
                assert (tc.equal(dsb_data, bdata))
