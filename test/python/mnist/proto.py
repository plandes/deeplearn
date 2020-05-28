from typing import Tuple
import logging
import torch
import itertools as it
from zensols.config import ExtendedInterpolationEnvConfig as AppConfig
from zensols.config import ImportConfigFactory
from zensols.deeplearn import TorchConfig
#from zensols.deeplearn.model import ModelManager

logger = logging.getLogger(__name__)


def factory():
    config = AppConfig('test-resources/mnist/mnist.conf',
                       env={'app_root': '.'})
    fac = ImportConfigFactory(config, reload=False)
    return fac


def dataset():
    fac = factory()
    dataset = fac('mnist_batch_stash')
    import itertools as it
    for p in it.islice(dataset.values(), 10):
        print(p, p.get_labels().shape, p.get_data().shape)
    print('-' * 10)
    dataset = dataset.splits['test']
    for p in it.islice(dataset.values(), 10):
        print(p, p.get_labels().shape, p.get_data().shape)


def assert_datasets():
    tc = TorchConfig(False)
    fac = factory()
    stash = fac('dataloader_stash')
    dataset = fac('mnist_batch_stash')
    dataset.delegate_attr = True
    ds_name = 'train val test'.split()
    name: str
    ds: Tuple[Tuple[torch.Tensor, torch.Tensor]]
    for name, ds in zip(ds_name, stash.get_data_by_split()):
        ds_start = 0
        ds_stash = dataset.splits[name]
        ds_data = torch.cat(tuple(map(lambda x: x[0], ds)))
        ds_labels = tc.from_iterable(map(lambda x: x[1], ds))
        logger.info(f'name: stash size: {len(ds_stash)}, ' +
                    f'data set size: {len(ds)}, ' +
                    f'stash X batch_size: {len(ds_stash) * 20}, ' +
                    f'data/label shapes: {ds_data.shape}/{ds_labels.shape}')
        assert len(ds) == (len(ds_stash) * dataset.delegate.batch_size)
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


def train_model():
    """Train, test the model, and save the results to the file system.

    """
    #logging.getLogger('mnist.model').setLevel(logging.DEBUG)
    logging.getLogger('zensols.deeplearn.model').setLevel(logging.INFO)
    fac = factory()
    executor = fac('executor')#, progress_bar=True, progress_bar_cols=120)
    executor.write()
    executor.train()
    if 0:
        print('testing trained model')
        executor.load()
        res = executor.test()
        res.write(verbose=False)
        #print(executor.get_predictions())
        return res


def main():
    print()
    TorchConfig.set_random_seed()
    logging.basicConfig(level=logging.WARN)
    logger.setLevel(logging.INFO)
    #logging.getLogger('zensols.deeplearn.model').setLevel(logging.WARN)
    #logging.getLogger('zensols.deeplearn.model.executor').setLevel(logging.DEBUG)
    run = [3]
    res = None
    for r in run:
        res = {1: dataset,
               2: assert_datasets,
               3: train_model}[r]()
    return res


main()
