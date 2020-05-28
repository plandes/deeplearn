import logging
from zensols.config import ExtendedInterpolationEnvConfig as AppConfig
from zensols.config import ImportConfigFactory
from zensols.deeplearn import TorchConfig
#from zensols.deeplearn.model import ModelManager


def factory():
    config = AppConfig('test-resources/mnist/mnist.conf',
                       env={'app_root': '.'})
    fac = ImportConfigFactory(config, reload=False)
    return fac


def dataset():
    fac = factory()
    dataset = fac('mnist_batch_stash')
    dataset.delegate_attr = True
    if 1:
        import itertools as it
        for p in it.islice(dataset.values(), 10):
            print(p, p.get_labels().shape, p.get_data().shape)
        print('-' * 10)
        dataset = dataset.splits['test']
        for p in it.islice(dataset.values(), 10):
            print(p, p.get_labels().shape, p.get_data().shape)
    keys = tuple(dataset.keys())
    keys = sorted(keys, key=lambda k: int(k))
    print(keys[:10])


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
    #logging.getLogger('zensols.deeplearn.model').setLevel(logging.WARN)
    #logging.getLogger('zensols.deeplearn.model.executor').setLevel(logging.DEBUG)
    run = [1]
    res = None
    for r in run:
        res = {1: dataset,
               2: train_model}[r]()
    return res


main()
