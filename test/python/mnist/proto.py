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
    dataset = fac('batch_dataset_stash')
    import itertools as it
    for p in it.islice(dataset.values(), 10):
        print(p.get_data().shape)


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
        }[r]()
    return res


main()
