import logging
from zensols.config import ExtendedInterpolationEnvConfig as AppConfig
from zensols.config import ImportConfigFactory
from zensols.deeplearn import TorchConfig
from zensols.deeplearn.model import ModelManager

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


def train_model():
    """Train, test the model, and save the results to the file system.

    """
    #logging.getLogger('mnist.model').setLevel(logging.DEBUG)
    #logging.getLogger('zensols.deeplearn.model').setLevel(logging.INFO)
    fac = factory()
    executor = fac('executor', progress_bar=True, progress_bar_cols=110)
    executor.write()
    executor.train()
    if 1:
        print('testing trained model')
        executor.load()
        res = executor.test()
        res.write()
        return res


def test_model():
    fac = factory()
    path = fac.config.populate(section='model_settings').path
    print('testing from path', path)
    mm = ModelManager(path, fac)
    executor = mm.load_executor()
    res = executor.test()
    executor.result_manager.dump(res)
    res.write()


def load_results():
    """Load the last set of results from the file system and print them out.

    """
    logging.getLogger('zensols.deeplearn.result').setLevel(logging.INFO)
    print('load previous results')
    fac = factory()
    executor = fac('executor')
    res = executor.result_manager.load()
    res.write()


def main():
    print()
    TorchConfig.set_random_seed()
    logging.basicConfig(level=logging.WARN)
    logger.setLevel(logging.INFO)
    run = [2, 3, 4]
    res = None
    for r in run:
        res = {1: dataset,
               2: train_model,
               3: test_model,
               4: load_results}[r]()
    return res


main()
