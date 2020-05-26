import logging
from zensols.config import ExtendedInterpolationEnvConfig as AppConfig
from zensols.config import ImportConfigFactory
from zensols.deeplearn import TorchConfig
from zensols.deeplearn.model import ModelManager


def factory():
    config = AppConfig('test-resources/iris/iris.conf', env={'app_root': '.'})
    fac = ImportConfigFactory(config, shared=True, reload=False)
    return fac


def train_model():
    """Train, test the model, and save the results to the file system.

    """
    fac = factory()
    executor = fac('executor')
    executor.progress_bar = True
    executor.write()
    print('using device', executor.torch_config.device)
    executor.train()
    print('testing trained model')
    #executor.load()
    res = executor.test()
    res.write(verbose=True)
    return res


def test_model():
    fac = factory()
    path = fac.config.populate(section='model_settings').path
    print('testing from path', path)
    mm = ModelManager(path, fac)
    executor = mm.load_executor()
    res = executor.test()
    res.write(verbose=False)


def load_results():
    """Load the last set of results from the file system and print them out.

    """
    logging.getLogger('zensols.deeplearn.result').setLevel(logging.INFO)
    print('load previous results')
    fac = factory()
    executor = fac('executor')
    res = executor.result_manager.load()
    res.write(verbose=False)


def main():
    print()
    TorchConfig.set_random_seed()
    logging.basicConfig(level=logging.WARN)
    logging.getLogger('zensols.deeplearn.model').setLevel(logging.WARN)
    #logging.getLogger('zensols.deeplearn.model.executor').setLevel(logging.DEBUG)
    #run = [1, 2, 3]
    run = [1]
    res = None
    for r in run:
        res = {1: train_model,
               2: test_model,
               3: load_results}[r]()
    return res


res = main()
