import logging
from zensols.config import ExtendedInterpolationEnvConfig as AppConfig
from zensols.config import ImportConfigFactory
from zensols.deeplearn import ModelManager


def factory():
    config = AppConfig(f'test-resources/executor.conf',
                       env={'app_root': '.'})
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
    res = executor.test()
    res.write()


def test_model():
    #logging.getLogger('zensols.config').setLevel(logging.DEBUG)
    fac = factory()
    path = fac.config.populate(section='model_settings').path
    print('path', path)
    mm = ModelManager(path, fac)
    executor = mm.load_executor()
    res = executor.test()
    res.write()


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
    import torch
    # set the random seed so things are predictable
    torch.manual_seed(7)
    logging.basicConfig(level=logging.WARN)
    run = [1, 2]
    for r in run:
        {1: train_model,
         2: test_model,
         3: load_results}[r]()


main()
