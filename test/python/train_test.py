import logging
from zensols.config import ExtendedInterpolationEnvConfig as AppConfig
from zensols.config import ImportConfigFactory


def factory():
    config = AppConfig(f'test-resources/executor.conf',
                       env={'app_root': '.'})
    fac = ImportConfigFactory(config, shared=True, reload=False)
    return fac


def run_model():
    """Train, test the model, and save the results to the file system.

    """
    fac = factory()
    executor = fac('executor')
    executor.progress_bar = True
    executor.write()
    print(executor.create_model())
    print('using device', executor.torch_config.device)
    executor.train()
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
    res.write(verbose=True)


def main():
    print()
    import torch
    # set the random seed so things are predictable
    torch.manual_seed(7)
    logging.basicConfig(level=logging.WARNING)
    run_model()
    load_results()


main()
