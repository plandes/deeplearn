import logging
from zensols.config import ExtendedInterpolationEnvConfig as AppConfig
from zensols.config import ImportConfigFactory


def factory():
    config = AppConfig(f'test-resources/manager.conf',
                       env={'app_root': '.'})
    fac = ImportConfigFactory(config, shared=True, reload=False)
    return fac


def run_model():
    fac = factory()
    manager = fac('manager')
    manager.progress_bar = True
    manager.write()
    print(manager.create_model())
    print('using device', manager.torch_config.device)
    manager.train()
    res = manager.test()
    res.write()


def main():
    print()
    import torch
    # set the random seed so things are predictable
    torch.manual_seed(7)
    logging.basicConfig(level=logging.WARNING)
    run_model()


main()
