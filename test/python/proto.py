import logging
from zensols.config import ExtendedInterpolationConfig as AppConfig
from zensols.config import ImportConfigFactory


def factory():
    config = AppConfig(f'test-resources/manager.conf')
    fac = ImportConfigFactory(config, shared=True, reload=False)
    return fac


def tmp():
    #logging.getLogger('iris_model').setLevel(logging.DEBUG)
    logging.getLogger('zensols.deeplearn.batch').setLevel(logging.WARN)
    fac = factory()
    manager = fac('manager')
    manager.write()
    return
    manager.train()
    res = manager.test()
    res.write()


def main():
    print()
    import torch
    # set the random seed so things are predictable
    torch.manual_seed(7)
    logging.basicConfig(level=logging.WARNING)
    tmp()


main()
