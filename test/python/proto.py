import logging
from zensols.config import ExtendedInterpolationConfig as AppConfig
from zensols.config import ImportConfigFactory


def factory():
    config = AppConfig(f'test-resources/manager.conf')
    fac = ImportConfigFactory(config, shared=True, reload=False)
    return fac


def tmp():
    #logging.getLogger('zensols.deeplearn').setLevel(logging.INFO)
    fac = factory()
    manager = fac('manager')
    manager.train()


def main():
    print()
    logging.basicConfig(level=logging.WARNING)
    tmp()


main()
