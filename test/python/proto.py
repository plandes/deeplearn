import itertools as it
import logging
from zensols.config import ExtendedInterpolationConfig as AppConfig
from zensols.config import ImportConfigFactory


def factory():
    config = AppConfig(f'test-resources/vectorize.conf')
    fac = ImportConfigFactory(config, shared=False, reload=False)
    return fac


def stash():
    fac = factory()
    #stash = fac('feature_subset_batch_dataset_stash')
    stash = fac('batch_dataset_stash')

    logging.getLogger('zensols.deeplearn.batch').setLevel(logging.WARNING)

    if 1:
        stash.clear()

    for v in it.islice(stash.values(), 2):
        print('F', v.attributes)
        print('F', v.feature_types)


def main():
    print()
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger('iris_dataset').setLevel(logging.DEBUG)
    stash()


main()
