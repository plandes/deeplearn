import itertools as it
import logging
from zensols.config import ExtendedInterpolationConfig as AppConfig
from zensols.config import ImportConfigFactory


def factory():
    config = AppConfig(f'test-resources/vectorize.conf')
    fac = ImportConfigFactory(config, shared=False, reload=False)
    return fac


def stash():
    logging.getLogger('zensols.deeplearn').setLevel(logging.INFO)
    fac = factory()
    if 0:
        stash = fac('batch_dataset_stash')
    else:
        stash = fac('batch_split_dataset_stash')

    print('HD', stash._delegate_has_data())

    print('S', stash.keys_by_split)
    print('K', tuple(stash.keys()))

    for v in stash.values():
        print('F', v)

    if 1:
        stash.clear()

    for v in stash.values():
        print('F', v, v.get_labels())

    if 1:
        print('S', stash.keys_by_split)
        print('K', tuple(stash.keys()))

    for v in stash.values():
        print('F', v)


def main():
    print()
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger('iris_dataset').setLevel(logging.DEBUG)
    stash()


main()
