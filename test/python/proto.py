import logging
from zensols.config import ExtendedInterpolationConfig as AppConfig
from zensols.config import ImportConfigFactory


def factory():
    config = AppConfig(f'test-resources/vectorize.conf')
    fac = ImportConfigFactory(config, shared=False, reload=False)
    return fac


def stash():
    #logging.getLogger('zensols.deeplearn').setLevel(logging.INFO)
    fac = factory()
    if 0:
        stash = fac('batch_dataset_stash')
    else:
        stash = fac('batch_split_dataset_stash')

    print('S', stash.keys_by_split)
    print('K', tuple(stash.keys()))

    kbs = stash.keys_by_split
    for sname in stash.split_names:
        sstash = stash.splits[sname]
        print(len(kbs[sname]), len(sstash))
        print(len(kbs[sname]), len(tuple(sstash.keys())))
        print(len(kbs[sname]), len(tuple(sstash.values())))

    # for i, (id, v) in enumerate(stash):
    #     print(i, id, v, v.get_labels().shape)


def main():
    print()
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger('iris_dataset').setLevel(logging.DEBUG)
    stash()


main()
