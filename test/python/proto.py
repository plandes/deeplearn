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
    stash = fac('batch_dataset_stash')
    print(stash.vectorizer_manager_set.feature_types)
    return
    if 1:
        stash.clear()
    for v in it.islice(stash.values(), 2):
        #print(stash.reconstitute_batch(v).data_points)
        #print(v.features['iseries'])
        print(v.get_labels())
        print(v.get_flower_dimensions())
        #print(v.attributes['flower_dims'])
        # for feature_type, arr in v.features.items():
        #     print(feature_type, arr.shape)


def main():
    print()
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger('iris_dataset').setLevel(logging.DEBUG)
    stash()


main()
