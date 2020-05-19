import logging
from zensols.config import ExtendedInterpolationEnvConfig as AppConfig
from zensols.config import ImportConfigFactory
from zensols.deeplearn import TorchConfig

logger = logging.getLogger(__name__)


def factory(reload=True):
    config = AppConfig(f'test-resources/adult.conf',
                       env={'app_root': '.'})
    fac = ImportConfigFactory(config, shared=True, reload_root=reload)
    return fac


def dataset():
    logging.getLogger('adult.data').setLevel(logging.DEBUG)
    fac = factory()
    ds = fac('dataset_split_stash')
    #ds.clear()
    ds.write()
    #meta = ds.delegate.metadata
    train = ds.splits['train']
    d = next(iter(train.values()))
    for i, v in d.iteritems():
        print(i, v)
    #print(d['age fnlwgt education_num capital_gain'.split()])


def metadata():
    logging.getLogger('adult.data').setLevel(logging.DEBUG)
    fac = factory()
    mng = fac('adult_vectorizer_manager')
    #mng.dataset_metadata.write()
    print(mng.feature_types)
    mng.write()


def batch():
    import itertools as it
    logging.getLogger('adult.data').setLevel(logging.DEBUG)
    #logging.getLogger('zensols.deeplearn.batch').setLevel(logging.DEBUG)
    fac = factory(False)
    #fac('adult_dataset_stash').write()
    stash = fac('adult_batch_stash')
    #stash.delegate.feature_vectorizer_manager.write()
    stash.write()
    print(len(stash))
    print(len(stash.splits['train']))
    for k, v in it.islice(stash, 10):
        print(k, v.get_labels().shape, v.get_features().shape)


def main():
    print()
    TorchConfig.set_random_seed()
    logging.basicConfig(level=logging.WARN)
    logging.getLogger('zensols.deeplearn.model').setLevel(logging.WARN)
    #dataset()
    #metadata()
    batch()


main()