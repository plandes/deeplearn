import logging
import itertools as it
from zensols.config import ExtendedInterpolationEnvConfig as AppConfig
from zensols.config import ImportConfigFactory
from zensols.deeplearn import TorchConfig

logger = logging.getLogger(__name__)


def factory(reload=True):
    config = AppConfig('test-resources/adult/adult.conf',
                       env={'app_root': '.'})
    fac = ImportConfigFactory(config, shared=True, reload_root=reload)
    return fac


def dataset():
    logging.getLogger('adult.data').setLevel(logging.DEBUG)
    fac = factory()
    ds = fac('adult_dataset_split_stash')
    #ds.clear()
    ds.write()
    train = ds.splits['train']
    import pandas as pd
    s = pd.DataFrame(it.islice(train.values(), 10))
    print(s)


def dataframe():
    fac = factory()
    ds = fac('adult_dataset_split_stash')
    df = ds.delegate.dataframe
    print(df['age'].max())
    print(df['education_num'].unique())
    print(df['capital_gain'].max())


def metadata():
    logging.getLogger('adult.data').setLevel(logging.DEBUG)
    fac = factory()
    mng = fac('adult_vectorizer_manager')
    #mng.dataset_metadata.write()
    print(mng.feature_types)
    mng.write()


def stash_info():
    from zensols.config.meta import ClassExplorer
    from zensols.persist import Stash
    fac = factory(False)
    stash = fac('adult_batch_stash')
    ClassExplorer(set([Stash])).write(stash)
    stash.delegate.feature_vectorizer_manager.write()
    stash.write()


def batch():
    logging.getLogger('zensols.deeplearn.batch.domain').setLevel(logging.INFO)
    #logging.getLogger('zensols.persist.composite').setLevel(logging.INFO)
    logging.getLogger('adult.data').setLevel(logging.DEBUG)
    fac = factory(False)
    stash = fac('adult_batch_stash')
    stash.write()
    print(f'flat shape: {stash.delegate.flattened_features_shape}')
    print(f'flat shape: {stash.delegate.label_shape}')
    for k, v in it.islice(stash, 1):
        print(k, v.get_labels().shape, v.get_features().shape)


def model():
    #logging.getLogger('zensols.deeplearn.batch.domain').setLevel(logging.INFO)
    logging.getLogger('adult.data').setLevel(logging.DEBUG)
    fac = factory(False)
    executor = fac('executor', progress_bar=True)
    executor.write()
    executor.train()
    print(executor.model)
    executor.load()
    res = executor.test()
    res.write(verbose=False)


def tmp():
    logging.getLogger('zensols.deeplearn.batch.domain').setLevel(logging.DEBUG)
    fac = factory(False)
    stash = fac('adult_batch_stash')
    stash.prime()


def main():
    print()
    TorchConfig.set_random_seed()
    logging.basicConfig(level=logging.WARN)
    logging.getLogger('zensols.deeplearn.model').setLevel(logging.WARN)
    run = 5
    {0: dataset,
     1: dataframe,
     2: metadata,
     3: stash_info,
     4: batch,
     5: model,
     6: tmp}[run]()


main()
