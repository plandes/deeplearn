import logging
from zensols.config import ExtendedInterpolationEnvConfig as AppConfig
from zensols.config import ImportConfigFactory
from zensols.deeplearn import TorchConfig
from zensols.deeplearn.model import ModelManager

logger = logging.getLogger(__name__)


def factory():
    """Create the configuration factory.

    """
    config = AppConfig('test-resources/iris/iris.conf', env={'app_root': '.'})
    fac = ImportConfigFactory(config, shared=True, reload=False)
    return fac


def dataset():
    """Print information about the dataset.

    """
    fac = factory()
    stash = fac('iris_dataset_stash')
    for batch in stash.splits['train'].values():
        print(batch)
        print(', '.join(batch.data_point_ids))
        print(batch.get_labels())
        print(batch.get_flower_dimensions())
        print('-' * 20)


def train_model():
    """Train, test the model, and save the results to the file system.

    """
    fac = factory()
    executor = fac('executor', progress_bar=True, progress_bar_cols=100)
    executor.write()
    if executor.net_settings.debug:
        logging.getLogger('iris.model').setLevel(logging.DEBUG)
        executor.progress_bar = False
    logger.info(f'using device: {executor.torch_config.device}')
    res = executor.train()
    if not executor.net_settings.debug:
        logger.info('testing trained model')
        executor.load()
        res = executor.test()
        res.write(verbose=False)
    return res


def test_model():
    """Load the model from disk and test it.

    """
    fac = factory()
    path = fac.config.populate(section='model_settings').path
    logger.info(f'testing from path: {path}')
    mm = ModelManager(path, fac)
    executor = mm.load_executor()
    res = executor.test()
    res.write(verbose=False)


def load_results():
    """Load the last set of results from the file system and print them out.

    """
    logging.getLogger('zensols.deeplearn.result').setLevel(logging.INFO)
    print('load previous results')
    fac = factory()
    executor = fac('executor')
    res = executor.result_manager.load()
    res.write(verbose=False)


def main():
    print()
    TorchConfig.set_random_seed()
    logging.basicConfig(level=logging.WARN)
    logging.getLogger('zensols.deeplearn.model').setLevel(logging.WARN)
    logger.setLevel(logging.INFO)
    run = [1]
    #run = [1, 2, 3]
    res = None
    for r in run:
        res = {0: dataset,
               1: train_model,
               2: test_model,
               3: load_results}[r]()
    return res


res = main()
