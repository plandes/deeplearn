from dataclasses import dataclass
import logging
from zensols.config import ExtendedInterpolationEnvConfig as AppConfig
from zensols.config import ImportConfigFactory
from zensols.deeplearn import TorchConfig
from zensols.deeplearn.model import ModelFacade

logger = logging.getLogger(__name__)


@dataclass
class IrisModelFacade(ModelFacade):
    def _configure_debug_logging(self):
        super()._configure_debug_logging()
        logging.getLogger('iris.model').setLevel(logging.DEBUG)

    def dataset(self):
        """Print information about the dataset.

        """
        stash = self.factory('iris_dataset_stash')
        for batch in stash.splits['train'].values():
            print(batch)
            print(', '.join(batch.data_point_ids))
            print(batch.get_labels())
            print(batch.get_flower_dimensions())
            print('-' * 20)


def facade() -> IrisModelFacade:
    config = AppConfig('test-resources/iris/iris.conf', env={'app_root': '.'})
    fac = ImportConfigFactory(config)
    return IrisModelFacade(fac)


def main():
    print()
    TorchConfig.set_random_seed()
    logging.basicConfig(level=logging.WARN)
    logging.getLogger('zensols.deeplearn.model').setLevel(logging.WARN)
    logger.setLevel(logging.INFO)
    fac = facade()
    run = [1]
    #run = [1, 2, 3]
    res = None
    for r in run:
        res = {0: fac.dataset,
               1: fac.debug,
               2: fac.train,
               3: fac.test,
               4: fac.write_results}[r]()
    return res


res = main()
