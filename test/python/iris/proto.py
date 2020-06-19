from dataclasses import dataclass
import logging
from zensols.persist import Deallocatable
from zensols.config import ExtendedInterpolationEnvConfig as AppConfig
from zensols.config import ImportConfigFactory
from zensols.deeplearn import TorchConfig
from zensols.deeplearn.model import ModelFacade, ModelFacadeCacheLevel

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


def facade(*args, **kwargs) -> IrisModelFacade:
    config = AppConfig('test-resources/iris/iris.conf', env={'app_root': '.'})
    fac = ImportConfigFactory(config)
    return IrisModelFacade(fac, *args, **kwargs)


def load():
    from pathlib import Path
    path = Path('target/iris/model.pt')
    facade = IrisModelFacade.load_from_path(path)
    facade.writer = None
    res = facade.test()
    res.write(verbose=True)


def find_leaks():
    #logging.getLogger('zensols.persist.annotation.Deallocatable').setLevel(logging.DEBUG)
    Deallocatable.ALLOCATION_TRACKING = True
    #Deallocatable.PRINT_TRACE = True
    fac = facade(cache_level=ModelFacadeCacheLevel.EXECUTOR)
    executor = fac.executor

    if 0:
        from zensols.config import ClassExplorer
        from zensols.persist import Stash
        ce = ClassExplorer({Stash})
        ce.write(fac.executor.dataset_stash)

    executor.train()
    executor.test()
    fac.deallocate()
    Deallocatable._print_undeallocated(True)


def main():
    print()
    TorchConfig.set_random_seed()
    logging.basicConfig(level=logging.WARN)
    logging.getLogger('zensols.deeplearn.model').setLevel(logging.WARN)
    logger.setLevel(logging.INFO)
    run = [3, 4, 5, 6]
    run = [0]
    res = None
    if run == [0]:
        res = find_leaks()
    else:
        fac = facade()
        fac.set_epochs(50)
        for r in run:
            res = {1: fac.dataset,
                   2: fac.debug,
                   3: fac.train,
                   4: fac.test,
                   5: fac.write_results,
                   6: load}[r]()
    return res


res = main()
