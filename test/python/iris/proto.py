from dataclasses import dataclass
import sys
import logging
from zensols.persist import Deallocatable, dealloc
from zensols.config import ExtendedInterpolationEnvConfig as AppConfig
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


def create_facade(*args, **kwargs) -> IrisModelFacade:
    Deallocatable.ALLOCATION_TRACKING = True
    config = AppConfig('test-resources/iris/iris.conf', env={'app_root': '.'})
    return IrisModelFacade(config, *args, **kwargs)


def load():
    Deallocatable.ALLOCATION_TRACKING = True
    from pathlib import Path
    path = Path('target/iris/model.pt')
    with dealloc(IrisModelFacade.load_from_path(path)) as facade:
        facade.reload()
        facade.writer = None
        res = facade.test()
        res.write(include_converged=True)
        facade.plot_result(save=True)


def end():
    print('deallocations:')
    Deallocatable._print_undeallocated(True)


def find_leaks():
    #logging.getLogger('zensols.persist.annotation.Deallocatable').setLevel(logging.DEBUG)
    Deallocatable.ALLOCATION_TRACKING = True
    #Deallocatable.PRINT_TRACE = True
    fac = create_facade()
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
    #logging.getLogger('zensols.config.meta').setLevel(logging.DEBUG)
    logger.setLevel(logging.INFO)
    run = [3, 4, 5, 6, 7, 8]
    res = None
    if run == [0]:
        res = find_leaks()
    elif run is None:
        load()
        res = end()
    else:
        fac = create_facade()
        fac.epochs = 50
        for r in run:
            res = {0: fac.dataset,
                   1: lambda: fac.write(include_object_graph=True),
                   2: fac.debug,
                   3: fac.train,
                   4: fac.test,
                   5: fac.persist_results,
                   6: fac.write_result,
                   7: load,
                   8: fac.deallocate,
                   9: end}[r]()
    return res


res = main()
