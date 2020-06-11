from dataclasses import dataclass, field
import sys
import logging
import pandas as pd
from io import TextIOWrapper
from zensols.config import Configurable, ConfigFactory
from zensols.persist import persisted
from zensols.util import time
from . import ModelManager, ModelExecutor

logger = logging.getLogger(__name__)


@dataclass
class ModelFacade(object):
    factory: ConfigFactory
    progress_bar: bool = field(default=True)
    progress_bar_cols: int = field(default=79)
    debug: bool = field(default=False)
    executor_name: str = field(default='executor')

    @property
    @persisted('_executor')
    def executor(self) -> ModelExecutor:
        executor = self.factory(
            self.executor_name,
            progress_bar=self.progress_bar,
            progress_bar_cols=self.progress_bar_cols)
        executor.net_settings.debug = self.debug
        return executor

    @property
    def config(self) -> Configurable:
        return self.factory.config

    def train(self):
        """Train and test or just debug the model depending on the configuration.

        """
        executor = self.executor
        try:
            if self.debug:
                self._configure_debug_logging()
                executor.progress_bar = False
                executor.model_settings.batch_limit = 1
            executor.write()
            logger.info('training...')
            with time('trained'):
                res = executor.train()
            if not self.debug:
                logger.info('testing...')
                with time('tested'):
                    res = executor.test()
                res.write()
        finally:
            executor.deallocate()

    def test(self):
        """Load the model from disk and test it.

        """
        path = self.config.populate(section='model_settings').path
        logger.info(f'testing from path: {path}')
        mm = ModelManager(path, self.factory)
        executor = mm.load_executor()
        res = executor.test()
        res.write(verbose=False)

    def write_results(self, depth: int = 0, writer: TextIOWrapper = sys.stdout,
                      verbose: bool = False):
        """Load the last set of results from the file system and print them out.

        """
        logging.getLogger('zensols.deeplearn.result').setLevel(logging.INFO)
        logger.info('load previous results')
        res = self.executor.result_manager.load()
        if res is None:
            raise ValueError('no results found')
        res.write(depth, writer, verbose)

    def get_predictions(self, *args, **kwargs) -> pd.DataFrame:
        executor = self.executor
        executor.load()
        return executor.get_predictions(*args, **kwargs)

    def write_predictions(self, writer: TextIOWrapper = sys.stdout):
        preds = self.get_predictions()
        print(preds.head(), file=writer)

    def _configure_debug_logging(self):
        lg = logging.getLogger('zensols.deepnlp.vectorize.vectorizers')
        lg.setLevel(logging.INFO)

    @staticmethod
    def no_sparse():
        from zensols.deeplearn.vectorize import SparseTensorFeatureContext
        SparseTensorFeatureContext.USE_SPARSE = False
