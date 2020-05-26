import logging
from pathlib import Path
import shutil
import unittest
from zensols.config import ExtendedInterpolationConfig as AppConfig
from zensols.config import ExtendedInterpolationEnvConfig as AppEnvConfig
from zensols.config import ImportConfigFactory
from zensols.deeplearn import TorchConfig

logger = logging.getLogger(__name__)


class TargetTestCase(unittest.TestCase):
    def recreate_factory(self):
        if hasattr(self.__class__, 'CONF_FILE'):
            path = self.CONF_FILE
            env = {'app_root': '.'}
            self.config = AppEnvConfig(path, env=env)
        else:
            path = f'test-resources/{self.CONF}.conf'
            self.config = AppConfig(path)
        self.fac = ImportConfigFactory(self.config)

    def setUp(self):
        self.recreate_factory()
        targ = Path('target')
        if targ.exists() and targ.is_dir():
            shutil.rmtree(targ)

    def assertTensorEquals(self, should, tensor):
        try:
            eq = TorchConfig.equal(should, tensor)
        except RuntimeError as e:
            logger.error(f'error comparing {should} with {tensor}')
            raise e
        if not eq:
            logger.error(f'tensor {should} does not equal {tensor}')
        self.assertTrue(eq)
