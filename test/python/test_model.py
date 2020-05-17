import logging
import unittest
from zensols.config import ExtendedInterpolationEnvConfig as AppConfig
from zensols.config import ImportConfigFactory
from zensols.deeplearn import TorchConfig

logger = logging.getLogger(__name__)


class TestModel(unittest.TestCase):
    def setUp(self):
        config = AppConfig(f'test-resources/executor.conf',
                           env={'app_root': '.'})
        self.fac = ImportConfigFactory(config, shared=True, reload=False)

    def assertClose(self, da, db):
        assert set(da.keys()) == set(db.keys())
        for k in da.keys():
            a = da[k]
            b = db[k]
            self.assertTrue(TorchConfig.close(a, b))

    def test_train(self):
        executor = self.fac('executor')
        executor.progress_bar = False
        executor.model_manager.keep_last_state_dict = True
        logger.debug(f'using device {executor.torch_config.device}')
        executor.train()
        logger.debug('testing trained model')
        executor.load_model()
        res = executor.test()
        self.assertLess(res.test.get_loss(), 0.1)
        self.assertGreater(res.test.micro_metrics['f1'], 0.2)
        self.assertGreater(res.test.macro_metrics['f1'], 0.2)

        tns = executor.model_manager.last_saved_state_dict
        ma = executor.model_manager.load_state_dict()
        self.assertClose(tns, ma)

        executor = self.fac('executor')
        res2 = executor.result_manager.load()

        self.assertEqual(res.train.get_loss(), res2.train.get_loss())
        self.assertEqual(res.validation.get_loss(), res2.validation.get_loss())
        self.assertEqual(res.test.get_loss(), res2.test.get_loss())

        self.assertEqual(res.validation.micro_metrics,
                         res2.validation.micro_metrics)
        self.assertEqual(res.train.micro_metrics, res2.train.micro_metrics)
        self.assertEqual(res.test.micro_metrics, res2.test.micro_metrics)
