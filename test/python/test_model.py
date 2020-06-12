import logging
import unittest
from zensols.config import ExtendedInterpolationEnvConfig as AppConfig
from zensols.config import ImportConfigFactory
from zensols.deeplearn import TorchConfig
from iris.model import IrisBatch

logger = logging.getLogger(__name__)


class TestModelBase(unittest.TestCase):
    def setUp(self):
        TorchConfig.set_random_seed()
        config = AppConfig('test-resources/iris/iris.conf',
                           env={'app_root': '.'})
        self.config = config
        self.fac = ImportConfigFactory(config, shared=True, reload=False)


class TestModel(TestModelBase):
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
        tns = executor.model_manager.last_saved_state_dict
        logger.debug('testing trained model')
        executor.load()
        res = executor.test()
        self.assertLess(res.test.ave_loss, 5)
        self.assertGreater(res.test.micro_metrics['f1'], 0.2)
        self.assertGreater(res.test.macro_metrics['f1'], 0.2)

        ma = executor.model_manager.checkpoint['model_state_dict']
        self.assertClose(tns, ma)

        executor = self.fac('executor')
        res2 = executor.result_manager.load()

        self.assertEqual(res.train.ave_loss, res2.train.ave_loss)
        self.assertEqual(res.validation.ave_loss, res2.validation.ave_loss)
        self.assertEqual(res.test.ave_loss, res2.test.ave_loss)

        self.assertEqual(res.validation.micro_metrics,
                         res2.validation.micro_metrics)
        self.assertEqual(res.train.micro_metrics, res2.train.micro_metrics)
        self.assertEqual(res.test.micro_metrics, res2.test.micro_metrics)

    def test_net_params(self):
        mfeats = self.config.get_option('middle_features', 'net_settings')
        self.assertEqual('eval: [5, 1]', mfeats)
        executor = self.fac('executor')
        self.assertEqual([5, 1], executor.net_settings.middle_features)
        self.assertEqual([5, 1], executor.get_network_parameter('middle_features'))
        executor.set_network_parameter('middle_features', [1, 2, 3])
        self.assertEqual([1, 2, 3], executor.net_settings.middle_features)
        self.assertEqual([1, 2, 3], executor.get_network_parameter('middle_features'))

    def test_model_params(self):
        bi = self.config.get_option('batch_iteration', 'model_settings')
        self.assertEqual('gpu', bi)
        executor = self.fac('executor')
        self.assertEqual('gpu', executor.model_settings.batch_iteration)
        self.assertEqual('gpu', executor.get_model_parameter('batch_iteration'))
        executor.set_model_parameter('batch_iteration', 'cpu')
        self.assertEqual('cpu', executor.model_settings.batch_iteration)
        self.assertEqual('cpu', executor.get_model_parameter('batch_iteration'))


class TestModelDeallocate(TestModelBase):
    def setUp(self):
        super().setUp()
        IrisBatch.TEST_ON = True
        self.executor = self.fac('executor')
        self.executor.progress_bar = False
        self.executor.model_settings.epochs = 1

    def tearDown(self):
        IrisBatch.TEST_INSTANCES.clear()

    def test_no_cache(self):
        executor = self.executor
        executor.model_settings.batch_iteration = 'gpu'
        executor.cache_batches = False
        logger.debug(f'using device {executor.torch_config.device}')
        executor.train()
        self.assertEqual(7, len(IrisBatch.TEST_INSTANCES))
        for b in IrisBatch.TEST_INSTANCES:
            self.assertEqual('deallocated', b.state_name)

    def test_cache(self):
        executor = self.executor
        executor.cache_batches = True
        executor.model_settings.batch_iteration = 'gpu'
        logger.debug(f'using device {executor.torch_config.device}')
        executor.train()
        self.assertEqual(7, len(IrisBatch.TEST_INSTANCES))
        for b in IrisBatch.TEST_INSTANCES:
            self.assertEqual('memory copied', b.state_name)

    def test_no_cache_cpu(self):
        executor = self.executor
        executor.cache_batches = False
        executor.model_settings.batch_iteration = 'cpu'
        logger.debug(f'using device {executor.torch_config.device}')
        executor.train()
        self.assertEqual(7, len(IrisBatch.TEST_INSTANCES))
        for b in IrisBatch.TEST_INSTANCES:
            self.assertEqual('deallocated', b.state_name)

    def test_cache_cpu(self):
        executor = self.executor
        executor.cache_batches = True
        executor.model_settings.batch_iteration = 'cpu'
        logger.debug(f'using device {executor.torch_config.device}')
        executor.train()
        self.assertEqual(7, len(IrisBatch.TEST_INSTANCES))
        for b in IrisBatch.TEST_INSTANCES:
            self.assertEqual('memory copied', b.state_name)
