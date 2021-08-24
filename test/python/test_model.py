import logging
import unittest
from pathlib import Path
from zensols.config import ExtendedInterpolationEnvConfig as AppConfig
from zensols.config import ImportConfigFactory
from zensols.persist import Deallocatable
from zensols.deeplearn import TorchConfig
from zensols.deeplearn.model import ModelFacade
from iris.model import IrisBatch

logger = logging.getLogger(__name__)


class TestModelBase(unittest.TestCase):
    def setUp(self):
        TorchConfig.init()
        config = AppConfig('test-resources/iris/iris.conf',
                           env={'app_root': '.'})
        self.config = config
        self.fac = ImportConfigFactory(config, shared=True, reload=False)

    def validate_results(self, res):
        self.assertLess(res.test.ave_loss, 5)
        self.assertGreater(res.test.metrics.micro.f1, 0.4)
        self.assertGreater(res.test.metrics.macro.f1, 0.4)


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
        executor.result_manager.dump(res)
        self.validate_results(res)

        ma = executor.model_manager._get_checkpoint(True)['model_state_dict']
        self.assertClose(tns, ma)

        executor = self.fac('executor')
        res2 = executor.result_manager.load()

        self.assertEqual(res.train.ave_loss, res2.train.ave_loss)
        self.assertEqual(res.validation.ave_loss, res2.validation.ave_loss)
        self.assertEqual(res.test.ave_loss, res2.test.ave_loss)

        self.assertEqual(res.validation.metrics.asdict(),
                         res2.validation.metrics.asdict())
        self.assertEqual(res.train.metrics.asdict(), res2.train.metrics.asdict())
        self.assertEqual(res.test.metrics.asdict(), res2.test.metrics.asdict())

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
        IrisBatch.TEST_ON = False

    def test_no_cache(self):
        executor = self.executor
        executor.model_settings.batch_iteration = 'gpu'
        executor.model_settings.cache_batches = False
        logger.debug(f'using device {executor.torch_config.device}')
        executor.train()
        self.assertEqual(7, len(IrisBatch.TEST_INSTANCES))
        for b in IrisBatch.TEST_INSTANCES:
            self.assertEqual('deallocated', b.state_name)

    def test_cache(self):
        executor = self.executor
        executor.model_settings.cache_batches = True
        executor.model_settings.batch_iteration = 'gpu'
        logger.debug(f'using device {executor.torch_config.device}')
        executor.train()
        self.assertEqual(7, len(IrisBatch.TEST_INSTANCES))
        for b in IrisBatch.TEST_INSTANCES:
            self.assertEqual('memory copied', b.state_name)

    def test_no_cache_cpu(self):
        executor = self.executor
        executor.model_settings.cache_batches = False
        executor.model_settings.batch_iteration = 'cpu'
        logger.debug(f'using device {executor.torch_config.device}')
        executor.train()
        self.assertEqual(7, len(IrisBatch.TEST_INSTANCES))
        for b in IrisBatch.TEST_INSTANCES:
            self.assertEqual('deallocated', b.state_name)

    def test_cache_cpu(self):
        executor = self.executor
        executor.model_settings.cache_batches = True
        executor.model_settings.batch_iteration = 'cpu'
        logger.debug(f'using device {executor.torch_config.device}')
        executor.train()
        self.assertEqual(7, len(IrisBatch.TEST_INSTANCES))
        for b in IrisBatch.TEST_INSTANCES:
            self.assertEqual('memory copied', b.state_name)


class TestFacade(TestModelBase):
    def test_facade(self):
        Deallocatable.ALLOCATION_TRACKING = True
        facade = ModelFacade(self.config, progress_bar=False)
        facade.writer = None
        facade.train()
        res = facade.test()
        self.validate_results(res)
        facade.deallocate()
        path = Path('target/iris/model')
        facade = ModelFacade.load_from_path(path, progress_bar=False)
        facade.writer = None
        res = facade.test()
        self.validate_results(res)
        facade.deallocate()
        #Deallocatable._print_undeallocated(True)
        self.assertEqual(0, Deallocatable._num_deallocations())
        Deallocatable.ALLOCATION_TRACKING = False
