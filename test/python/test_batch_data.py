import logging
from util import TargetTestCase
from zensols.config import ImportConfigFactory

logger = logging.getLogger(__name__)


class TestBatchData(TargetTestCase):
    CONF_FILE = 'test-resources/iris/iris.conf'

    def get_ds(self):
        executor = self.fac('executor')
        train, valid, test = executor._get_dataset_splits()
        ds = tuple(map(lambda b: b.to(), train.values()))
        del executor
        del valid
        del test
        del train
        return ds

    def _test_batch_consistency(self, use_gpu):
        self.config.set_option('use_gpu', str(use_gpu), 'gpu_torch_config')
        self.fac = ImportConfigFactory(self.config)
        a_ds = self.get_ds()

        self.config.set_option('use_gpu', str(use_gpu), 'gpu_torch_config')
        self.fac = ImportConfigFactory(self.config)
        b_ds = self.get_ds()

        for i in range(2):
            for a, b in zip(a_ds, b_ds):
                al = a.get_labels()
                bl = b.get_labels()
                af = a.get_flower_dimensions()
                bf = b.get_flower_dimensions()
                # there's a batch that contain all enum=0 in the labels
                #self.assertGreater(al.sum(), 0)
                #self.assertGreater(bl.sum(), 0)
                self.assertGreater(af.sum(), 0)
                self.assertGreater(bf.sum(), 0)
                self.assertEqual(al.device, bl.device)
                self.assertEqual(af.device, bf.device)
                self.assertEqual(al.device, af.device)
                self.assertNotEqual(id(a), id(b))
                self.assertNotEqual(id(al), id(bl))
                self.assertTensorEquals(al, bl)
                self.assertNotEqual(id(af), id(bf))
                self.assertTensorEquals(af, bf)

    def test_batch_consistency(self):
        for i in range(10):
            self._test_batch_consistency(False)
        for i in range(10):
            self._test_batch_consistency(True)
