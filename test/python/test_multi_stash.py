import logging
from pathlib import Path
from util import TargetTestCase
import zensols.deeplearn.batch

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TargetMultiStash(TargetTestCase):
    CONF = 'vectorize'

    def setUp(self):
        super().setUp()
        self.stash = self.fac('batch_dataset_stash')

    def test_create(self):
        batch_path = Path('target/batch/data')
        stash = self.stash
        stash.clear()
        self.assertEqual(10, len(stash))
        self.assertEqual(10, len(tuple(batch_path.iterdir())))
        stash.clear()
        self.assertEqual(0, len(tuple(batch_path.iterdir())))
        self.assertEqual(10, len(stash))
        self.assertEqual(10, len(tuple(batch_path.iterdir())))

    def test_attributes(self):
        stash = self.stash
        self.assertEqual(10, len(stash))
        for k, v in stash:
            self.assertTrue(isinstance(int(k), int))
            self.assertEqual(3, v.get_labels().shape[1])
            self.assertEqual(4, v.get_flower_dimensions().shape[1])
