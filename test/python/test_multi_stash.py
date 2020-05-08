import logging
from pathlib import Path
from util import TargetTestCase

logger = logging.getLogger(__name__)


class TargetMultiStash(TargetTestCase):
    CONF = 'vectorize'

    def test_multi_stash(self):
        batch_path = Path('target/batch/data')
        stash = self.fac('batch_dataset_stash')
        stash.clear()
        self.assertEqual(10, len(stash))
        self.assertEqual(10, len(tuple(batch_path.iterdir())))
        stash.clear()
        self.assertEqual(0, len(tuple(batch_path.iterdir())))
        self.assertEqual(10, len(stash))
        self.assertEqual(10, len(tuple(batch_path.iterdir())))
