import logging
from pathlib import Path
from util import TargetTestCase

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

    def test_manager_config(self):
        ms = self.stash.vectorizer_manager_set
        self.assertEqual(set('iseries ilabel'.split()), ms.feature_types)
        attribs = set('label flower_dims'.split())
        for k, v in self.stash:
            self.assertEqual(attribs, set(v.attributes.keys()))

    def test_manager_feature_subset(self):
        self.stash = self.fac('feature_subset_batch_dataset_stash')
        ms = self.stash.vectorizer_manager_set
        self.assertEqual(set('iseries ilabel'.split()), ms.feature_types)
        attribs = set('label'.split())
        for k, v in self.stash:
            self.assertEqual(attribs, set(v.attributes.keys()))
