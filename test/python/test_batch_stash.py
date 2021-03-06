import logging
from pathlib import Path
from util import TargetTestCase

logger = logging.getLogger(__name__)


class TestBatchStash(object):
    CONF = 'batch-stash'

    BATCH_SIZE = 6

    def test_create(self):
        batch_path = Path('target/stash/batch/data')
        stash = self.stash
        stash.clear()
        self.assertEqual(self.BATCH_SIZE, len(stash))
        self.assertEqual(self.BATCH_SIZE, len(tuple(batch_path.iterdir())))
        stash.clear()
        self.assertEqual(0, len(tuple(batch_path.iterdir())))
        self.assertEqual(self.BATCH_SIZE, len(stash))
        self.assertEqual(self.BATCH_SIZE, len(tuple(batch_path.iterdir())))

    def test_recreate(self):
        stash = self.stash
        self.assertEqual(self.BATCH_SIZE, len(stash))
        for k, v in stash:
            self.assertTrue(isinstance(int(k), int))
            self.assertEqual(3, v.get_labels().shape[1])
            self.assertEqual(4, v.get_flower_dimensions().shape[1])
        stash.clear()
        for k, v in stash:
            self.assertTrue(isinstance(int(k), int))
            self.assertEqual(3, v.get_labels().shape[1])
            self.assertEqual(4, v.get_flower_dimensions().shape[1])

    def test_attributes(self):
        stash = self.stash
        self.assertEqual(self.BATCH_SIZE, len(stash))
        for k, v in stash:
            self.assertTrue(isinstance(int(k), int))
            self.assertEqual(3, v.get_labels().shape[1])
            self.assertEqual(4, v.get_flower_dimensions().shape[1])

    def test_manager_config(self):
        ms = self.stash.vectorizer_manager_set
        self.assertEqual(set('iseries ilabel'.split()), ms.feature_ids)
        attribs = set('label flower_dims'.split())
        for k, v in self.stash:
            self.assertEqual(attribs, set(v.attributes.keys()))
            # use batch mapping if this is ever needed again
            #self.assertEqual({'ilabel': 'label', 'iseries': 'flower_dims'}, v.feature_ids)

    def test_manager_feature_subset(self):
        stash = self.fac('feature_subset_batch_dataset_stash')
        ms = stash.vectorizer_manager_set
        self.assertEqual(set('iseries ilabel'.split()), ms.feature_ids)
        attribs = set('label'.split())
        for k, v in stash:
            self.assertEqual(attribs, set(v.attributes.keys()))
            # use batch mapping if this is ever needed again
            #self.assertEqual({'ilabel': 'label'}, v.feature_ids)

    def test_to_gpu(self):
        stash = self.stash
        model_tc = stash.model_torch_config
        model_dev = model_tc.device
        cpu_dev = model_tc.cpu_device
        logger.info(f'cpu={cpu_dev}, model={model_dev}')
        for k, batch in stash:
            self.assertEqual(cpu_dev, batch.get_labels().device)
            m_batch = batch.to()
            self.assertEqual(model_dev, m_batch.get_labels().device)
            self.assertNotEqual(id(batch), id(m_batch))
            self.assertEqual(batch.id, m_batch.id)
            self.assertEqual(batch.split_name, m_batch.split_name)
            self.assertEqual(batch.data_point_ids, m_batch.data_point_ids)
        for k, batch in stash:
            dev = batch.get_labels().device
            self.assertEqual(cpu_dev, dev)

    def test_key_split(self):
        stash = self.stash
        cnts = stash.counts_by_key
        keys_by_split = stash.keys_by_split
        self.assertEqual(set('dev train test'.split()), stash.split_names)
        self.assertEqual(3, len(cnts))
        self.assertEqual(3, len(keys_by_split))
        self.assertEqual(len(stash),
                         sum(map(len, stash.keys_by_split.values())))


class TestBatchStashNonSplit(TargetTestCase, TestBatchStash):
    def setUp(self):
        super().setUp()
        self.stash = self.fac('batch_dataset_stash')


class TestBatchStashSplit(TargetTestCase, TestBatchStash):
    def setUp(self):
        super().setUp()
        self.stash = self.fac('batch_split_dataset_stash')
        self.stash.delegate_attr = True

    def test_splits(self):
        stash = self.stash
        kbs = stash.keys_by_split
        for sname in stash.split_names:
            sstash = stash.splits[sname]
            self.assertEqual(len(kbs[sname]), len(sstash))
            self.assertEqual(len(kbs[sname]), len(tuple(sstash.keys())))
            self.assertEqual(len(kbs[sname]), len(tuple(sstash.values())))
