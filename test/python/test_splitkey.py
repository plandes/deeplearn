from dataclasses import dataclass
import unittest
import logging
import shutil
import json
from pathlib import Path
from zensols.persist import ReadOnlyStash, CacheStash
from zensols.config import ExtendedInterpolationConfig as AppConfig
from zensols.config import ImportConfigFactory
from zensols.deeplearn import SplitStashContainer
from util import TargetTestCase

logger = logging.getLogger(__name__)

if 0:
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('zensols.deeplearn.stash').setLevel(logging.DEBUG)


class RangeStash(ReadOnlyStash):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def load(self, name: str):
        n = int(name)
        if n >= self.n:
            return None
        return name

    def keys(self):
        return map(str, range(self.n))


@dataclass
class DelegatingCasheStash(CacheStash):
    def __post_init__(self):
        super().__post_init__()
        self.delegate_attr = True


class TestSplitKey(TargetTestCase):
    CONF = 'splitkey'

    def setUp(self):
        super().setUp()
        with open('test-resources/keys.json') as f:
            keys_cont = json.load(f)
        self.keys = {k: set(keys_cont[k]) for k in keys_cont}

        with open('test-resources/range-keys.json') as f:
            keys_cont = json.load(f)
        self.keys_range = {k: set(keys_cont[k]) for k in keys_cont}

        self.df_path = Path('target/df.dat')
        self.key_path = Path('target/keys.dat')

    def _test_len(self, stash, should_len):
        self.assertEqual(should_len, len(stash))
        self.assertTrue(self.df_path.exists())
        should = set(map(str, range(should_len)))
        self.assertEqual(should, set(stash.keys()))
        should = set('train test dev'.split())
        self.assertEqual(should, stash.split_names)

    def test_split_key(self):
        self.assertFalse(self.df_path.exists())
        stash = self.fac('splitkey_stash')
        should_len = 150
        self.assertEqual((should_len, 6), stash.dataframe.shape)
        self._test_len(stash, should_len)
        self.assertFalse(self.key_path.exists())
        self.assertEqual(self.keys, stash.keys_by_split)
        self.assertTrue(self.key_path.exists())
        should = {'sepal_length': 4.9,
                  'sepal_width': 3.0,
                  'petal_length': 1.4,
                  'petal_width': 0.2,
                  'species': 'setosa',
                  'ds_type': 'train'}
        self.assertTrue(should, (dict(stash['1'])))
        stash.clear()
        self.assertFalse(self.df_path.exists())
        self.assertFalse(self.key_path.exists())

    def _test_split_ds(self, stash):
        self.assertFalse(self.key_path.exists())
        slen = 30
        self._test_len(stash, slen)
        self.assertTrue(self.df_path.exists())
        self.assertTrue(self.key_path.exists())
        for i in range(slen):
            self.assertEqual(i, stash[i])
        self.assertFalse(stash.exists(slen))
        self.assertEqual(self.keys_range, stash.keys_by_split)

    def test_split_ds(self):
        self.assertFalse(self.df_path.exists())
        stash = self.fac('dataset_stash')
        self._test_split_ds(stash)
        stash.clear()
        self.assertFalse(self.df_path.exists())
        self.assertFalse(self.key_path.exists())

    def test_split_ds_cache_back(self):
        self.assertFalse(self.df_path.exists())
        stash = self.fac('cached_dataset_stash')
        self._test_split_ds(stash)
        stash.clear()
        self.assertFalse(self.df_path.exists())
        self.assertFalse(self.key_path.exists())

    def test_get_split(self):
        stash = self.fac('dataset_stash')
        self.assertEqual(3, len(self.keys_range))
        for k, v in self.keys_range.items():
            ds = stash.splits[k]
            self.assertEqual(len(self.keys_range[k]), len(ds.keys()))
            self.assertEqual(len(self.keys_range[k]), len(tuple(ds.values())))
            self.assertEqual(self.keys_range[k], set(ds.keys()))
        train = stash.splits['train']
        self.assertEqual('train', train.split_name)
        self.assertTrue(isinstance(train, SplitStashContainer))
        pairs = tuple(train)
        self.assertEqual(29, len(pairs))
        for i, v in pairs:
            self.assertEqual(i, v)

    def test_key_type(self):
        stash = self.fac('splitkey_stash')
        keys = tuple(stash.keys())
        self.assertTrue(len(keys) > 0)
        for k in keys:
            self.assertEquals(str, type(k))
        for split, keys in stash.keys_by_split.items():
            keys = tuple(keys)
            self.assertTrue(len(keys) > 0)
            self.assertEquals(str, type(keys[0]))
