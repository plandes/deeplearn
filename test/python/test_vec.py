import logging
import pandas as pd
from zensols.deeplearn import FeatureVectorizerManager
from util import TargetTestCase

logger = logging.getLogger(__name__)


class TargetTestCase(TargetTestCase):
    CONF = 'vectorize'

    def test_manager(self):
        vm = self.fac('iris_vectorizer_manager')
        self.assertTrue(isinstance(vm, FeatureVectorizerManager))
        self.assertEqual(set(['ilabel', 'iseries']), vm.feature_types)

    def test_stash_types(self):
        stash = self.fac('dataset_stash')
        row = stash['1']
        self.assertTrue(isinstance(row, pd.Series))

    def test_vectorize_label(self):
        vm = self.fac('iris_vectorizer_manager')
        vec = vm['ilabel']
        arr = vec.transform(['setosa', 'versicolor', 'virginica', 'setosa', 'versicolor'])
        should = vm.torch_config.singleton(
            [[1., 0., 0.],
             [0., 1., 0.],
             [0., 0., 1.],
             [1., 0., 0.],
             [0., 1., 0.]])
        self.assertTensorEquals(should, arr)
