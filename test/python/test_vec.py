import logging
import torch
from util import TargetTestCase
from zensols.deeplearn.vectorize import (
    FeatureVectorizer, FeatureVectorizerManager,
    MaskFeatureVectorizer
)

logger = logging.getLogger(__name__)


class TestVectorizers(TargetTestCase):
    CONF_FILE = 'test-resources/vectorizers.conf'
    MASK_SIZES = 6, 1, 4, 0, 3

    def test_sized_mask(self):
        mng: FeatureVectorizerManager = self.fac('vectorizer_manager')
        vec: FeatureVectorizer = mng.vectorizers['mask_sized']
        self.assertEqual(MaskFeatureVectorizer, type(vec))
        self.assertEqual(10, vec.size)
        inp = tuple(map(lambda ln: tuple(range(ln)), self.MASK_SIZES))
        out = vec.transform(inp)
        should = torch.tensor([
            [True, True, True, True, True, True, False, False, False, False],
            [True, False, False, False, False, False, False, False, False, False],
            [True, True, True, True, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False, False],
            [True, True, True, False, False, False, False, False, False, False]])
        self.assertTensorEquals(should, out)

    def test_non_sized_mask(self):
        mng: FeatureVectorizerManager = self.fac('vectorizer_manager')
        vec: FeatureVectorizer = mng.vectorizers['mask_non_sized']
        self.assertEqual(MaskFeatureVectorizer, type(vec))
        self.assertEqual(-1, vec.size)
        inp = tuple(map(lambda ln: tuple(range(ln)), self.MASK_SIZES))
        out = vec.transform(inp)
        should = torch.tensor([
            [True, True, True, True, True, True],
            [True, False, False, False, False, False],
            [True, True, True, True, False, False],
            [False, False, False, False, False, False],
            [True, True, True, False, False, False]])
        self.assertTensorEquals(should, out)
