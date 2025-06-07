"""Explore the facade in memory object graph

"""
__author__ = 'Paul Landes'


from zensols.config import ClassExplorer
from zensols.persist import Stash
from zensols.deeplearn.vectorize import (
    FeatureVectorizer,
    FeatureVectorizerManager,
    FeatureVectorizerManagerSet,
)
from . import ModelExecutor


class FacadeClassExplorer(ClassExplorer):
    """A class explorer that includes interesting and noteable framework classes
    to print.

    """
    def __init__(self, *args, **kwargs):
        if 'include_classes' in kwargs:
            include_classes = kwargs['include_classes']
        else:
            include_classes = set()
            kwargs['include_classes'] = include_classes
        incs = {Stash,
                FeatureVectorizer,
                FeatureVectorizerManager,
                FeatureVectorizerManagerSet,
                ModelExecutor}
        include_classes.update(incs)
        super().__init__(*args, **kwargs)
