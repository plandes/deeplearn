"""Multi processing with torch.

"""
__author__ = 'Paul Landes'
import warnings
from ..multi import TorchMultiProcessStash


warnings.warn(
    ('module zensols.deeplearn.batch.multi will be removed in version 1.15 '
     'Please use zensols.deeplearn.multi instead.'),
    DeprecationWarning,
    # Points the warning to the caller's line number
    stacklevel=2)
