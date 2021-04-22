"""Provides neural network layer implementations, which are all subclasses of
:class:`torch.nn.Module`.

"""

from zensols.deeplearn import ModelError


class LayerError(ModelError):
    """Thrown for all deep learning layer errors.

    """
    pass


from .linear import *
from .crf import *
from .recur import *
from .recurcrf import *
from .conv import *
