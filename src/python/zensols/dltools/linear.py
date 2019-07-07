"""Linear layer tools

"""
__author__ = 'Paul Landes'

from torch import nn
from zensols.actioncli import persisted


class LinearLayerFactory(object):
    """Utility class to create linear layers.

    """
    def __init__(self, in_shape, out=None, out_percent=None):
        """Initialize the factory.

        :param in_shape: the shape of the layer
        :param out: the output size of the reshaped layer or use
                    ``out_percent`` if ``None``
        :param out_percent: the output reshaped layer as a percentage of the
                            input size ``in_shape`` if not None, othewise use
                            ``out`.

        """
        self.in_shape = in_shape
        if out is None:
            self.out_features = int(self.flatten_dim * out_percent)
        else:
            self.out_features = out

    @property
    @persisted('_flatten_dim')
    def flatten_dim(self):
        """Return the dimension of a flattened layer represened by the this instances's
        layer.

        """
        from functools import reduce
        return reduce(lambda x, y: x * y, self.in_shape)

    @property
    @persisted('_out_shape')
    def out_shape(self):
        """Return the shape of the output layer.

        """
        return (1, self.flatten_dim)

    def linear(self):
        """Return a new PyTorch linear layer.
        """
        return nn.Linear(self.flatten_dim, self.out_features)

    def __str__(self):
        return f'{self.in_shape} -> {self.out_shape}'
