"""Utiliies for encoding and decoding tensors.

"""
__author__ = 'Paul Landes'

from typing import Tuple, List, Sequence
from dataclasses import dataclass
from itertools import chain
from functools import reduce
import torch
from torch import Tensor
from . import TorchConfig


@dataclass
class NonUniformDimensionEncoder(object):
    """Encode a sequence of tensors, each of arbitrary dimensionality, as a 1-D
    array.  Then decode the 1-D array back to the original.

    """
    torch_config: TorchConfig

    def encode(self, arrs: Sequence[Tensor]) -> Tensor:
        """Encode a sequence of tensors, each of arbitrary dimensionality, as a
        1-D array.

        """
        def map_tensor_meta(arr: Tensor) -> Tuple[int]:
            sz = arr.shape
            tm = [len(sz)]
            tm.extend(sz)
            return tm

        tmeta = [len(arrs)]
        tmeta.extend(chain.from_iterable(map(map_tensor_meta, arrs)))
        tmeta = self.torch_config.singleton(tmeta, dtype=arrs[0].dtype)
        arrs = [tmeta] + list(map(lambda t: t.flatten(), arrs))
        enc = torch.cat(arrs)
        return enc

    def decode(self, arr: Tensor) -> Tuple[Tensor]:
        """Decode the 1-D array back to the original.

        """
        ix_type = torch.long
        shapes_len = arr[0].type(ix_type)
        one = torch.tensor([1], dtype=ix_type, device=arr.device)
        one.autograd = False
        start = one.clone()
        shapes: List[int] = []
        for i in range(shapes_len):
            sz_len = arr[start].type(ix_type)
            start += one
            end = (start + sz_len).type(ix_type)
            sz = arr[start:end]
            shapes.append(sz)
            start = end
        arrs = []
        for shape in shapes:
            ln = reduce(lambda x, y: x.type(ix_type) * y.type(ix_type), shape)
            end = (start + ln).type(ix_type)
            shape = tuple(map(int, shape))
            x = arr[start:end]
            x = x.view(shape)
            arrs.append(x)
            start = end
        return tuple(arrs)
