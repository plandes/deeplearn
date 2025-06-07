"""CUDA access and utility module.

"""
__author__ = 'Paul Landes'

from typing import List, Dict, Type
import torch
import numpy as np


class TorchTypes(object):
    """A utility class to convert betwen numpy and torch classes.  It also
    provides metadata for types that make other conversions, such as same
    precision cross types (i.e. int64 -> float64).

    """
    TYPES = [{'desc': '32-bit floating point',
              'name': 'float32',
              'types': set([torch.float32, torch.float]),
              'numpy': np.float32,
              'sparse': torch.sparse.FloatTensor,
              'cpu': torch.FloatTensor,
              'gpu': torch.cuda.FloatTensor},
             {'desc': '64-bit floating point',
              'name': 'float64',
              'types': set([torch.float64, torch.double]),
              'numpy': np.float64,
              'sparse': torch.sparse.DoubleTensor,
              'cpu': torch.DoubleTensor,
              'gpu': torch.cuda.DoubleTensor},
             {'desc': '16-bit floating point',
              'name': 'float16',
              'types': set([torch.float16, torch.half]),
              'numpy': np.float16,
              'sparse': torch.sparse.HalfTensor,
              'cpu': torch.HalfTensor,
              'gpu': torch.cuda.HalfTensor},
             {'desc': '8-bit integer (unsigned)',
              'name': 'uint8',
              'types': set([torch.uint8]),
              'numpy': np.uint8,
              'sparse': torch.sparse.ByteTensor,
              'cpu': torch.ByteTensor,
              'gpu': torch.cuda.ByteTensor},
             {'desc': '8-bit integer (signed)',
              'name': 'int8',
              'types': set([torch.int8]),
              'numpy': np.int8,
              'sparse': torch.sparse.CharTensor,
              'cpu': torch.CharTensor,
              'gpu': torch.cuda.CharTensor},
             {'desc': '16-bit integer (signed)',
              'name': 'int16',
              'types': set([torch.int16, torch.short]),
              'numpy': np.int16,
              'sparse': torch.sparse.ShortTensor,
              'cpu': torch.ShortTensor,
              'gpu': torch.cuda.ShortTensor},
             {'desc': '32-bit integer (signed)',
              'name': 'int32',
              'types': set([torch.int32, torch.int]),
              'numpy': np.int32,
              'sparse': torch.sparse.IntTensor,
              'cpu': torch.IntTensor,
              'gpu': torch.cuda.IntTensor},
             {'desc': '64-bit integer (signed)',
              'name': 'int64',
              'types': set([torch.int64, torch.long]),
              'numpy': np.int64,
              'sparse': torch.sparse.LongTensor,
              'cpu': torch.LongTensor,
              'gpu': torch.cuda.LongTensor},
             {'desc': 'Boolean',
              'name': 'bool',
              'types': set([torch.bool]),
              'numpy': bool,
              'cpu': torch.BoolTensor,
              'gpu': torch.cuda.BoolTensor}]
    """A list of dicts containig conversions between types."""

    NAME_TO_TYPE = {t['name']: t for t in TYPES}
    """A map of type to metadata."""

    FLOAT_TO_INT = {torch.float16: torch.int16,
                    torch.float32: torch.int32,
                    torch.float64: torch.int64}

    INT_TO_FLOAT = {torch.int16: torch.float16,
                    torch.int32: torch.float32,
                    torch.int64: torch.float64}

    FLOAT_TYPES = frozenset(FLOAT_TO_INT.keys())

    INT_TYPES = frozenset(INT_TO_FLOAT.keys())

    @classmethod
    def all_types(self) -> List[dict]:
        return self.TYPES

    @classmethod
    def types(self) -> Dict[str, List[dict]]:
        if not hasattr(self, '_types'):
            types = {}
            for d in self.all_types():
                for t in d['types']:
                    types[t] = d
            self._types = types
        return self._types

    @classmethod
    def type_from_string(self, type_name: str) -> torch.dtype:
        types = self.NAME_TO_TYPE[type_name]['types']
        return next(iter(types))

    @classmethod
    def get_tensor_class(self, torch_type: torch.dtype, cpu_type: bool) -> Type:
        types = self.types()
        entry = types[torch_type]
        key = 'cpu' if cpu_type else 'gpu'
        return entry[key]

    @classmethod
    def get_sparse_class(self, torch_type: torch.dtype) -> Type:
        types = self.types()
        entry = types[torch_type]
        return entry['sparse']

    @classmethod
    def get_numpy_type(self, torch_type: torch.dtype) -> Type:
        types = self.types()
        entry = types[torch_type]
        return entry['numpy']

    @classmethod
    def float_to_int(self, torch_type: torch.dtype) -> Type:
        return self.FLOAT_TO_INT[torch_type]

    @classmethod
    def int_to_float(self, torch_type: torch.dtype) -> Type:
        return self.INT_TO_FLOAT[torch_type]

    @classmethod
    def is_float(self, torch_type: torch.dtype) -> bool:
        return torch_type in self.FLOAT_TYPES

    @classmethod
    def is_int(self, torch_type: Type) -> bool:
        return torch_type in self.INT_TYPES
