"""Result diff utilities.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field
import sys
from io import TextIOBase
from zensols.config import Configurable, Writable, ConfigurableDiffer
from . import ModelResult, ModelResultManager


@dataclass
class ModelResultComparer(Writable):
    """This class performs a diff on two classes and reports the differences.

    """
    rm: ModelResultManager = field()
    """The manager used to retrieve the model results."""

    res_id_a: str = field()
    """The result ID of the first archived result set to diff."""

    res_id_b: str = field()
    """The result ID of the second archived result set to diff."""

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        res_a: ModelResult = self.rm.results_stash[self.res_id_a].model_result
        res_b: ModelResult = self.rm.results_stash[self.res_id_b].model_result
        self._write_line(f'{self.res_id_a}:', depth, writer)
        res_a.test.write(depth + 1, writer)
        self._write_line(f'{self.res_id_b}:', depth, writer)
        res_b.test.write(depth + 1, writer)
        self._write_line(f'{self.res_id_a} -> {self.res_id_b}:', depth, writer)
        conf_a: Configurable = res_a.config
        conf_b: Configurable = res_b.config
        diff = ConfigurableDiffer(conf_a, conf_b)
        diff.write(depth + 1, writer)
