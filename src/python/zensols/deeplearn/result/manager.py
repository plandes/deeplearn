from __future__ import annotations
"""A class that persists results in various formats.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Iterable
from dataclasses import dataclass, field
import logging
import re
import pickle
import shutil
from pathlib import Path
from tkinter import TclError
from zensols.persist import (
    persisted,
    DirectoryStash, Stash, ReadOnlyStash, IncrementKeyDirectoryStash,
)
from zensols.config import Dictable
from .. import ModelError, ModelSettings
from . import ModelResult, ModelResultGrapher

logger = logging.getLogger(__name__)


@dataclass
class ArchivedResult(Dictable):
    """An archived result that provides access to the outcomes the training,
    validation and optionally test phases of a model execution

    :see: :class:`.ModelResultManager`

    """
    _DICTABLE_ATTRIBUTES = {'model_result'}
    _DICTABLE_WRITE_EXCLUDES = _DICTABLE_ATTRIBUTES

    id: int = field()
    """The result incremented identitifer."""

    name: str = field()
    """The result's unique name, which includes :obj:`id`."""

    txt_path: Path = field()
    """The path results as a text file."""

    result_path: Path = field()
    """The path to pickled results file."""

    model_path: Path = field()
    """The path to the directory with the PyTorch model and state files."""

    png_path: Path = field()
    """The path to the training/validation loss results."""

    json_path: Path = field()
    """The path to the results as a parsable JSON file."""

    @property
    @persisted('_result')
    def model_result(self) -> ModelResult:
        """The results container of the run."""
        with open(self.result_path, 'rb') as f:
            return pickle.load(f)


@dataclass
class _ArchivedResultStash(ReadOnlyStash):
    """Creates instances of :class:`.ArchivedResult` using a delegate
    :class:`~zensols.persist.stash.DirectoryStash` for getting path values.

    """
    manager: ModelResultManager = field()
    """The manager containing the results."""

    stash: DirectoryStash = field()
    """The stash that reads the results persisted by
    :class:`.ModelResultManager`.

    """
    def load(self, name: str) -> ArchivedResult:
        path: Path = self.stash.key_to_path(name)
        m: re.Match = self.manager.file_regex.match(path.name)
        if m is None:
            raise ModelError(f'Unknown model results name: {name}')
        name, id, ext = m.groups()
        params = dict(id=int(id), name=name, result_path=path)
        for ext in self.manager._EXTENSIONS:
            k = f'{ext}_path'
            params[k] = self.manager._get_next_path(ext=ext, key=id)
        return ArchivedResult(**params)

    def exists(self, name: str) -> bool:
        return self.stash.exists(name)

    def keys(self) -> Iterable[str]:
        return self.stash.keys()


@dataclass
class ModelResultManager(IncrementKeyDirectoryStash):
    """Saves and loads results from runs (:class:`.ModelResult`) of the
    :class:`zensols.deeplearn.model.executor.ModelExecutor`.  Keys incrementing
    integers, one for each save, which usually corresponds to the run of the
    model executor.

    The stash's :obj:`path` points to where results are persisted with all file
    format versions.

    """
    _EXTENSIONS = 'txt model png json'.split()

    name: str = field(default=None)
    """The name of the manager in the configuration."""

    model_path: Path = field(default=True)
    """The path to where the results are stored."""

    save_text: bool = field(default=True)
    """If ``True`` save the results as a text file."""

    save_plot: bool = field(default=True)
    """If ``True`` save the plot to the file system."""

    save_json: bool = field(default=True)
    """If ``True`` save the results as a JSON file."""

    file_pattern: str = field(default='{prefix}-{key}.{ext}')
    """The pattern used to store the model and results files."""

    file_regex: re.Pattern = field(
        default=re.compile(r'^(.+)-(.+?)\.([^.]+)$'))
    """An regular expression analogue to :obj:`file_pattern`."""

    def __post_init__(self):
        self.prefix = self.to_file_name(self.name)
        super().__post_init__(self.prefix)

    @property
    @persisted('_read_stash')
    def results_stash(self) -> Stash:
        """Return a stash that provides access to previous results (not just the last
        results).  The stash iterates over the model results directory with
        :class:`.ArchivedResult` values.

        """
        return _ArchivedResultStash(self, DirectoryStash(path=self.path))

    @staticmethod
    def to_file_name(name: str) -> str:
        return ModelSettings.normalize_name(name)

    def _get_next_path(self, ext: str, key: str = None) -> Path:
        if key is None:
            key = self.get_last_key(False)
        params = {'prefix': self.prefix, 'key': key, 'ext': ext}
        fname = self.file_pattern.format(**params)
        path = self.path / fname
        return path

    def get_next_text_path(self) -> Path:
        return self._get_next_path('txt')

    def get_next_model_path(self) -> Path:
        return self._get_next_path('model')

    def get_next_graph_path(self) -> Path:
        return self._get_next_path('png')

    def get_next_json_path(self) -> Path:
        return self._get_next_path('json')

    def dump(self, result: ModelResult):
        super().dump(result)
        if self.model_path is not None:
            src = self.model_path
            dst = self.get_next_model_path()
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'copying model {src} -> {dst}')
            if dst.exists():
                logger.warning(f'already exists--deleting: {dst}')
                shutil.rmtree(dst)
            if not src.is_dir():
                raise ModelError(
                    f'No such directory: {src}--' +
                    'possibly because the model never learned')
            shutil.copytree(src, dst)
        if self.save_text:
            self.save_text_result(result)
        if self.save_json:
            self.save_json_result(result)
        if self.save_plot:
            self.save_plot_result(result)

    def get_grapher(self, figsize: Tuple[int, int] = (15, 5),
                    title: str = None) -> ModelResultGrapher:
        """Return an instance of a model grapher.  This class can plot results of
        ``res`` using ``matplotlib``.

        :see: :class:`.ModelResultGrapher`

        """
        title = self.name if title is None else title
        path = self.get_next_graph_path()
        return ModelResultGrapher(title, figsize, save_path=path)

    def save_plot_result(self, result: ModelResult):
        """Plot and save results of the validation and training loss.

        """
        try:
            grapher = self.get_grapher()
            grapher.plot([result])
            grapher.save()
        except TclError as e:
            # _tkinter.TclError: couldn't connect to display <IP>
            logger.warning('could not render plot, probably because ' +
                           f'disconnected from display: {e}')

    def save_text_result(self, result: ModelResult):
        """Save the text results of the model.

        """
        path = self.get_next_text_path()
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'saving text results to {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            result.write(writer=f, include_settings=True,
                         include_config=True, include_converged=True)

    def save_json_result(self, result: ModelResult):
        """Save the results of the model in JSON format.

        """
        path = self.get_next_json_path()
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'saving json results to {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            result.asjson(writer=f, indent=4)
