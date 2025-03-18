"""A class that persists results in various formats.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import Tuple, Iterable, Set, List, Dict, Any
from dataclasses import dataclass, field
import logging
import re
import pickle
import json
import shutil
from pathlib import Path
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
    _EXTENSIONS = frozenset('txt model png json'.split())

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

    def get_paths(self, excludes: Set[str] = frozenset()) -> Iterable[Path]:
        """Get all paths in the result as an iterable.

        :param excludes: the extensions to exclude from the returned paths

        """
        exts: Set[str] = set(self._EXTENSIONS) - excludes
        return map(lambda at: getattr(self, f'{at}_path'), exts)

    def clear(self) -> List[Path]:
        paths: List[Path] = list(self.get_paths())
        paths.append(self.result_path)
        for path in paths:
            if path == self.model_path and path.is_dir():
                shutil.rmtree(path)
            elif path.is_file():
                path.unlink()
        return paths

    def __str__(self) -> str:
        return self.name


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
    prefix: str = field()
    """The prefix to use when creating the basename's stem file portion.  This
    is used by :class:`..model.pack.ModelPacker` to index other models besides
    the model name from the application config.

    """
    def load(self, name: str) -> ArchivedResult:
        path: Path = self.stash.key_to_path(name)
        name, id, ext = self.manager.parse_file_name(path.name)
        params = dict(id=int(id), name=name, result_path=path)
        for ext in self.manager._EXTENSIONS:
            k = f'{ext}_path'
            params[k] = self.manager._get_next_path(
                ext=ext, key=id, prefix=self.prefix)
        return ArchivedResult(**params)

    def exists(self, name: str) -> bool:
        return self.stash.exists(name)

    def keys(self) -> Iterable[str]:
        return self.stash.keys()

    def clear(self):
        arch: ArchivedResult
        for arch in self.values():
            arch.clear()


@dataclass
class ModelResultManager(IncrementKeyDirectoryStash):
    """Saves and loads results from runs (:class:`.ModelResult`) of the
    :class:`~zensols.deeplearn.model.executor.ModelExecutor`.  Keys incrementing
    integers, one for each save, which usually corresponds to the run of the
    model executor.

    The stash's :obj:`path` points to where results are persisted with all file
    format versions.

    """
    _EXTENSIONS = ArchivedResult._EXTENSIONS

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
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"model result manager using prefix: '{self.prefix}'")
        super().__post_init__(self.prefix)

    def create_results_stash(self, prefix: str = None) -> Stash:
        """Return a stash that provides access to previous results (not just the
        last results).  The stash iterates over the model results directory with
        :class:`.ArchivedResult` values.

        :param prefix: the prefix to use when creating the basename's stem file
                       portion; if ``None`` use a file name version of
                       :obj:`name`

        """
        return _ArchivedResultStash(
            self, DirectoryStash(path=self.path), prefix)

    @property
    @persisted('_results_stash')
    def results_stash(self) -> Stash:
        """The canonical results stash for the application configured prefix.

        :see: :meth:`create_results_stash`

        """
        return self.create_results_stash()

    @staticmethod
    def to_file_name(name: str) -> str:
        """Return a file name string from human readable ``name``."""
        return ModelSettings.normalize_name(name)

    def parse_file_name(self, res_id: str, raise_ex: bool = True) -> \
            Tuple[str, str, str]:
        m: re.Match = self.file_regex.match(res_id)
        if m is None and raise_ex:
            raise ModelError(f'Unknown model results id: {res_id}')
        if m is not None:
            return m.groups()

    def _get_next_path(self, ext: str, key: str = None,
                       prefix: str = None) -> Path:
        if key is None:
            key = self.get_last_key(False)
        prefix = self.prefix if prefix is None else prefix
        params = {'prefix': prefix, 'key': key, 'ext': ext}
        fname = self.file_pattern.format(**params)
        path = self.path / fname
        return path

    def get_last_id(self) -> str:
        """Get the last result ID."""
        key: str = self.get_last_key(False)
        return self.key_to_path(key).stem

    def get_next_text_path(self) -> Path:
        """Return a path to the available text file to be written."""
        return self._get_next_path('txt')

    def get_next_model_path(self) -> Path:
        """Return a path to the available model file to be written."""
        return self._get_next_path('model')

    def get_next_graph_path(self) -> Path:
        """Return a path to the available graph file to be written."""
        return self._get_next_path('png')

    def get_next_json_path(self) -> Path:
        """Return a path to the available JSON file to be written."""
        return self._get_next_path('json')

    def dump(self, result: ModelResult):
        # save the results as the ``.dat`` file
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
        """Return an instance of a model grapher.  This class can plot results
        of ``res`` using ``matplotlib``.

        :see: :class:`.ModelResultGrapher`

        """
        title = self.name if title is None else title
        path = self.get_next_graph_path()
        return ModelResultGrapher(title, figsize, save_path=path)

    def save_plot_result(self, result: ModelResult):
        """Plot and save results of the validation and training loss.

        """
        from tkinter import TclError
        try:
            grapher = self.get_grapher()
            grapher.plot_loss([result])
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
        res: Dict[str, Any] = result.asflatdict()
        assert 'configuration' not in res
        res['configuration'] = result.config.asdict()
        with open(path, 'w') as f:
            json.dump(res, f, indent=4)
