"""A class that persists results in various formats.

"""
__author__ = 'Paul Landes'

from typing import Tuple
from dataclasses import dataclass, field
import logging
import re
import shutil
from pathlib import Path
from tkinter import TclError
from zensols.persist import IncrementKeyDirectoryStash
from .. import ModelError
from . import ModelResult, ModelResultGrapher

logger = logging.getLogger(__name__)


@dataclass
class ModelResultManager(IncrementKeyDirectoryStash):
    """Saves and loads results from runs (:class:`.ModelResult`) of the
    :class:`zensols.deeplearn.model.executor.ModelExecutor`.  Keys incrementing
    integers, one for each save, which usually corresponds to the run of the
    model executor.

    :param model_path: if not ``None`` the model persisted by
                       :class:`zensols.deeplearn.model.manager.ModelManager` is
                       saved to disk

    :param save_text: if ``True`` save the verbose result output (from
                      :meth:`.ModelResult.write`) of the results run

    :param save_plot: if ``True`` save the plot using :meth:`save_plot`

    """
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

    def __post_init__(self):
        self.prefix = self.to_file_name(self.name)
        super().__post_init__(self.prefix)

    @staticmethod
    def to_file_name(name: str) -> str:
        regex = r'[:()\[\]_ \t-]+'
        name = name.lower()
        name = re.sub(regex, '-', name)
        name = re.sub((regex + '$'), '', name)
        return name

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
