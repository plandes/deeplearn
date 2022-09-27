"""Model packaging and distribution.

"""
__author__ = 'Paul Landes'

from typing import Optional, ClassVar
from dataclasses import dataclass, field
import logging
from pathlib import Path
from zipfile import ZipFile
from zensols.persist import Stash
from zensols.install import Installer
from ..result import ArchivedResult
from . import ModelError, ModelResultManager, ModelExecutor

logger = logging.getLogger(__name__)


@dataclass
class ModelPacker(object):
    """Creates distribution model packages by creating a zip file of everything
    needed to by a client to use the model.

    """
    _PT_MODEL_DIR: ClassVar[str] = 'ptmodel'

    executor: ModelExecutor = field()
    """The result manager used to obtain the results and model to package."""

    version: str = field()
    """The version used to encode the package."""

    installer: Optional[Installer] = field(default=None)
    """If set, used to create a path to the model file."""

    def pack(self, res_id: str, output_dir: Path) -> Path:
        """Create a distribution model package on the file system.

        :param res_id: the result ID or use the last if not given (if optional)

        :return: the path to the generated distribution model package

        """
        verpath: str = 'v' + self.version.replace('.', '_')
        result_manager: ModelResultManager = self.executor.result_manager
        res_stash: Stash = result_manager.results_stash
        result: ArchivedResult = res_stash.get(res_id)
        if result is None:
            raise ModelError(f'No such result ID: {res_id}')
        output_file: Path = output_dir / f'{result.name}-{verpath}.zip'
        arch_suffix: str = 'model'
        arch_prefix: str = f'{result.name}-{verpath}'
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'packing {res_id}')
            result.write_to_log(logger, depth=1)
        with ZipFile(output_file, 'w') as zf:
            if result is None:
                raise ModelError(f'No such result: {res_id}')
            else:
                for path in result.get_paths():
                    arch_name: str = f'{arch_prefix}/{arch_suffix}{path.suffix}'
                    if logger.isEnabledFor(logging.INFO):
                        logger.info(f'adding file: {path} -> {arch_name}')
                    if path.is_dir():
                        for subpath in path.iterdir():
                            m_prefix = f'{self._PT_MODEL_DIR}/{subpath.name}'
                            zf.write(subpath, f'{arch_prefix}/{m_prefix}')
                    else:
                        zf.write(path, arch_name)
            zf.writestr(f'{arch_prefix}/{arch_suffix}.version', self.version)
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'wrote: {output_file}')
        return output_file

    @property
    def installed_model_path(self) -> Path:
        """Return the path to the model to be PyTorch loaded."""
        if self.installer is not None:
            res_path: Path = self.installer.get_singleton_path()
            path: Path = res_path / self._PT_MODEL_DIR
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'loading model {path}')
            return path

    def install_model(self) -> Path:
        """Install the model if it isn't already and return a path to it."""
        model_path: Path = self.installed_model_path
        if model_path is not None:
            self.installer.install()
        return model_path
