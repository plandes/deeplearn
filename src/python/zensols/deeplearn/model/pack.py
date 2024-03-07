"""Model packaging and distribution.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Dict, Any, Optional, Type, Union, ClassVar
from dataclasses import dataclass, field
from abc import ABCMeta
import sys
import logging
from pathlib import Path
import json
from io import StringIO, TextIOBase
from zipfile import ZipFile
from zensols.persist import persisted, PersistableContainer, Stash
from zensols.config import Writable
from zensols.introspect import ClassImporter
from zensols.install import Installer
from zensols.config import (
    ConfigurableError, Configurable, DictionaryConfig, IniConfig, ConfigFactory,
)
from ..result import ArchivedResult
from . import ModelError, ModelResultManager, ModelExecutor, ModelFacade

logger = logging.getLogger(__name__)


@dataclass
class _PackerBase(object, metaclass=ABCMeta):
    """A base class for packing and unpacking models.

    """
    _PT_MODEL_DIR: ClassVar[str] = 'ptmodel'
    """Model directory name."""

    _ARCHIVE_SUFFIX: ClassVar[str] = 'model'
    """The file stem of the model's packed files. """

    version: str = field()
    """The version used to encode the package."""


@dataclass
class ModelPacker(_PackerBase):
    """Creates distribution model packages by creating a zip file of everything
    needed to by a client to use the model.

    """
    executor: ModelExecutor = field()
    """The result manager used to obtain the results and model to package."""

    installer: Optional[Installer] = field(default=None)
    """If set, used to create a path to the model file."""

    def _to_ini(self, config: Configurable) -> str:
        ini = IniConfig()
        sio = StringIO()
        config.copy_sections(ini)
        config.parser.write(sio)
        return sio.getvalue()

    def pack(self, res_id: str, output_dir: Path) -> Path:
        """Create a distribution model package on the file system.

        :param res_id: the result ID or use the last if not given (if optional)

        :return: the path to the generated distribution model package

        """
        verpath: str = 'v' + self.version.replace('.', '_')
        result_manager: ModelResultManager = self.executor.result_manager
        res_prefix: str = result_manager.parse_file_name(f'{res_id}._')[0]
        res_stash: Stash = result_manager.create_results_stash(res_prefix)
        result: ArchivedResult = res_stash.get(res_id)
        if result is None:
            raise ModelError(f'No such result ID: {res_id}')
        output_file: Path = output_dir / f'{result.name}-{verpath}.zip'
        arch_suffix: str = self._ARCHIVE_SUFFIX
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


@dataclass
class ModelUnpacker(_PackerBase, PersistableContainer, Writable):
    """Unpacks a model created by :class:`.ModelPacker`.  Intances of this class
    should be deallocated with :meth:`~zensols.persist.dealloc.Deallocatable`
    after using it, which in turn deallocates the :obj:`facade`.

    """
    config_factory: ConfigFactory = field()
    """Used to get the model facade class when :obj:`facade_source` is a
    string.

    """
    installer: Installer = field()
    """Used to create a path to the model file."""

    model_packer_name: str = field(default='deeplearn_model_packer')
    """The section of the :class:`.ModelPacker` used to create the model.  This
    defaults to the section in resource library ``resources/cli-pack.conf``.

    """
    facade_source: Union[str, Type[ModelFacade]] = field(default='facade')
    """The client facade section name or the :class:`.ModelFacade."""

    model_config_overwrites: Configurable = field(default=None)
    """Configuration that overwrites the packaged model configuration."""

    def __post_init__(self):
        # use PersistableContainer for deallocation
        PersistableContainer.__init__(self)

    @property
    def installed_model_path(self) -> Path:
        """Return the path to the model to be PyTorch loaded."""
        res_path: Path = self.installer.get_singleton_path()
        path: Path = res_path / self._PT_MODEL_DIR
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'loading model {path}')
        return path

    @persisted('_installed_model')
    def install_model(self) -> Path:
        """Install the model if it isn't already and return a path to it."""
        model_path: Path = self.installed_model_path
        self.installer.install()
        return model_path

    @property
    @persisted('_installed_version')
    def installed_version(self) -> str:
        """The version of the model installed on the file system per the
        ``model.version`` file.

        """
        res_path: Path = self.installer.get_singleton_path()
        version_file: Path = res_path / f'{self._ARCHIVE_SUFFIX}.version'
        return version_file.read_text(encoding='utf-8').strip()

    @persisted('__validate_version', transient=True)
    def _validate_version(self, facade: 'ModelFacade') -> bool:
        packer_version: str = facade.config_factory.config.get_option(
            'version', self.model_packer_name)
        if packer_version != self.version:
            model_name: str = facade.model_settings.model_name
            logger.warning(
                f'API {model_name} version ({self.version}) does not ' +
                f'match the trained model version ({packer_version})')
            return False
        if packer_version != self.installed_version:
            logger.warning(
                f'API {model_name} installed file version ' +
                f'({self.installed_version}) does not ' +
                f'match the trained model version ({packer_version})')
            return False
        return True

    def _get_model_config(self) -> Configurable:
        config: Configurable = None
        res_path: Path = self.installer.get_singleton_path()
        path: Path = res_path / f'{self._ARCHIVE_SUFFIX}.json'
        if path.is_file():
            with open(path) as f:
                data: Dict[str, Any] = json.load(f)
            if 'configuration' in data:
                config = DictionaryConfig(data['configuration'])
        if config is None:
            config = self.config_factory.config
        return config

    def _get_facade_class(self) -> Type[ModelFacade]:
        if isinstance(self.facade_source, str):
            config: Configurable = self._get_model_config()
            class_name: str = config.get_option(
                'class_name', self.facade_source)
            return ClassImporter(class_name).get_class()
        else:
            return self.facade_source

    @property
    @persisted('_facade')
    def facade(self) -> ModelFacade:
        """The cached facade from installed model.  This installs the model if
        isn't already.

        :return: a model facade that allows the caller to deallocate

        """
        model_path: Path = self.install_model()
        facade_cls: Type[ModelFacade] = self._get_facade_class()
        facade = facade_cls.load_from_path(
            path=model_path,
            model_config_overwrites=self.model_config_overwrites)
        self._validate_version(facade)
        return facade

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        facade: ModelFacade = self.facade
        self._write_line(f'Version: {self.installed_version}', depth, writer)
        self._write_line(f'Source: {self.installer.resources[0].url}',
                         depth, writer)
        self._write_line(f'Installed: {self.installed_model_path}',
                         depth, writer)
        self._write_block(facade.executor.model_result_report, depth, writer)


class SubsetConfig(DictionaryConfig):
    """A :class:`~zensols.config.configbase.Configurable` that takes a subset of
    the application configuration.  This is useful to pass to
    :meth:`.ModelFacade.load_from_path` to merge application into the packed
    model's configuration.

    """
    def __init__(self, config_factory: ConfigFactory, sections: Tuple[str, ...],
                 options: Tuple[str, ...], option_delim: str = ':'):
        """Initialize the instance.

        :param config_factory: the application config and factory

        :param sections: a list of sections to subset

        :param options: a list of ``<section>:<option>``, each of which is added
                        to the subset

        :param option_delim: the string used to delimit sections and options in
                             ``options``

        """
        super().__init__()
        src: Configurable = config_factory.config
        src.copy_sections(self, sections=sections)
        option: str
        for option in options:
            sec_name: Tuple[str, str] = option.split(option_delim)
            if len(sec_name) != 2:
                raise ConfigurableError('Wrong format: expecting delim ' +
                                        f'{option_delim} but got: {option}')
            sec, name = sec_name
            val: str = src.get_option(name, sec)
            self.set_option(name, val, sec)
