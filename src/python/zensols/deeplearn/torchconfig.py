"""CUDA access and utility module.

"""
__author__ = 'Paul Landes'

from typing import Dict, Iterable, Any, Tuple, Union, Type, List
import sys
import logging
import gc
from io import TextIOBase
import random
import torch
from torch import Tensor
from torch import nn
from torch.multiprocessing import set_start_method
import numpy as np
from zensols.config import Writable
from zensols.persist import persisted, PersistableContainer, PersistedWork
from . import TorchTypes

logger = logging.getLogger(__name__)


class CudaInfo(Writable):
    """A utility class that provides information about the CUDA configuration for
    the current (hardware) environment.

    """

    @property
    def num_devices(self) -> int:
        """Return number of devices connected.

        """
        import pycuda.driver as cuda
        return cuda.Device.count()

    @property
    def attributes(self, device_id=0) -> dict:
        """Get attributes of device with device Id = device_id

        """
        import pycuda.driver as cuda
        return cuda.Device(device_id).get_attributes()

    def write_attributes(self, writer=sys.stdout):
        """Write the GPU attributes.

        :see: attributes

        """
        for k, v in self.attributes.items():
            writer.write(f'{k} -> {v}\n')

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        """Class representation as number of devices connected and about them.

        :see: cuda

        """
        import pycuda.driver as cuda
        num = self.num_devices
        self._write_line(f'{num} device(s) found:', depth, writer)
        for i in range(num):
            name = f'{i+1}) {cuda.Device(i).name()} (Id: {i})'
            mem = f'memory: {cuda.Device(i).total_memory()/1e9:.2f} GB'
            self._write_line(name, depth + 1, writer)
            self._write_line(mem, depth + 2, writer)

    def __str__(self):
        return f'CUDA devices: {self.num_devices}'


class TorchConfig(PersistableContainer, Writable):
    """A utility class that provides access to CUDA APIs.  It provides information
    on the current CUDA configuration and convenience methods to create, copy
    and modify tensors.  These are handy for any given CUDA configuration and
    can back off to the CPU when CUDA isn't available.

    """
    CPU_DEVICE = 'cpu'
    RANDOM_SEED: dict = None

    def __init__(self, use_gpu: bool = True, data_type: type = torch.float32,
                 cuda_device_index: int = None):
        """Initialize this configuration.

        :param use_gpu: whether or not to use CUDA/GPU

        :param data_type: the default data type to use when creating new
                          tensors in this configuration

        :param cuda_device_index: the CUDA device to use, which defaults to 0
                                  if CUDA if ``use_gpu`` is ``True``

        """
        super().__init__()
        logger.debug(f'use_gpu: {use_gpu}')
        self.use_gpu = use_gpu
        self.data_type = data_type
        # we can't globally cache this in case there are multiple instances of
        # this class for which have different values of `use_gpu`
        self._init_device_pw = PersistedWork('_init_device_pw', self)
        self._cpu_device_pw = PersistedWork('_cpu_device_pw', self, cache_global=True)
        self._cpu_device_pw._mark_deallocated()
        self._cuda_device_index = cuda_device_index

    @persisted('_init_device_pw')
    def _init_device(self) -> torch.device:
        """Attempt to initialize CUDA, and if successful, return the CUDA device.

        """
        is_avail = torch.cuda.is_available()
        use_gpu = self.use_gpu and is_avail
        logger.debug(f'use cuda: {self.use_gpu}, is avail: {is_avail}')
        if use_gpu:
            import pycuda.driver as cuda
            import pycuda.autoinit
            logger.debug('trying to initialize CUDA')
            cuda.init()
            logger.info('successfully initialized CUDA')
            cuda_dev = torch.cuda.current_device()
            device = torch.device('cuda', cuda_dev)
            self.cuda_device_index = cuda_dev
        else:
            device = torch.device(self.CPU_DEVICE)
        return device

    @property
    @persisted('_cpu_device_pw')
    def cpu_device(self) -> torch.device:
        """Return the CPU CUDA device, which is the device type configured to utilize
        the CPU (rather than the GPU).

        """
        return torch.device(self.CPU_DEVICE)

    @property
    def device(self) -> torch.device:
        """Return the torch device configured.

        """
        if not hasattr(self, '_device'):
            if self.use_gpu:
                if self._cuda_device_index is None:
                    self._device = self._init_device()
                else:
                    self._device = torch.device(
                        'cuda', self._cuda_device_index)
            else:
                self._device = self.cpu_device
        return self._device

    @device.setter
    def device(self, device: torch.device):
        """Set (force) the device to be used in this configuration."""
        self._device = device
        torch.cuda.set_device(device)
        logger.info(f'using device: {device}')

    @property
    def using_cpu(self) -> bool:
        """Return ``True`` if this configuration is using the CPU device."""
        return self.device.type == self.CPU_DEVICE

    @classmethod
    def is_on_cpu(cls, arr: Tensor) -> bool:
        """Return ``True`` if the passed tensor is on the CPU."""
        return arr.device.type == cls.CPU_DEVICE

    @property
    def gpu_available(self) -> bool:
        """Return whether or not CUDA GPU access is available."""
        return self._init_device().type != self.CPU_DEVICE

    @property
    def cuda_devices(self) -> Tuple[torch.device]:
        """Return all cuda devices.

        """
        return tuple(map(lambda n: torch.device('cuda', n),
                         range(torch.cuda.device_count())))

    @property
    def cuda_device_index(self) -> Union[int, None]:
        """Return the CUDA device index if CUDA is being used for this configuration.
        Otherwise return ``None``.

        """
        device = self.device
        if device.type == 'cuda':
            return device.index

    @cuda_device_index.setter
    def cuda_device_index(self, device: int):
        """Set the CUDA device index for this configuration."""
        self.device = torch.device('cuda', device)

    def same_device(self, tensor_or_model) -> bool:
        """Return whether or not a tensor or model is in the same memory space as this
        configuration instance.

        """
        device = self.device
        return hasattr(tensor_or_model, 'device') and \
            tensor_or_model.device == device

    @staticmethod
    def in_memory_tensors() -> List[Tensor]:
        """Returns all in-memory tensors and parameters.

        :see: :meth:`~zensols.deeplearn.cli.app.show_leaks`

        """
        arrs: List[Tensor] = []
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or \
                   (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    arrs.append(obj)
            except Exception:
                pass
        return arrs

    @classmethod
    def write_in_memory_tensors(cls: Type, writer: TextIOBase = sys.stdout):
        """Prints in-memory tensors and parameters.

        :see: :class:`~zensols.deeplearn.torchconfig.TorchConfig`

        """
        objs: List[Tensor] = cls.in_memory_tensors()
        for obj in objs:
            writer.write(f'{type(obj)}: {tuple(obj.shape)} on {obj.device}\n')

    @staticmethod
    def empty_cache():
        """Empty the CUDA torch cache.  This releases memory in the GPU and should not
        be necessary to call for normal use cases.

        """
        torch.cuda.empty_cache()

    @property
    def info(self) -> CudaInfo:
        """Return the CUDA information, which include specs of the device.

        """
        self._init_device()
        return CudaInfo()

    @property
    def tensor_class(self) -> Type[torch.dtype]:
        """Return the class type based on the current configuration of this instance.
        For example, if using ``torch.float32`` on the GPU,
        ``torch.cuda.FloatTensor`` is returned.

        """
        return TorchTypes.get_tensor_class(self.data_type, self.using_cpu)

    @property
    def numpy_data_type(self) -> Type[torch.dtype]:
        """Return the numpy type that corresponds to this instance's configured
        ``data_type``.

        """
        return TorchTypes.get_numpy_type(self.data_type)

    def to(self, tensor_or_model: Union[nn.Module, Tensor]) -> \
            Union[nn.Module, Tensor]:
        """Copy the tensor or model to the device this to that of this configuration.

        """
        if not self.same_device(tensor_or_model):
            tensor_or_model = tensor_or_model.to(self.device)
        if isinstance(tensor_or_model, nn.Module) and \
           tensor_or_model != self.data_type:
            tensor_or_model.type(self.data_type)
        return tensor_or_model

    @classmethod
    def to_cpu_deallocate(cls, *arrs: Tuple[Tensor]) -> \
            Union[Tuple[Tensor], Tensor]:
        """Safely copy detached memory to the CPU and delete local instance (possibly
        GPU) memory to speed up resource deallocation.  If the tensor is
        already on the CPU, it's simply passed back.  Otherwise the tensor is
        deleted.

        This method is robust with ``None``, which are skipped and substituted
        as ``None`` in the output.

        :param arrs: the tensors the copy to the CPU (if not already)

        :return: the singleton tensor if only one ``arrs`` is passed;
                 otherwise, the CPU copied tensors from the input

        """
        cpus = []
        for arr in arrs:
            if arr is None or (cls.is_on_cpu(arr) and not arr.requires_grad):
                cpu_arr = arr
            else:
                cpu_arr = arr.detach().clone().cpu()
                # suggest to interpreter to mark for garbage collection
                # immediately
                del arr
            cpus.append(cpu_arr)
        return cpus[0] if len(cpus) == 1 else tuple(cpus)

    def clone(self, tensor: Tensor, requires_grad: bool = True) -> Tensor:
        """Clone a tensor.

        """
        return tensor.detach().clone().requires_grad_(requires_grad)

    def _populate_defaults(self, kwargs):
        """Add keyword arguments to typical torch tensor creation functions.

        """
        if 'dtype' not in kwargs:
            kwargs['dtype'] = self.data_type
        kwargs['device'] = self.device

    def from_iterable(self, array: Iterable[Any]) -> Tensor:
        """Return a one dimenstional tensor created from ``array`` using the type and
        device in the current instance configuration.

        """
        cls = self.tensor_class
        if not isinstance(array, tuple) and not isinstance(array, list):
            array = tuple(array)
        return cls(array)

    def singleton(self, *args, **kwargs) -> Tensor:
        """Return a new tensor using ``torch.tensor``.

        """
        self._populate_defaults(kwargs)
        return torch.tensor(*args, **kwargs)

    def float(self, *args, **kwargs) -> Tensor:
        """Return a new tensor using ``torch.tensor`` as a float type.

        """
        kwargs['dtype'] = self.float_type
        self._populate_defaults(kwargs)
        return torch.tensor(*args, **kwargs)

    def int(self, *args, **kwargs) -> Tensor:
        """Return a new tensor using ``torch.tensor`` as a int type.

        """
        kwargs['dtype'] = self.int_type
        self._populate_defaults(kwargs)
        return torch.tensor(*args, **kwargs)

    def sparse(self, indicies: Tuple[int], values: Tuple[float],
               shape: Tuple[int, int]):
        """Create a sparce tensor from indexes and values.

        """
        i = torch.LongTensor(indicies)
        v = torch.FloatTensor(values)
        cls = TorchTypes.get_sparse_class(self.data_type)
        return cls(i, v, shape, device=self.device)

    def is_sparse(self, arr: Tensor) -> bool:
        """Return whether or not a tensor a sparse.
        """
        return arr.layout == torch.sparse_coo

    def empty(self, *args, **kwargs) -> Tensor:
        """Return a new tesor using ``torch.empty``.

        """
        self._populate_defaults(kwargs)
        return torch.empty(*args, **kwargs)

    def zeros(self, *args, **kwargs) -> Tensor:
        """Return a new tensor of zeros using ``torch.zeros``.

        """
        self._populate_defaults(kwargs)
        return torch.zeros(*args, **kwargs)

    def ones(self, *args, **kwargs) -> Tensor:
        """Return a new tensor of zeros using ``torch.zeros``.

        """
        self._populate_defaults(kwargs)
        return torch.ones(*args, **kwargs)

    def from_numpy(self, arr: np.ndarray) -> Tensor:
        """Return a new tensor generated from a numpy aray using ``torch.from_numpy``.
        The array type is converted if necessary.

        """
        tarr = torch.from_numpy(arr)
        if arr.dtype != self.numpy_data_type:
            tarr = tarr.type(self.data_type)
        return self.to(tarr)

    def cat(self, *args, **kwargs) -> Tensor:
        """Concatenate tensors in to one tensor using ``torch.cat``.

        """
        return self.to(torch.cat(*args, **kwargs))

    def to_type(self, arr: Tensor) -> Tensor:
        """Convert the type of the given array to the type of this instance.

        """
        if self.data_type != arr.dtype:
            arr = arr.type(self.data_type)
        return arr

    @property
    def float_type(self) -> Type:
        """Return the float type that represents this configuration, converting to the
        corresponding precision from integer if necessary.

        :return: the float that represents this data, or ``None`` if neither
                 float nor int

        """
        dtype = self.data_type
        if TorchTypes.is_int(dtype):
            return TorchTypes.int_to_float(dtype)
        elif TorchTypes.is_float(dtype):
            return dtype

    @property
    def int_type(self) -> Type:
        """Return the int type that represents this configuration, converting to the
        corresponding precision from integer if necessary.

        :return: the int that represents this data, or ``None`` if neither
                 int nor float

        """
        dtype = self.data_type
        if TorchTypes.is_float(dtype):
            return TorchTypes.float_to_int(dtype)
        elif TorchTypes.is_int(dtype):
            return dtype

    @staticmethod
    def equal(a: Tensor, b: Tensor) -> bool:
        """Return whether or not two tensors are equal.  This does an exact cell
        comparison.

        """
        return torch.all(a.eq(b)).item()

    @staticmethod
    def close(a: Tensor, b: Tensor) -> bool:
        """Return whether or not two tensors are equal.  This does an exact cell
        comparison.

        """
        return torch.allclose(a, b)

    @classmethod
    def get_random_seed(cls: Type) -> int:
        """Get the cross system random seed, meaning the seed applied to CUDA and the
        Python *random* library.

        """
        if cls.RANDOM_SEED is not None:
            return cls.RANDOM_SEED['seed']

    @classmethod
    def get_random_seed_context(cls: Type) -> Dict[str, Any]:
        """Return the random seed context given to :py:meth:`set_random_seed` to
        restore across models for consistent results.

        """
        return cls.RANDOM_SEED

    @classmethod
    def set_random_seed(cls: Type, seed: int = 0, disable_cudnn: bool = True,
                        rng_state: bool = True):
        """Set the random number generator for PyTorch.


        :param seed: the random seed to be set

        :param disable_cudnn: if ``True`` disable NVidia's backend cuDNN
                              hardware acceleration, which might have
                              non-deterministic features

        :param rng_state: set the CUDA random state array to zeros

        :see: `Torch Random Seed <https://discuss.pytorch.org/t/random-seed-initialization/7854>`_

        :see: `Reproducibility <https://discuss.pytorch.org/t/non-reproducible-result-with-gpu/1831>`_

        """
        cls.RANDOM_SEED = {'seed': seed,
                           'disable_cudnn': disable_cudnn,
                           'rng_state': rng_state}

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            if rng_state:
                new_states = []
                for state in torch.cuda.get_rng_state_all():
                    zeros = torch.zeros(state.shape, dtype=state.dtype)
                    new_states.append(zeros)
                torch.cuda.set_rng_state_all(new_states)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(0)

        if disable_cudnn:
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    @classmethod
    def init(cls: Type, spawn_multiproc: str = 'spawn',
             seed_kwargs: Dict[str, Any] = {}):
        """Initialize the PyTorch framework.  This includes:

          * Configuration of PyTorch multiprocessing so subprocesses can access
            the GPU, and

          * Setting the random seed state.

        The needs to be initialized at the very beginning of your program.

        Example::
            def main():
                from zensols.deeplearn.init import TorchInitializer
                TorchInitializer.init()

        **Note**: this method is separate from :meth:`set_random_seed` because
        that method is called by the framework to reset the seed after a model
        is unpickled.

        :see: :mod:`torch.multiprocessing`

        :see: :meth:`set_random_seed`

        """
        if cls.RANDOM_SEED is None:
            cls.set_random_seed(**seed_kwargs)
            try:
                if logger.isEnabledFor(logging.INFO):
                    logger.info('invoking pool with torch spawn method')
                if spawn_multiproc:
                    set_start_method('spawn')
                else:
                    set_start_method('forkserver', force=True)
            except RuntimeError as e:
                logger.warning(f'could not invoke spawn on pool: {e}')

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        if self.gpu_available:
            self.info.write(depth, writer)
        else:
            self._write_line('CUDA is not available', depth, writer)
        self._write_line(f'selected device: {self.device}', depth, writer)

    def __str__(self):
        return f'use cuda: {self.use_gpu}, device: {self.device}'

    def __repr__(self):
        return self.__str__()


class printopts(object):
    """Object used with a ``with`` scope that sets options, then sets them back.

    Example::

        with printopts(profile='full', linewidth=120):
            print(tensor)

    :see: `PyTorch Documentation <https://pytorch.org/docs/master/generated/torch.set_printoptions.html>`_

    """
    DEFAULTS = {'precision': 4,
                'threshold': 1000,
                'edgeitems': 3,
                'linewidth': 80,
                'profile': 'default',
                'sci_mode': None}

    def __init__(self, **kwargs):
        if len(kwargs) > 0:
            torch.set_printoptions(**kwargs)

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        torch.set_printoptions(**self.DEFAULTS)
