"""CUDA access and utility module.

"""
__author__ = 'Paul Landes'

import logging
import torch
import sys
from zensols.persist import persisted

logger = logging.getLogger(__name__)


class CudaInfo(object):
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

    def write(self, writer=sys.stdout):
        """Class representation as number of devices connected and about them.

        :see: cuda

        """
        import pycuda.driver as cuda
        num = self.num_devices
        writer.write(f'{num} device(s) found:\n')
        for i in range(num):
            writer.write(f'    {i+1}) {cuda.Device(i).name()} (Id: {i})\n' +
                         f'{" " * 10}Memory: ' +
                         f'{cuda.Device(i).total_memory()/1e9:.2f} GB\n')


class CudaConfig(object):
    """A utility class that provides access to CUDA APIs.  It provides information
    on the current CUDA configuration and convenience methods to create, copy
    and modify tensors.  These are handy for any given CUDA configuration and
    can back off to the CPU when CUDA isn't available.

    """
    CPU_DEVICE = 'cpu'

    def __init__(self, use_gpu=True, data_type=torch.float32):
        """Initialize this configuration.

        :param use_gpu: whether or not to use CUDA/GPU
        :param data_type: the data type to use when creating new tensors in
                          this configuration

        """
        logger.debug(f'use_gpu: {use_gpu}')
        self.use_gpu = use_gpu
        self.data_type = data_type

    @persisted('__init_device', cache_global=True)
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
            torch.cuda.set_device(cuda_dev)
            device = f'cuda:{cuda_dev}'
        else:
            device = self.CPU_DEVICE
        device = torch.device(device)
        logger.info(f'using device: {device}')
        return device

    @property
    @persisted('_cpu_device', cache_global=True)
    def cpu_device(self) -> torch.device:
        """Return the CUDA device (or CPU if CUDA is not available).

        """
        return torch.device(self.CPU_DEVICE)

    @property
    def device(self) -> torch.device:
        """Return the torch device configured.

        """
        if not hasattr(self, '_device'):
            if self.use_gpu:
                self._device = self._init_device()
            else:
                self._device = self.cpu_device
        return self._device

    @device.setter
    def device(self, device: torch.device):
        """Set (force) the device to be used in this configuration.

        """
        self._device = device

    def empty_cache(self):
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
    def gpu_available(self) -> bool:
        """Return whether or not CUDA GPU access is available.

        """
        return self._init_device().type != self.CPU_DEVICE

    def same_device(self, tensor_or_model) -> bool:
        """Return whether or not a tensor or model is in the same memory space as this
        configuration instance.

        """
        device = self.device
        return hasattr(tensor_or_model, 'device') and \
            tensor_or_model.device == device

    def to(self, tensor_or_model):
        """Copy the tensor or model to the device this to that of this configuration.

        """
        if not self.same_device(tensor_or_model):
            tensor_or_model = tensor_or_model.to(self.device)
        return tensor_or_model

    def _populate_defaults(self, kwargs):
        """Add keyword arguments to typical torch tensor creation functions.

        """
        if 'dtype' not in kwargs:
            kwargs['dtype'] = self.data_type
        kwargs['device'] = self.device

    def singleton(self, *args, **kwargs):
        """Return a new tensor using ``torch.tensor``.

        """
        self._populate_defaults(kwargs)
        return torch.tensor(*args, **kwargs)

    def empty(self, *args, **kwargs):
        """Return a new tesor using ``torch.empty``.

        """
        self._populate_defaults(kwargs)
        return torch.empty(*args, **kwargs)

    def zeros(self, *args, **kwargs):
        """Return a new tensor of zeros using ``torch.zeros``.

        """
        self._populate_defaults(kwargs)
        return torch.zeros(*args, **kwargs)

    def from_numpy(self, *args, **kwargs):
        """Return a new tensor generated from a numpy aray using ``torch.from_numpy``.

        """
        return self.to(torch.from_numpy(*args, **kwargs))

    def cat(self, *args, **kwargs):
        """Concatenate tensors in to one tensor using ``torch.cat``.

        """
        return self.to(torch.cat(*args, **kwargs))

    def write(self, writer=sys.stdout):
        if self.gpu_available:
            self.info.write(writer)
        else:
            writer.write('CUDA is not available')

    def __str__(self):
        return f'use cuda: {self.use_gpu}, device: {self.device}'

    def __repr__(self):
        return self.__str__()
