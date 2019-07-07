"""CUDA access and utility module.

"""
__author__ = 'Paul Landes'

import logging
import torch
import sys
from zensols.actioncli import persisted

logger = logging.getLogger(__name__)


class CudaInfo(object):
    """A utility class that provides information about the CUDA configuration for
    the current (hardware) environment.

    """

    def num_devices(self):
        """Return number of devices connected.

        """
        import pycuda.driver as cuda
        return cuda.Device.count()

    def devices(self):
        """Get info on all devices connected.

        """
        import pycuda.driver as cuda
        num = cuda.Device.count()
        print('%d device(s) found:' % num)
        for i in range(num):
            print(f'{cuda.Device(i).name()} (Id: {i})')

    def mem_info(self):
        """Get available and total memory of all devices.

        """
        import pycuda.driver as cuda
        available, total = cuda.mem_get_info()
        print('Available: {available/1e9:.2f} GB\nTotal:     {total/1e9:.2f} GB')

    def attributes(self, device_id=0):
        """Get attributes of device with device Id = device_id

        """
        import pycuda.driver as cuda
        return cuda.Device(device_id).get_attributes()

    def write_attributes(self, writer=sys.stdout):
        for att in self.attributes():
            writer.write(f'{att}\n')

    def write(self, writer=sys.stdout):
        """Class representation as number of devices connected and about them.

        """
        import pycuda.driver as cuda
        num = cuda.Device.count()
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
    def __init__(self, use_cuda=True, data_type=torch.float32):
        self.use_cuda = use_cuda
        self.data_type = data_type

    @persisted('__init_device', cache_global=True)
    def _init_device(self):
        use_cuda = self.use_cuda and torch.cuda.is_available()
        if use_cuda:
            import pycuda.driver as cuda
            import pycuda.autoinit
            cuda.init()
            cuda_dev = torch.cuda.current_device()
            torch.cuda.set_device(cuda_dev)
            device = f'cuda:{cuda_dev}'
        else:
            device = 'cpu'
        device = torch.device(device)
        logger.info(f'using device: {device}')
        return device

    @property
    def device(self):
        if not hasattr(self, '_device'):
            self.set_default_device()
        return self._device

    @device.setter
    def device(self, device):
        self._device = device

    def set_default_device(self):
        self._device = self._init_device()
        return self._device

    def empty_cache(self):
        torch.cuda.empty_cache()

    @property
    def info(self):
        self._init_device()
        return CudaInfo()

    def to(self, tensor_or_model):
        device = self.device
        if not hasattr(tensor_or_model, 'device') or \
           tensor_or_model.device != device:
            tensor_or_model = tensor_or_model.to(self.device)
        return tensor_or_model

    def _populate_defaults(self, kwargs):
        if 'dtype' not in kwargs:
            kwargs['dtype'] = self.data_type
        kwargs['device'] = self.device

    def singleton(self, *args, **kwargs):
        self._populate_defaults(kwargs)
        return torch.tensor(*args, **kwargs)

    def empty(self, *args, **kwargs):
        self._populate_defaults(kwargs)
        return torch.empty(*args, **kwargs)

    def zeros(self, *args, **kwargs):
        self._populate_defaults(kwargs)
        return torch.zeros(*args, **kwargs)

    def from_numpy(self, *args, **kwargs):
        return self.to(torch.from_numpy(*args, **kwargs))

    def cat(self, *args, **kwargs):
        return self.to(torch.cat(*args, **kwargs))

    def write(self, writer=sys.stdout):
        self.info.write(writer)
