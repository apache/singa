# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# =============================================================================
'''
This script includes Device class and its subclasses for python users
to call singa::Device and its methods.

TODO(wangwei) implement py CudaGPU class.
'''

from . import singa_wrap as singa


class Device(object):
    """ Class and member functions for singa::Device.

    Create Device instances using the CreateXXXDevice.
    """

    def __init__(self, id, device):
        """Device constructor given device ID.

        Args:
            id (int): device ID.
            device: swig shared_ptr<Device>
        """
        self.id = id
        self.singa_device = device

    def set_rand_seed(self, seed):
        self.singa_device.SetRandSeed(seed)

    def get_host(self):
        return self.singa_device.host()

    def get_id(self):
        return self.singa_device.id()


def get_num_gpus():
    assert singa.USE_CUDA, 'SINGA has not been compiled with CUDA enabled.'
    return singa.Platform.GetNumGPUs()


def get_gpu_ids():
    assert singa.USE_CUDA, 'SINGA has not been compiled with CUDA enabled.'
    return singa.Platform.GetGPUIDs()


def get_gpu_mem_size(id):
    assert singa.USE_CUDA, 'SINGA has not been compiled with CUDA enabled.'
    return singa.Platform.GetGPUMemSize(id)


def get_num_opencl_platforms():
    assert singa.USE_OPENCL, 'SINGA has not been compiled with OpenCL enabled.'
    return singa.Platform.GetNumOpenclPlatforms()


def get_num_opencl_devices():
    assert singa.USE_OPENCL, 'SINGA has not been compiled with OpenCL enabled.'
    return singa.Platform.GetNumOpenclDevices()


def device_query(id, verbose=False):
    assert singa.USE_CUDA, 'SINGA has not been compiled with CUDA enabled.'
    return singa.Platform.DeviceQuery(id, verbose)


def create_cuda_gpus(num):
    '''Create a list of CudaGPU devices.

    Args:
        num (int): number of device to create.
    Returns:
        a list of swig converted CudaGPU devices.
    '''
    assert singa.USE_CUDA, 'SINGA has not been compiled with CUDA enabled.'
    return singa.Platform.CreateCudaGPUs(num)


def create_cuda_gpu():
    '''Create a single CudaGPU device.

    Returns:
        a swig converted CudaGPU device.
    '''
    assert singa.USE_CUDA, 'SINGA has not been compiled with CUDA enabled.'
    return singa.Platform.CreateCudaGPUs(1)[0]


def create_cuda_gpus_on(device_ids):
    '''Create a list of CudaGPU devices.

    Args:
        device_ids (list): a list of GPU card IDs.

    Returns:
        a list of swig converted CudaGPU devices.
    '''
    assert singa.USE_CUDA, 'SINGA has not been compiled with CUDA enabled.'
    return singa.Platform.CreateCudaGPUsOn(device_ids)


def create_cuda_gpu_on(device_id):
    '''Create a CudaGPU device on the given device ID.

    Args:
        device_id (int): GPU card ID.

    Returns:
        a swig converted CudaGPU device.
    '''
    assert singa.USE_CUDA, 'SINGA has not been compiled with CUDA enabled.'
    devices = create_cuda_gpus_on([device_id])
    return devices[0]


def create_opencl_device():
    '''Create the default OpenCL device.

    Returns:
        a swig converted OpenCL device.
    '''
    assert singa.USE_OPENCL, 'SINGA has not been compiled with OpenCL enabled.'
    return singa.Platform.GetDefaultOpenclDevice()


default_device = singa.Platform.GetDefaultDevice()


def get_default_device():
    '''Get the default host device which is a CppCPU device'''
    return default_device
