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
to call singa::Device and its methods
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


class Platform(object):
    @staticmethod
    def get_num_gpus():
        return singa.Platform.GetNumGPUs()

    @staticmethod
    def get_gpu_ids():
        return singa.Platform.GetGPUIDs()

    @staticmethod
    def get_gpu_mem_size(id):
        return singa.Platform.GetGPUMemSize(id)

    @staticmethod
    def device_query(id, verbose=False):
        return singa.Platform.DeviceQuery(id, verbose)

    @staticmethod
    def create_raw_cuda_gpus(num):
        return singa.Platform.CreateCudaGPUs(num)

    @staticmethod
    def create_cuda_gpu():
        return singa.Platform.CreateCudaGPUs(1)[0]
