.. Licensed to the Apache Software Foundation (ASF) under one
   or more contributor license agreements.  See the NOTICE file
   distributed with this work for additional information
   regarding copyright ownership.  The ASF licenses this file
   to you under the Apache License, Version 2.0 (the
   "License"); you may not use this file except in compliance
   with the License.  You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing,
   software distributed under the License is distributed on an
   "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
   KIND, either express or implied.  See the License for the
   specific language governing permissions and limitations
   under the License.


Device
=======


The Device abstract represents any hardware device with memory and compuation units.
All [Tensor operations](tensor.html) are scheduled by the resident device for execution.
Tensor memory is also managed by the device's memory manager. Therefore, optimization
of memory and execution are implemented in the Device class.

Specific devices
----------------
Currently, SINGA has three Device implmentations,

1. CudaGPU for an Nvidia GPU card which runs Cuda code
2. CppCPU for a CPU which runs Cpp code
3. OpenclGPU for a GPU card which runs OpenCL code


Python API
----------

.. automodule:: singa.device
   :members: create_cuda_gpus, create_cuda_gpus_on, get_default_device


The following code provides examples of creating devices::

   from singa import device
   cuda = device.create_cuda_gpu_on(0)  # use GPU card of ID 0
   host = device.get_default_device()  # get the default host device (a CppCPU)
   ary1 = device.create_cuda_gpus(2)  # create 2 devices, starting from ID 0
   ary2 = device.create_cuda_gpus([0,2])  # create 2 devices on ID 0 and 2


CPP API
---------
