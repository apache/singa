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


Tensor
========

Each Tensor instance is a multi-dimensional array allocated on a specific
Device instance. Tensor instances store variables and provide
linear algebra operations over different types of hardware devices without user
awareness. Note that users need to make sure the tensor operands are
allocated on the same device except copy functions.


Tensor implementation
---------------------

SINGA has three different sets of implmentations of Tensor functions, one for each
type of Device.

* 'tensor_math_cpp.h' implements operations using Cpp (with CBLAS) for CppGPU devices.
* 'tensor_math_cuda.h' implements operations using Cuda (with cuBLAS) for CudaGPU devices.
* 'tensor_math_opencl.h' implements operations using OpenCL for OpenclGPU devices.

Python API
----------


.. automodule:: singa.tensor
   :members:


CPP API
---------
