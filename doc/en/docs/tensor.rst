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
