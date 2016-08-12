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

There are two set of tensor functions,
1. Tensor member functions, which would change the internal state of the Tensor instance.
2. tensor module functions, which accepts Tensor instances as arguments and return
Tensor instances.


Create Tensor instances
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: singa.tensor.Tensor


Tensor instances can be constructed from Numpy array,

.. automodule:: singa.tensor
   :members: from_numpy


Set Tensor values
~~~~~~~~~~~~~~~~~











