Device
=======


The Device abstract represent a hardware device with memory and compuation units.

Specific devices
----------------
Currently, SINGA has three Device implmentations,

1. CudaGPU for an Nvidia GPU card which runs Cuda code
2. CppCPU for a CPU which runs Cpp code
3. OpenclGPU for a GPU card which runs OpenCL code


Create devices
---------------

Python API
~~~~~~~~~~

.. autofunction:: device.create_cuda_gpus

.. autofunction:: device.create_cuda_gpus_on

.. autofunction:: device.create_cuda_gpu_on

.. autofunction:: device.get_default_device


The following code shows how to create devices,

.. code:: python

   from singa import device
   cuda = device.create_cuda_gpu_on(0)  # use GPU card of ID 0
   host = device.get_default_device()  # get the default host device (a CppCPU)
   ary1 = device.create_cuda_gpus(2)  # create 2 devices, starting from ID 0
   ary2 = device.create_cuda_gpus([0,2])  # create 2 devices on ID 0 and 2



CPP API
~~~~~~~



