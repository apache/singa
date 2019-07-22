from singa import device

# avoid singleton error
gpu_dev = device.create_cuda_gpu()
cpu_dev = device.get_default_device()
