import unittest
from builtins import str

from singa import tensor
from singa import singa_wrap as singa
from singa import device
from singa import autograd

autograd.training = True

CTensor = singa.Tensor

gpu_dev = device.create_cuda_gpu()
cpu_dev = device.get_default_device()

dy = CTensor([2, 1, 2, 2])
singa.Gaussian(0.0, 1.0, dy)

conv = autograd.Conv2D(3, 1, 2)  # (in_channels, out_channels, kernel_size)
conv_without_bias = autograd.Conv2D(3,1,2,bias=False)


def _tuple_to_string(t):
    lt = [str(x) for x in t]
    return '(' + ', '.join(lt) + ')'


class TestPythonOperation(unittest.TestCase):

    def check_shape(self, actual, expect):
        self.assertEqual(actual, expect, 'shape mismatch, actual shape is %s'
                         ' exepcted is %s' % (_tuple_to_string(actual),
                                              _tuple_to_string(expect))
                         )

    def test_conv2d_gpu(self):
        gpu_input_tensor = tensor.Tensor(shape=(2, 3, 3, 3), device=gpu_dev)
        gpu_input_tensor.gaussian(0.0, 1.0)

        y = conv(gpu_input_tensor)  # PyTensor
        dx, dW, db = conv.backward(dy)  # CTensor

        self.check_shape(y.shape, (2, 1, 2, 2))
        self.check_shape(dx.shape(), (2, 3, 3, 3))
        self.check_shape(dW.shape(), (1, 3, 2, 2))
        self.check_shape(db.shape(), (1,))

        #forward without bias
        y_without_bias=conv_without_bias(gpu_input_tensor)
        self.check_shape(y.shape, (2, 1, 2, 2))

    def test_conv2d_cpu(self):
        cpu_input_tensor = tensor.Tensor(shape=(2, 3, 3, 3), device=cpu_dev)
        cpu_input_tensor.gaussian(0.0, 1.0)

        y = conv(cpu_input_tensor)  # PyTensor
        dx, dW, db = conv.backward(dy)  # CTensor

        self.check_shape(y.shape, (2, 1, 2, 2))
        self.check_shape(dx.shape(), (2, 3, 3, 3))
        self.check_shape(dW.shape(), (1, 3, 2, 2))
        self.check_shape(db.shape(), (1,))

        #forward without bias
        y_without_bias=conv_without_bias(cpu_input_tensor)
        self.check_shape(y.shape, (2, 1, 2, 2))

if __name__ == '__main__':
    unittest.main()
