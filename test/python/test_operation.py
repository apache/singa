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
        # (in_channels, out_channels, kernel_size)
        conv_0 = autograd.Conv2D(3, 1, 2)
        conv_without_bias_0 = autograd.Conv2D(3, 1, 2, bias=False)

        gpu_input_tensor = tensor.Tensor(shape=(2, 3, 3, 3), device=gpu_dev)
        gpu_input_tensor.gaussian(0.0, 1.0)

        y = conv_0(gpu_input_tensor)  # PyTensor
        dx, dW, db = y.creator.backward(dy)  # CTensor

        self.check_shape(y.shape, (2, 1, 2, 2))
        self.check_shape(dx.shape(), (2, 3, 3, 3))
        self.check_shape(dW.shape(), (1, 3, 2, 2))
        self.check_shape(db.shape(), (1,))

        # forward without bias
        y_without_bias = conv_without_bias_0(gpu_input_tensor)
        self.check_shape(y_without_bias.shape, (2, 1, 2, 2))

    def test_conv2d_cpu(self):
        # (in_channels, out_channels, kernel_size)
        conv_1 = autograd.Conv2D(3, 1, 2)
        conv_without_bias_1 = autograd.Conv2D(3, 1, 2, bias=False)

        cpu_input_tensor = tensor.Tensor(shape=(2, 3, 3, 3), device=cpu_dev)
        cpu_input_tensor.gaussian(0.0, 1.0)

        y = conv_1(cpu_input_tensor)  # PyTensor
        dx, dW, db = y.creator.backward(dy)  # CTensor

        self.check_shape(y.shape, (2, 1, 2, 2))
        self.check_shape(dx.shape(), (2, 3, 3, 3))
        self.check_shape(dW.shape(), (1, 3, 2, 2))
        self.check_shape(db.shape(), (1,))

        # forward without bias
        y_without_bias = conv_without_bias_1(cpu_input_tensor)
        self.check_shape(y_without_bias.shape, (2, 1, 2, 2))

    def test_batchnorm2d_gpu(self):
        batchnorm_0 = autograd.BatchNorm2d(3)

        gpu_input_tensor = tensor.Tensor(shape=(2, 3, 3, 3), device=gpu_dev)
        gpu_input_tensor.gaussian(0.0, 1.0)

        dy = CTensor([2, 3, 3, 3])
        singa.Gaussian(0.0, 1.0, dy)

        y = batchnorm_0(gpu_input_tensor)
        dx, ds, db = y.creator.backward(dy)

        self.check_shape(y.shape, (2, 3, 3, 3))
        self.check_shape(dx.shape(), (2, 3, 3, 3))
        self.check_shape(ds.shape(), (3,))
        self.check_shape(db.shape(), (3,))

if __name__ == '__main__':
    unittest.main()
