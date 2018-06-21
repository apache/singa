import unittest
from builtins import str

from singa import tensor
from singa import singa_wrap as singa
from singa import device
from singa import autograd

autograd.training = True

CTensor = singa.Tensor

dev = device.create_cuda_gpu()

gpu_input_tensor = tensor.Tensor(shape=(2, 3, 3, 3), device=dev)
gpu_input_tensor.gaussian(0.0, 1.0)

dy = CTensor([2, 1, 2, 2])
singa.Gaussian(0.0, 1.0, dy)
dy.ToDevice(dev)

conv = autograd.Conv2d_GPU(3, 1, 2)  # (in_channels, out_channels, kernel_size)


def _tuple_to_string(t):
    lt = [str(x) for x in t]
    return '(' + ', '.join(lt) + ')'


class TestPythonOperation(unittest.TestCase):

    def check_shape(self, actual, expect):
        self.assertEqual(actual, expect, 'shape mismatch, actual shape is %s'
                         ' exepcted is %s' % (_tuple_to_string(actual),
                                              _tuple_to_string(expect))
                         )

    def test(self):
        y = conv(gpu_input_tensor)  # PyTensor
        dx, dW, db = conv.backward(dy)  # CTensor
        
        self.check_shape(y.shape, (2, 1, 2, 2))
        self.check_shape(dx.shape(), (2, 3, 3, 3))
        self.check_shape(dW.shape(), (1, 3, 2, 2))
        self.check_shape(db.shape(), (1,))

if __name__ == '__main__':
    unittest.main()
