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
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import unittest
from builtins import str

from singa import tensor
from singa import singa_wrap as singa
from singa import autograd
from singa import singa_wrap
from cuda_helper import gpu_dev, cpu_dev

import numpy as np

autograd.training = True

CTensor = singa.Tensor

dy = CTensor([2, 1, 2, 2])
singa.Gaussian(0.0, 1.0, dy)


def _tuple_to_string(t):
    lt = [str(x) for x in t]
    return '(' + ', '.join(lt) + ')'


def axis_helper(y_shape, x_shape):
    """
    check which axes the x has been broadcasted
    Args:
        y_shape: the shape of result
        x_shape: the shape of x
    Return:
        a tuple refering the axes 
    """
    res = []
    j = len(x_shape) - 1
    for i in range(len(y_shape) - 1, -1, -1):
        if j < 0 or x_shape[j] != y_shape[i]:
            res.append(i)
        j -= 1
    return tuple(res[::-1])


def prepare_inputs_targets_for_rnn_test():
    x_0 = np.random.random((2, 3)).astype(np.float32)
    x_1 = np.random.random((2, 3)).astype(np.float32)
    x_2 = np.random.random((2, 3)).astype(np.float32)

    h_0 = np.zeros((2, 2)).astype(np.float32)

    t_0 = np.random.random((2, 2)).astype(np.float32)
    t_1 = np.random.random((2, 2)).astype(np.float32)
    t_2 = np.random.random((2, 2)).astype(np.float32)

    x0 = tensor.Tensor(device=gpu_dev, data=x_0)
    x1 = tensor.Tensor(device=gpu_dev, data=x_1)
    x2 = tensor.Tensor(device=gpu_dev, data=x_2)

    h0 = tensor.Tensor(device=gpu_dev, data=h_0)

    t0 = tensor.Tensor(device=gpu_dev, data=t_0)
    t1 = tensor.Tensor(device=gpu_dev, data=t_1)
    t2 = tensor.Tensor(device=gpu_dev, data=t_2)

    inputs = [x0, x1, x2]
    targets = [t0, t1, t2]
    return inputs, targets, h0


class TestPythonOperation(unittest.TestCase):

    def check_shape(self, actual, expect):
        self.assertEqual(
            actual, expect, 'shape mismatch, actual shape is %s'
            ' exepcted is %s' %
            (_tuple_to_string(actual), _tuple_to_string(expect)))

    def test_Greater_cpu(self):
        x0 = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                       0.9]).reshape(3, 2).astype(np.float32)
        x1 = np.array([0, -0.3, 0, 0.1, 0, 0.9]).reshape(3,
                                                         2).astype(np.float32)
        y = np.greater(x0, x1)
        x0 = tensor.from_numpy(x0)
        x1 = tensor.from_numpy(x1)
        x0.to_device(cpu_dev)
        x1.to_device(cpu_dev)

        result = autograd.greater(x0, x1)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)

    def test_Greater_gpu(self):
        x0 = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                       0.9]).reshape(3, 2).astype(np.float32)
        x1 = np.array([0, -0.3, 0, 0.1, 0, 0.9]).reshape(3,
                                                         2).astype(np.float32)
        y = np.greater(x0, x1)
        x0 = tensor.from_numpy(x0)
        x1 = tensor.from_numpy(x1)
        x0.to_device(gpu_dev)
        x1.to_device(gpu_dev)
        result = autograd.greater(x0, x1)
        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)

    def test_conv2d_cpu(self):
        # (in_channels, out_channels, kernel_size)
        conv_0 = autograd.Conv2d(3, 1, 2)
        conv_without_bias_0 = autograd.Conv2d(3, 1, 2, bias=False)

        cpu_input_tensor = tensor.Tensor(shape=(2, 3, 3, 3), device=cpu_dev)
        cpu_input_tensor.gaussian(0.0, 1.0)

        dy = tensor.Tensor(shape=(2, 1, 2, 2), device=cpu_dev)
        dy.gaussian(0.0, 1.0)

        y = conv_0(cpu_input_tensor)  # PyTensor
        dx, dW, db = y.creator.backward(dy.data)  # CTensor

        self.check_shape(y.shape, (2, 1, 2, 2))
        self.check_shape(dx.shape(), (2, 3, 3, 3))
        self.check_shape(dW.shape(), (1, 3, 2, 2))
        self.check_shape(db.shape(), (1,))

        # forward without bias
        y_without_bias = conv_without_bias_0(cpu_input_tensor)
        self.check_shape(y_without_bias.shape, (2, 1, 2, 2))

    def test_conv2d_gpu(self):
        # (in_channels, out_channels, kernel_size)
        conv_0 = autograd.Conv2d(3, 1, 2)
        conv_without_bias_0 = autograd.Conv2d(3, 1, 2, bias=False)

        gpu_input_tensor = tensor.Tensor(shape=(2, 3, 3, 3), device=gpu_dev)
        gpu_input_tensor.gaussian(0.0, 1.0)

        dy = tensor.Tensor(shape=(2, 1, 2, 2), device=gpu_dev)
        dy.gaussian(0.0, 1.0)

        y = conv_0(gpu_input_tensor)  # PyTensor
        dx, dW, db = y.creator.backward(dy.data)  # CTensor

        self.check_shape(y.shape, (2, 1, 2, 2))
        self.check_shape(dx.shape(), (2, 3, 3, 3))
        self.check_shape(dW.shape(), (1, 3, 2, 2))
        self.check_shape(db.shape(), (1,))

        # forward without bias
        y_without_bias = conv_without_bias_0(gpu_input_tensor)
        self.check_shape(y_without_bias.shape, (2, 1, 2, 2))

    def _conv_same_pad(self, dev, pad_mode, is_2d):
        if is_2d:
            x_h, w_h, k_h, p_h = 32, 4, 4, 1
            if pad_mode == "SAME_LOWER":
                o_p = (0, 1, 0, 1)
            else:
                o_p = (1, 0, 1, 0)
        else:
            x_h, w_h, k_h, p_h = 1, 1, 1, 0
            if pad_mode == "SAME_LOWER":
                o_p = (0, 0, 0, 1)
            else:
                o_p = (0, 0, 1, 0)
        x = tensor.Tensor(shape=(3, 3, x_h, 32), device=dev)
        x.gaussian(0.0, 1.0)

        w = tensor.Tensor(shape=(3, 3, w_h, 4), device=dev)
        w.gaussian(0.0, 1.0)

        # with the same padding, the padding should be 3
        # for SAME_UPPER, is (1, 1) + (0, 1)
        # for SAME_LOWER, is (1, 1) + (1, 0)

        x_shape = x.shape
        kernel = (k_h, 4)
        padding = (p_h, 1)
        stride = (1, 1)
        group = 1
        bias = False
        in_channels = x_shape[1]
        w_shape = w.shape
        out_channels = w_shape[0]
        assert w_shape[1] == in_channels // group

        if dev == cpu_dev:
            handle = singa.ConvHandle(x.data, kernel, stride, padding,
                                      in_channels, out_channels, bias, group)
        else:
            handle = singa.CudnnConvHandle(x.data, kernel, stride, padding,
                                           in_channels, out_channels, bias,
                                           group)
        y = autograd._Conv2d(handle, o_p)(x, w)[0]

        dy = np.ones((3, 3, x_h, 32), dtype=np.float32)
        dy = tensor.from_numpy(dy)
        dy.to_device(dev)

        dx, dW = y.creator.backward(dy.data)
        self.check_shape(y.shape, (3, 3, x_h, 32))
        self.check_shape(dx.shape(), (3, 3, x_h, 32))
        self.check_shape(dW.shape(), (3, 3, w_h, 4))

    def test_conv2d_same_pad_cpu(self):
        self._conv_same_pad(cpu_dev, "SAME_LOWER", True)
        self._conv_same_pad(cpu_dev, "SAME_UPPER", True)

    def test_conv2d_same_pad_gpu(self):
        self._conv_same_pad(gpu_dev, "SAME_LOWER", True)
        self._conv_same_pad(gpu_dev, "SAME_UPPER", True)

    def test_conv1d_same_pad_cpu(self):
        self._conv_same_pad(cpu_dev, "SAME_LOWER", False)
        self._conv_same_pad(cpu_dev, "SAME_UPPER", False)

    def test_conv1d_same_pad_gpu(self):
        self._conv_same_pad(gpu_dev, "SAME_LOWER", False)
        self._conv_same_pad(gpu_dev, "SAME_UPPER", False)

    def _pooling_same_pad(self, dev, pad_mode, is_2d):
        if is_2d:
            x_h, k_h, p_h = 32, 4, 1
            if pad_mode == "SAME_LOWER":
                o_p = (0, 1, 0, 1)
            else:
                o_p = (1, 0, 1, 0)
        else:
            x_h, k_h, p_h = 1, 1, 0
            if pad_mode == "SAME_LOWER":
                o_p = (0, 0, 0, 1)
            else:
                o_p = (0, 0, 1, 0)
        x = tensor.Tensor(shape=(3, 3, x_h, 32), device=dev)
        x.gaussian(0.0, 1.0)

        # with the same padding, the padding should be 3
        # for SAME_UPPER, is (1, 1) + (0, 1)
        # for SAME_LOWER, is (1, 1) + (1, 0)

        x_shape = x.shape
        kernel = (k_h, 4)
        # we add 4 padding here and hope the conv and trim one padding then
        padding = (p_h, 1)
        stride = (1, 1)

        if dev == cpu_dev:
            handle = singa.PoolingHandle(x.data, kernel, stride, padding, True)
        else:
            handle = singa.CudnnPoolingHandle(x.data, kernel, stride, padding,
                                              True)

        y = autograd._Pooling2d(handle, o_p)(x)[0]

        dy = np.ones((3, 3, x_h, 32), dtype=np.float32)
        dy = tensor.from_numpy(dy)
        dy.to_device(dev)

        dx = y.creator.backward(dy.data)
        self.check_shape(y.shape, (3, 3, x_h, 32))
        self.check_shape(dx.shape(), (3, 3, x_h, 32))

    def test_pooling2d_same_pad_cpu(self):
        self._pooling_same_pad(cpu_dev, "SAME_LOWER", True)
        self._pooling_same_pad(cpu_dev, "SAME_UPPER", True)

    def test_pooling2d_same_pad_gpu(self):
        self._pooling_same_pad(gpu_dev, "SAME_LOWER", True)
        self._pooling_same_pad(gpu_dev, "SAME_UPPER", True)

    def test_pooling1d_same_pad_cpu(self):
        self._pooling_same_pad(cpu_dev, "SAME_LOWER", False)
        self._pooling_same_pad(cpu_dev, "SAME_UPPER", False)

    def test_pooling1d_same_pad_gpu(self):
        self._pooling_same_pad(gpu_dev, "SAME_LOWER", False)
        self._pooling_same_pad(gpu_dev, "SAME_UPPER", False)

    def test_sum_cpu(self):
        x = np.array([0.1, -1.0, 0.4, 4.0, -0.9,
                      9.0]).reshape(3, 2).astype(np.float32)
        x1 = np.array([0.1, 1.0, 0.4, 4.0, 0.9,
                       9.0]).reshape(3, 2).astype(np.float32)
        y = x + x1
        dy = np.ones((3, 2), dtype=np.float32)
        grad0 = dy
        grad1 = dy
        x = tensor.from_numpy(x)
        x1 = tensor.from_numpy(x1)
        dy = tensor.from_numpy(dy)
        x.to_device(cpu_dev)
        x1.to_device(cpu_dev)
        dy.to_device(cpu_dev)

        result = autograd.sum(x, x1)
        dx0, dx1 = result.creator.backward(dy.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx0)),
                                             grad0,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx1)),
                                             grad1,
                                             decimal=5)

    def test_sum_gpu(self):
        x = np.array([0.1, -1.0, 0.4, 4.0, -0.9,
                      9.0]).reshape(3, 2).astype(np.float32)
        x1 = np.array([0.1, 1.0, 0.4, 4.0, 0.9,
                       9.0]).reshape(3, 2).astype(np.float32)
        y = x + x1
        dy = np.ones((3, 2), dtype=np.float32)
        grad0 = dy
        grad1 = dy
        x = tensor.from_numpy(x)
        x1 = tensor.from_numpy(x1)
        dy = tensor.from_numpy(dy)
        x.to_device(gpu_dev)
        x1.to_device(gpu_dev)
        dy.to_device(gpu_dev)

        result = autograd.sum(x, x1)
        dx0, dx1 = result.creator.backward(dy.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx0)),
                                             grad0,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx1)),
                                             grad1,
                                             decimal=5)

    def test_SeparableConv2d_gpu(self):
        # SeparableConv2d(in_channels, out_channels, kernel_size)
        separ_conv = autograd.SeparableConv2d(8, 16, 3, padding=1)

        x = np.random.random((10, 8, 28, 28)).astype(np.float32)
        x = tensor.Tensor(device=gpu_dev, data=x)

        y1 = separ_conv.depthwise_conv(x)
        y2 = separ_conv.point_conv(y1)

        dy1, dW_depth = y2.creator.backward(y2.data)
        dx, dW_spacial = y1.creator.backward(dy1)

        self.check_shape(y2.shape, (10, 16, 28, 28))

        self.check_shape(dy1.shape(), (10, 8, 28, 28))
        self.check_shape(dW_depth.shape(), (16, 8, 1, 1))

        self.check_shape(dx.shape(), (10, 8, 28, 28))
        self.check_shape(dW_spacial.shape(), (8, 1, 3, 3))

        y = separ_conv(x)
        self.check_shape(y.shape, (10, 16, 28, 28))

    def test_batchnorm2d_cpu(self):
        batchnorm_0 = autograd.BatchNorm2d(3)

        cpu_input_tensor = tensor.Tensor(shape=(2, 3, 3, 3), device=cpu_dev)
        cpu_input_tensor.gaussian(0.0, 1.0)

        dy = cpu_input_tensor.clone().data

        y = batchnorm_0(cpu_input_tensor)
        dx, ds, db = y.creator.backward(dy)

        self.check_shape(y.shape, (2, 3, 3, 3))
        self.check_shape(dx.shape(), (2, 3, 3, 3))
        self.check_shape(ds.shape(), (3,))
        self.check_shape(db.shape(), (3,))

    def test_batchnorm2d_gpu(self):
        batchnorm_0 = autograd.BatchNorm2d(3)

        gpu_input_tensor = tensor.Tensor(shape=(2, 3, 3, 3), device=gpu_dev)
        gpu_input_tensor.gaussian(0.0, 1.0)

        dy = gpu_input_tensor.clone().data

        y = batchnorm_0(gpu_input_tensor)
        dx, ds, db = y.creator.backward(dy)

        self.check_shape(y.shape, (2, 3, 3, 3))
        self.check_shape(dx.shape(), (2, 3, 3, 3))
        self.check_shape(ds.shape(), (3,))
        self.check_shape(db.shape(), (3,))

    def test_vanillaRNN_gpu_tiny_ops_shape_check(self):
        # gradients shape check.
        inputs, target, h0 = prepare_inputs_targets_for_rnn_test()
        rnn = autograd.RNN(3, 2)

        hs, _ = rnn(inputs, h0)

        loss = autograd.softmax_cross_entropy(hs[0], target[0])
        for i in range(1, len(hs)):
            l = autograd.softmax_cross_entropy(hs[i], target[i])
            loss = autograd.add(loss, l)
        # d=autograd.infer_dependency(loss.creator)
        # print(d)
        for t, dt in autograd.backward(loss):
            self.check_shape(t.shape, dt.shape)

    def test_LSTM_gpu_tiny_ops_shape_check(self):
        # gradients shape check.
        inputs, target, h0 = prepare_inputs_targets_for_rnn_test()
        c_0 = np.random.random((2, 1)).astype(np.float32)
        c0 = tensor.Tensor(device=gpu_dev, data=c_0)

        rnn = autograd.LSTM(3, 2)

        hs, _, _ = rnn(inputs, (h0, c0))
        loss = autograd.softmax_cross_entropy(hs[0], target[0])

        for i in range(1, len(hs)):
            l = autograd.softmax_cross_entropy(hs[i], target[i])
            loss = autograd.add(loss, l)
        # d=autograd.infer_dependency(loss.creator)
        # print(d)
        for t, dt in autograd.backward(loss):
            self.check_shape(t.shape, dt.shape)

    def gradients_check(self, func, param, autograds, h=0.0005, df=1):
        # param: PyTensor
        # autograds: numpy_tensor
        p = tensor.to_numpy(param)
        it = np.nditer(p, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            diff = np.zeros_like(p)
            diff[idx] += h
            diff = tensor.from_numpy(diff)
            diff.to_device(gpu_dev)

            param += diff
            pos = func()
            pos = tensor.to_numpy(pos)

            param -= diff
            param -= diff
            neg = func()
            neg = tensor.to_numpy(neg)

            numerical_grad = np.sum((pos - neg) * df) / (2 * h)
            #print((autograds[idx] - numerical_grad)/numerical_grad)
            # threshold set as -5% to +5%
            #self.assertAlmostEqual((autograds[idx] - numerical_grad)/(numerical_grad+0.0000001), 0., places=1)
            self.assertAlmostEqual(autograds[idx] - numerical_grad,
                                   0.,
                                   places=2)

            it.iternext()

    def test_numerical_gradients_check_for_vallina_rnn(self):
        inputs, target, h0 = prepare_inputs_targets_for_rnn_test()

        rnn = autograd.RNN(3, 2)

        def valinna_rnn_forward():
            hs, _ = rnn(inputs, h0)

            loss = autograd.softmax_cross_entropy(hs[0], target[0])
            for i in range(1, len(hs)):
                l = autograd.softmax_cross_entropy(hs[i], target[i])
                loss = autograd.add(loss, l)
            #grads = autograd.gradients(loss)
            return loss

        loss1 = valinna_rnn_forward()
        auto_grads = autograd.gradients(loss1)

        for param in rnn.params:
            auto_grad = tensor.to_numpy(auto_grads[param])

            self.gradients_check(valinna_rnn_forward, param, auto_grad)

    def test_numerical_gradients_check_for_lstm(self):
        inputs, target, h0 = prepare_inputs_targets_for_rnn_test()
        c_0 = np.zeros((2, 2)).astype(np.float32)
        c0 = tensor.Tensor(device=gpu_dev, data=c_0)

        rnn = autograd.LSTM(3, 2)

        def lstm_forward():
            hs, _, _ = rnn(inputs, (h0, c0))

            loss = autograd.softmax_cross_entropy(hs[0], target[0])
            for i in range(1, len(hs)):
                l = autograd.softmax_cross_entropy(hs[i], target[i])
                loss = autograd.add(loss, l)
            return loss

        loss1 = lstm_forward()
        auto_grads = autograd.gradients(loss1)

        for param in rnn.params:
            auto_grad = tensor.to_numpy(auto_grads[param])

            self.gradients_check(lstm_forward, param, auto_grad)

    def test_MeanSquareError(self):
        X = np.array([4.3, 5.4, 3.3, 3.6, 5.7,
                      6.0]).reshape(3, 2).astype(np.float32)
        T = np.array([4.4, 5.3, 3.2, 3.7, 5.4,
                      6.3]).reshape(3, 2).astype(np.float32)
        x = tensor.from_numpy(X)
        t = tensor.from_numpy(T)
        x.to_device(gpu_dev)
        t.to_device(gpu_dev)

        loss = autograd.mse_loss(x, t)
        dx = loss.creator.backward()[0]

        loss_np = tensor.to_numpy(loss)[0]
        self.assertAlmostEqual(loss_np, 0.0366666, places=4)
        self.check_shape(dx.shape(), (3, 2))

    def test_Abs(self):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5,
                      0.5]).reshape(3, 2).astype(np.float32)
        XT = np.array([0.8, 1.2, 3.3, 3.6, 0.5,
                       0.5]).reshape(3, 2).astype(np.float32)
        x = tensor.from_numpy(X)
        x.to_device(gpu_dev)

        result = autograd.abs(x)
        dx = result.creator.backward(x.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result), XT)
        self.check_shape(dx.shape(), (3, 2))

    def test_Mean_gpu(self):
        x0 = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                       0.9]).reshape(3, 2).astype(np.float32)
        x1 = np.array([0, -0.3, 0, 0.1, 0, 0.9]).reshape(3,
                                                         2).astype(np.float32)
        y = (x0 + x1) / 2
        grad = np.ones(x0.shape) / 2
        x0 = tensor.from_numpy(x0)
        x1 = tensor.from_numpy(x1)
        x0.to_device(gpu_dev)
        x1.to_device(gpu_dev)

        result = autograd.mean(x0, x1)
        dy = tensor.from_numpy(np.ones((3, 2)).astype(np.float32))
        dy.to_device(gpu_dev)
        dx0, dx1 = result.creator.backward(dy.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx0)),
                                             grad,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx1)),
                                             grad,
                                             decimal=5)

    def test_Mean_cpu(self):
        x0 = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                       0.9]).reshape(3, 2).astype(np.float32)
        x1 = np.array([0, -0.3, 0, 0.1, 0, 0.9]).reshape(3,
                                                         2).astype(np.float32)
        y = (x0 + x1) / 2
        grad = np.ones(x0.shape) / 2
        x0 = tensor.from_numpy(x0)
        x1 = tensor.from_numpy(x1)
        x0.to_device(cpu_dev)
        x1.to_device(cpu_dev)

        result = autograd.mean(x0, x1)
        dy = tensor.from_numpy(np.ones((3, 2)).astype(np.float32))
        dy.to_device(cpu_dev)
        dx0, dx1 = result.creator.backward(dy.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx0)),
                                             grad,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx1)),
                                             grad,
                                             decimal=5)

    def test_Exp(self):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5,
                      0.5]).reshape(3, 2).astype(np.float32)
        XT = np.exp(X)
        x = tensor.from_numpy(X)
        x.to_device(gpu_dev)

        result = autograd.exp(x)
        dx = result.creator.backward(x.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        self.check_shape(dx.shape(), (3, 2))

    def test_Identity_cpu(self):
        x = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                      0.9]).reshape(3, 2).astype(np.float32)
        y = x.copy()
        grad = np.ones(x.shape)
        x = tensor.from_numpy(x)
        x.to_device(cpu_dev)

        result = autograd.identity(x)
        dy = tensor.from_numpy(np.ones((3, 2)).astype(np.float32))
        dy.to_device(cpu_dev)
        dx = result.creator.backward(dy.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             grad,
                                             decimal=5)
        self.check_shape(dx.shape(), (3, 2))

    def test_Identity_gpu(self):
        x = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                      0.9]).reshape(3, 2).astype(np.float32)
        y = x.copy()
        grad = np.ones(x.shape)
        x = tensor.from_numpy(x)
        x.to_device(gpu_dev)

        result = autograd.identity(x)
        dy = tensor.from_numpy(np.ones((3, 2)).astype(np.float32))
        dy.to_device(gpu_dev)
        dx = result.creator.backward(dy.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             grad,
                                             decimal=5)
        self.check_shape(dx.shape(), (3, 2))

    def test_LeakyRelu(self):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5,
                      0.5]).reshape(3, 2).astype(np.float32)
        XT = np.array([0.8, -0.012, 3.3, -0.036, -0.005,
                       0.5]).reshape(3, 2).astype(np.float32)
        x = tensor.from_numpy(X)
        x.to_device(gpu_dev)

        result = autograd.leakyrelu(x)

        dx = result.creator.backward(x.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result), XT)
        self.check_shape(dx.shape(), (3, 2))

    def test_Relu_cpu(self):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5,
                      0.5]).reshape(3, 2).astype(np.float32)
        XT = np.maximum(X, 0)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(cpu_dev)
        dy.to_device(cpu_dev)

        result = autograd.relu(x)
        dx = result.creator.backward(dy.data)

        G = (X > 0).astype(np.float32)
        DX = np.multiply(G, DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_Relu_gpu(self):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5,
                      0.5]).reshape(3, 2).astype(np.float32)
        XT = np.maximum(X, 0)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(gpu_dev)
        dy.to_device(gpu_dev)

        result = autograd.relu(x)
        dx = result.creator.backward(dy.data)

        G = (X > 0).astype(np.float32)
        DX = np.multiply(G, DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_Cos_cpu(self):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5,
                      0.5]).reshape(3, 2).astype(np.float32)
        XT = np.cos(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(cpu_dev)
        dy.to_device(cpu_dev)

        result = autograd.cos(x)
        dx = result.creator.backward(dy.data)

        G = -np.sin(X)
        DX = np.multiply(G, DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_Cos_gpu(self):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5,
                      0.5]).reshape(3, 2).astype(np.float32)
        XT = np.cos(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(gpu_dev)
        dy.to_device(gpu_dev)

        result = autograd.cos(x)
        dx = result.creator.backward(dy.data)

        G = -np.sin(X)
        DX = np.multiply(G, DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_Cosh_cpu(self):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5,
                      0.5]).reshape(3, 2).astype(np.float32)
        XT = np.cosh(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(cpu_dev)
        dy.to_device(cpu_dev)

        result = autograd.cosh(x)
        dx = result.creator.backward(dy.data)

        G = np.sinh(X)
        DX = np.multiply(G, DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_Cosh_gpu(self):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5,
                      0.5]).reshape(3, 2).astype(np.float32)
        XT = np.cosh(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(gpu_dev)
        dy.to_device(gpu_dev)

        result = autograd.cosh(x)
        dx = result.creator.backward(dy.data)

        G = np.sinh(X)
        DX = np.multiply(G, DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_Acos_cpu(self):
        X = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                      0.9]).reshape(3, 2).astype(np.float32)
        XT = np.arccos(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(cpu_dev)
        dy.to_device(cpu_dev)

        result = autograd.acos(x)
        dx = result.creator.backward(dy.data)

        G = -1.0 / np.sqrt(1.0 - np.square(X))
        DX = np.multiply(G, DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_Acos_gpu(self):
        X = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                      0.9]).reshape(3, 2).astype(np.float32)
        XT = np.arccos(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(gpu_dev)
        dy.to_device(gpu_dev)

        result = autograd.acos(x)
        dx = result.creator.backward(dy.data)

        G = -1.0 / np.sqrt(1.0 - np.square(X))
        DX = np.multiply(G, DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_Acosh_cpu(self):
        X = np.array([1.1, 1.5, 1.9, 2.2, 2.5,
                      2.8]).reshape(3, 2).astype(np.float32)
        XT = np.arccosh(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(cpu_dev)
        dy.to_device(cpu_dev)

        result = autograd.acosh(x)
        dx = result.creator.backward(dy.data)

        G = 1.0 / np.multiply(np.sqrt(X - 1.0), np.sqrt(X + 1.0))
        DX = np.multiply(G, DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_Acosh_gpu(self):
        X = np.array([1.1, 1.5, 1.9, 2.2, 2.5,
                      2.8]).reshape(3, 2).astype(np.float32)
        XT = np.arccosh(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(gpu_dev)
        dy.to_device(gpu_dev)

        result = autograd.acosh(x)
        dx = result.creator.backward(dy.data)

        G = 1.0 / np.multiply(np.sqrt(X - 1.0), np.sqrt(X + 1.0))
        DX = np.multiply(G, DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_Sin_cpu(self):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5,
                      0.5]).reshape(3, 2).astype(np.float32)
        XT = np.sin(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(cpu_dev)
        dy.to_device(cpu_dev)

        result = autograd.sin(x)
        dx = result.creator.backward(dy.data)

        G = np.cos(X)
        DX = np.multiply(G, DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_Sin_gpu(self):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5,
                      0.5]).reshape(3, 2).astype(np.float32)
        XT = np.sin(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(gpu_dev)
        dy.to_device(gpu_dev)

        result = autograd.sin(x)
        dx = result.creator.backward(dy.data)

        G = np.cos(X)
        DX = np.multiply(G, DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_Sinh_cpu(self):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5,
                      0.5]).reshape(3, 2).astype(np.float32)
        XT = np.sinh(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(cpu_dev)
        dy.to_device(cpu_dev)

        result = autograd.sinh(x)
        dx = result.creator.backward(dy.data)

        G = np.cosh(X)
        DX = np.multiply(G, DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_Sinh_gpu(self):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5,
                      0.5]).reshape(3, 2).astype(np.float32)
        XT = np.sinh(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(gpu_dev)
        dy.to_device(gpu_dev)

        result = autograd.sinh(x)
        dx = result.creator.backward(dy.data)

        G = np.cosh(X)
        DX = np.multiply(G, DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_Asin_cpu(self):
        X = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                      0.9]).reshape(3, 2).astype(np.float32)
        XT = np.arcsin(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(cpu_dev)
        dy.to_device(cpu_dev)

        result = autograd.asin(x)
        dx = result.creator.backward(dy.data)

        G = 1.0 / np.sqrt(1.0 - np.square(X))
        DX = np.multiply(G, DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_Asin_gpu(self):
        X = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                      0.9]).reshape(3, 2).astype(np.float32)
        XT = np.arcsin(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(gpu_dev)
        dy.to_device(gpu_dev)

        result = autograd.asin(x)
        dx = result.creator.backward(dy.data)

        G = 1.0 / np.sqrt(1.0 - np.square(X))
        DX = np.multiply(G, DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_Asinh_cpu(self):
        X = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                      0.9]).reshape(3, 2).astype(np.float32)
        XT = np.arcsinh(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(cpu_dev)
        dy.to_device(cpu_dev)

        result = autograd.asinh(x)
        dx = result.creator.backward(dy.data)

        G = 1.0 / np.sqrt(np.square(X) + 1.0)
        DX = np.multiply(G, DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_Less_gpu(self):
        x0 = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                       0.9]).reshape(3, 2).astype(np.float32)
        x1 = np.array([0, -0.3, 0, 0.1, 0, 0.9]).reshape(3,
                                                         2).astype(np.float32)
        y = np.less(x0, x1)
        x0 = tensor.from_numpy(x0)
        x1 = tensor.from_numpy(x1)
        x0.to_device(gpu_dev)
        x1.to_device(gpu_dev)

        result = autograd.less(x0, x1)
        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)

    def test_Less_cpu(self):
        x0 = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                       0.9]).reshape(3, 2).astype(np.float32)
        x1 = np.array([0, -0.3, 0, 0.1, 0, 0.9]).reshape(3,
                                                         2).astype(np.float32)
        y = np.less(x0, x1)
        x0 = tensor.from_numpy(x0)
        x1 = tensor.from_numpy(x1)
        x0.to_device(cpu_dev)
        x1.to_device(cpu_dev)

        result = autograd.less(x0, x1)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)

    def test_Asinh_gpu(self):
        X = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                      0.9]).reshape(3, 2).astype(np.float32)
        XT = np.arcsinh(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(gpu_dev)
        dy.to_device(gpu_dev)

        result = autograd.asinh(x)
        dx = result.creator.backward(dy.data)

        G = 1.0 / np.sqrt(np.square(X) + 1.0)
        DX = np.multiply(G, DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_Tan_cpu(self):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5,
                      0.5]).reshape(3, 2).astype(np.float32)
        XT = np.tan(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(cpu_dev)
        dy.to_device(cpu_dev)

        result = autograd.tan(x)
        dx = result.creator.backward(dy.data)

        G = 1.0 / np.square(np.cos(X))
        DX = np.multiply(G, DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_Tan_gpu(self):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5,
                      0.5]).reshape(3, 2).astype(np.float32)
        XT = np.tan(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(gpu_dev)
        dy.to_device(gpu_dev)

        result = autograd.tan(x)
        dx = result.creator.backward(dy.data)

        G = 1.0 / np.square(np.cos(X))
        DX = np.multiply(G, DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_Tanh_cpu(self):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5,
                      0.5]).reshape(3, 2).astype(np.float32)
        XT = np.tanh(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(cpu_dev)
        dy.to_device(cpu_dev)

        result = autograd.tanh(x)
        dx = result.creator.backward(dy.data)

        G = 1.0 / np.square(np.cosh(X))
        DX = np.multiply(G, DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_Tanh_gpu(self):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5,
                      0.5]).reshape(3, 2).astype(np.float32)
        XT = np.tanh(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(gpu_dev)
        dy.to_device(gpu_dev)

        result = autograd.tanh(x)
        dx = result.creator.backward(dy.data)

        G = 1.0 / np.square(np.cosh(X))
        DX = np.multiply(G, DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_Atan_cpu(self):
        X = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                      0.9]).reshape(3, 2).astype(np.float32)
        XT = np.arctan(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(cpu_dev)
        dy.to_device(cpu_dev)

        result = autograd.atan(x)
        dx = result.creator.backward(dy.data)

        G = 1.0 / (1.0 + np.square(X))
        DX = np.multiply(G, DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_Atan_gpu(self):
        X = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                      0.9]).reshape(3, 2).astype(np.float32)
        XT = np.arctan(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(gpu_dev)
        dy.to_device(gpu_dev)

        result = autograd.atan(x)
        dx = result.creator.backward(dy.data)

        G = 1.0 / (1.0 + np.square(X))
        DX = np.multiply(G, DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_Atanh_cpu(self):
        X = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                      0.9]).reshape(3, 2).astype(np.float32)
        XT = np.arctanh(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(cpu_dev)
        dy.to_device(cpu_dev)

        result = autograd.atanh(x)
        dx = result.creator.backward(dy.data)

        G = 1.0 / (1.0 - np.square(X))
        DX = np.multiply(G, DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_Atanh_gpu(self):
        X = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                      0.9]).reshape(3, 2).astype(np.float32)
        XT = np.arctanh(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(gpu_dev)
        dy.to_device(gpu_dev)

        result = autograd.atanh(x)
        dx = result.creator.backward(dy.data)

        G = 1.0 / (1.0 - np.square(X))
        DX = np.multiply(G, DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_Sub_cpu(self):
        X0 = np.array([7, -5, 0.2, -0.1, 0.3, 4]).reshape(3,
                                                          2).astype(np.float32)
        X1 = np.array([0.6, -1.3, 0.1, -0.1, 0.4,
                       0.3]).reshape(3, 2).astype(np.float32)
        XT = np.subtract(X0, X1)

        DY = np.ones((3, 2), dtype=np.float32)
        x0 = tensor.from_numpy(X0)
        x1 = tensor.from_numpy(X1)
        dy = tensor.from_numpy(DY)
        x0.to_device(cpu_dev)
        x1.to_device(cpu_dev)
        dy.to_device(cpu_dev)

        result = autograd.sub(x0, x1)
        dx0, dx1 = result.creator.backward(dy.data)

        DX0 = np.multiply(DY, 1.0)
        DX1 = np.multiply(DY, -1.0)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx0)),
                                             DX0,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx1)),
                                             DX1,
                                             decimal=5)

    def test_Sub_gpu(self):
        X0 = np.array([7, -5, 0.2, -0.1, 0.3, 4]).reshape(3,
                                                          2).astype(np.float32)
        X1 = np.array([0.6, -1.3, 0.1, -0.1, 0.4,
                       0.3]).reshape(3, 2).astype(np.float32)
        XT = np.subtract(X0, X1)

        DY = np.ones((3, 2), dtype=np.float32)
        x0 = tensor.from_numpy(X0)
        x1 = tensor.from_numpy(X1)
        dy = tensor.from_numpy(DY)
        x0.to_device(gpu_dev)
        x1.to_device(gpu_dev)
        dy.to_device(gpu_dev)

        result = autograd.sub(x0, x1)
        dx0, dx1 = result.creator.backward(dy.data)
        DX0 = np.multiply(DY, 1.0)
        DX1 = np.multiply(DY, -1.0)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx0)),
                                             DX0,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx1)),
                                             DX1,
                                             decimal=5)

    def test_Pow_cpu(self):
        X0 = np.array([7, 5, 0.2, 0.1, 0.3, 4]).reshape(3, 2).astype(np.float32)
        X1 = np.array([-1.0, 2.0, -1.0, -2.1, 1.0,
                       -2.0]).reshape(3, 2).astype(np.float32)
        XT = np.power(X0, X1)

        DY = np.ones((3, 2), dtype=np.float32)
        x0 = tensor.from_numpy(X0)
        x1 = tensor.from_numpy(X1)
        dy = tensor.from_numpy(DY)
        x0.to_device(cpu_dev)
        x1.to_device(cpu_dev)
        dy.to_device(cpu_dev)

        result = autograd.pow(x0, x1)
        dx0, dx1 = result.creator.backward(dy.data)

        G0 = np.multiply(X1, np.power(X0, (X1 - 1.0)))
        DX0 = np.multiply(G0, DY)
        G1 = np.multiply(np.power(X0, X1), np.log(X0))
        DX1 = np.multiply(G1, DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx0)),
                                             DX0,
                                             decimal=4)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx1)),
                                             DX1,
                                             decimal=4)

    def test_Pow_gpu(self):
        X0 = np.array([7, 5, 0.2, 0.1, 0.3, 4]).reshape(3, 2).astype(np.float32)
        X1 = np.array([-1.0, 2.0, -1.0, -2.1, 1.0,
                       -2.0]).reshape(3, 2).astype(np.float32)
        XT = np.power(X0, X1)

        DY = np.ones((3, 2), dtype=np.float32)
        x0 = tensor.from_numpy(X0)
        x1 = tensor.from_numpy(X1)
        dy = tensor.from_numpy(DY)
        x0.to_device(gpu_dev)
        x1.to_device(gpu_dev)
        dy.to_device(gpu_dev)

        result = autograd.pow(x0, x1)
        dx0, dx1 = result.creator.backward(dy.data)

        G0 = np.multiply(X1, np.power(X0, (X1 - 1.0)))
        DX0 = np.multiply(G0, DY)
        G1 = np.multiply(np.power(X0, X1), np.log(X0))
        DX1 = np.multiply(G1, DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx0)),
                                             DX0,
                                             decimal=4)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx1)),
                                             DX1,
                                             decimal=4)

    def test_SoftSign_cpu(self):
        # y = x / (1 + np.abs(x))
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5,
                      0.5]).reshape(3, 2).astype(np.float32)
        XT = X / (1 + np.absolute(X))
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(cpu_dev)
        dy.to_device(cpu_dev)

        result = autograd.softsign(x)
        dx = result.creator.backward(dy.data)

        G = 1.0 / np.square(np.absolute(X) + 1.0)
        DX = np.multiply(G, DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_SoftSign_gpu(self):
        # y = x / (1 + np.abs(x))
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5,
                      0.5]).reshape(3, 2).astype(np.float32)
        XT = X / (1 + np.absolute(X))
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(gpu_dev)
        dy.to_device(gpu_dev)

        result = autograd.softsign(x)
        dx = result.creator.backward(dy.data)

        G = 1.0 / np.square(np.absolute(X) + 1.0)
        DX = np.multiply(G, DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_SoftPlus_cpu(self):
        #y = np.log(np.exp(x) + 1)
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5,
                      0.5]).reshape(3, 2).astype(np.float32)
        XT = np.log(np.exp(X) + 1)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(cpu_dev)
        dy.to_device(cpu_dev)

        result = autograd.softplus(x)
        dx = result.creator.backward(dy.data)

        G = 1.0 / (1.0 + np.exp(-X))
        DX = np.multiply(G, DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_SoftPlus_gpu(self):
        #y = np.log(np.exp(x) + 1)
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5,
                      0.5]).reshape(3, 2).astype(np.float32)
        XT = np.log(np.exp(X) + 1)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(gpu_dev)
        dy.to_device(gpu_dev)

        result = autograd.softplus(x)
        dx = result.creator.backward(dy.data)

        G = 1.0 / (1.0 + np.exp(-X))
        DX = np.multiply(G, DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_unsqueeze_cpu(self):
        data = [0.1, -1.0, 0.4, 4.0, -0.9, 9.0]

        x = np.array(data).reshape(1, 2, 3).astype(np.float32)
        y = x.reshape(1, 1, 2, 3, 1)
        dy = np.ones((1, 1, 2, 3, 1), dtype=np.float32)
        grad = dy.reshape(1, 2, 3)

        x = tensor.from_numpy(x)
        dy = tensor.from_numpy(dy)
        x.to_device(cpu_dev)
        dy.to_device(cpu_dev)

        result = autograd.unsqueeze(x, [0, 4])
        dx = result.creator.backward(dy.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             grad,
                                             decimal=5)

    def test_unsqueeze_gpu(self):
        x = np.array([0.1, -1.0, 0.4, 4.0, -0.9,
                      9.0]).reshape(1, 2, 3).astype(np.float32)
        y = x.reshape(1, 1, 2, 3, 1)
        dy = np.ones((1, 1, 2, 3, 1), dtype=np.float32)
        grad = dy.reshape(1, 2, 3)

        x = tensor.from_numpy(x)
        dy = tensor.from_numpy(dy)
        x.to_device(cpu_dev)
        dy.to_device(cpu_dev)

        result = autograd.unsqueeze(x, [0, 4])
        dx = result.creator.backward(dy.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             grad,
                                             decimal=5)

    def test_Sqrt_cpu(self):
        X = np.array([0.1, 1.0, 0.4, 4.0, 0.9,
                      9.0]).reshape(3, 2).astype(np.float32)
        XT = np.sqrt(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(cpu_dev)
        dy.to_device(cpu_dev)

        result = autograd.sqrt(x)
        dx = result.creator.backward(dy.data)

        G = 0.5 * np.power(X, -0.5)
        DX = np.multiply(G, DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_Sqrt_gpu(self):
        X = np.array([0.1, 1.0, 0.4, 4.0, 0.9,
                      9.0]).reshape(3, 2).astype(np.float32)
        XT = np.sqrt(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(gpu_dev)
        dy.to_device(gpu_dev)

        result = autograd.sqrt(x)
        dx = result.creator.backward(dy.data)

        G = 0.5 * np.power(X, -0.5)
        DX = np.multiply(G, DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_transpose_cpu(self):
        x = np.random.randn(3, 2, 1)
        y = x.transpose(1, 2, 0)
        dy = np.random.randn(*(y.shape))
        grad = dy.transpose((2, 0, 1))

        x = tensor.from_numpy(x)
        dy = tensor.from_numpy(dy)
        x.to_device(cpu_dev)
        dy.to_device(cpu_dev)

        result = autograd.transpose(x, (1, 2, 0))
        dx = result.creator.backward(dy.data)
        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             grad,
                                             decimal=5)

    def test_transpose_gpu(self):
        x = np.random.randn(3, 2, 1)
        y = x.transpose(1, 2, 0)
        dy = np.random.randn(*(y.shape))
        grad = dy.transpose((2, 0, 1))

        x = tensor.from_numpy(x)
        dy = tensor.from_numpy(dy)
        x.to_device(gpu_dev)
        dy.to_device(gpu_dev)

        result = autograd.transpose(x, (1, 2, 0))
        dx = result.creator.backward(dy.data)
        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             grad,
                                             decimal=5)

    def test_Sign_cpu(self):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5,
                      0.5]).reshape(3, 2).astype(np.float32)
        XT = np.sign(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(gpu_dev)
        dy.to_device(gpu_dev)
        result = autograd.sign(x)
        dx = result.creator.backward(dy.data)
        DX = np.multiply(DY, 0)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_Sign_gpu(self):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5,
                      0.5]).reshape(3, 2).astype(np.float32)
        XT = np.sign(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(gpu_dev)
        dy.to_device(gpu_dev)
        result = autograd.sign(x)
        dx = result.creator.backward(dy.data)
        DX = np.multiply(DY, 0)
        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_Log_cpu(self):
        X = np.array([0.1, 1.0, 0.4, 1.4, 0.9,
                      2.0]).reshape(3, 2).astype(np.float32)
        XT = np.log(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(gpu_dev)
        dy.to_device(gpu_dev)
        result = autograd.log(x)
        dx = result.creator.backward(dy.data)
        #dx = 1/x
        G = 1.0 / X
        DX = np.multiply(G, DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_Log_gpu(self):
        X = np.array([0.1, 1.0, 0.4, 1.4, 0.9,
                      2.0]).reshape(3, 2).astype(np.float32)
        XT = np.log(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(gpu_dev)
        dy.to_device(gpu_dev)
        result = autograd.log(x)
        dx = result.creator.backward(dy.data)
        #dx = 1/x
        G = 1.0 / X
        DX = np.multiply(G, DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_mul_cpu(self):
        x = np.array([0.1, -1.0, 0.4, 4.0, -0.9,
                      9.0]).reshape(3, 2).astype(np.float32)
        x1 = np.array([0.1, 1.0, 0.4, 4.0, 0.9,
                       9.0]).reshape(3, 2).astype(np.float32)
        y = x * x1
        dy = np.array([0.1, 1.0, 0.4, 4.0, 0.9,
                       9.0]).reshape(3, 2).astype(np.float32)
        grad0 = x1 * dy
        grad1 = x * dy

        x = tensor.from_numpy(x)
        slope = tensor.from_numpy(x1)
        dy = tensor.from_numpy(dy)
        x.to_device(cpu_dev)
        slope.to_device(cpu_dev)
        dy.to_device(cpu_dev)

        result = autograd.mul(x, slope)
        dx0, dx1 = result.creator.backward(dy.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx0)),
                                             grad0,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx1)),
                                             grad1,
                                             decimal=5)

    def test_mul_gpu(self):
        x = np.array([0.1, -1.0, 0.4, 4.0, -0.9,
                      9.0]).reshape(3, 2).astype(np.float32)
        x1 = np.array([0.1, 1.0, 0.4, 4.0, 0.9,
                       9.0]).reshape(3, 2).astype(np.float32)
        y = x * x1
        dy = np.array([0.1, 1.0, 0.4, 4.0, 0.9,
                       9.0]).reshape(3, 2).astype(np.float32)
        grad0 = x1 * dy
        grad1 = x * dy

        x = tensor.from_numpy(x)
        slope = tensor.from_numpy(x1)
        dy = tensor.from_numpy(dy)
        x.to_device(gpu_dev)
        slope.to_device(gpu_dev)
        dy.to_device(gpu_dev)

        result = autograd.mul(x, slope)
        dx0, dx1 = result.creator.backward(dy.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx0)),
                                             grad0,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx1)),
                                             grad1,
                                             decimal=5)

    def test_reshape_cpu(self):
        x = np.array([0.1, -1.0, 0.4, 4.0, -0.9,
                      9.0]).reshape(3, 2).astype(np.float32)
        y = x.reshape(2, 3)
        dy = np.array([1, 2, 3, 4, 5, 6]).reshape(2, 3).astype(np.float32)
        grad = dy.reshape(3, 2)

        x = tensor.from_numpy(x)
        dy = tensor.from_numpy(dy)
        x.to_device(cpu_dev)
        dy.to_device(cpu_dev)

        result = autograd.reshape(x, (2, 3))
        dx = result.creator.backward(dy.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             grad,
                                             decimal=5)

    def test_reshape_gpu(self):
        x = np.array([0.1, -1.0, 0.4, 4.0, -0.9,
                      9.0]).reshape(3, 2).astype(np.float32)
        y = x.reshape(2, 3)
        dy = np.array([1, 2, 3, 4, 5, 6]).reshape(2, 3).astype(np.float32)
        grad = dy.reshape(3, 2)

        x = tensor.from_numpy(x)
        dy = tensor.from_numpy(dy)
        x.to_device(gpu_dev)
        dy.to_device(gpu_dev)

        result = autograd.reshape(x, (2, 3))
        dx = result.creator.backward(dy.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             grad,
                                             decimal=5)

    def test_max_cpu(self):
        X0 = np.array([0.1, 0.2, 2.0, 0.0, 0.1,
                       0.2]).reshape(3, 2).astype(np.float32)
        X1 = np.array([1.0, 2.0, 1.0, 2.1, 0.0,
                       2.0]).reshape(3, 2).astype(np.float32)
        XT = np.maximum(X0, X1)

        DY = np.ones((3, 2), dtype=np.float32)
        x0 = tensor.from_numpy(X0)
        x1 = tensor.from_numpy(X1)
        dy = tensor.from_numpy(DY)
        x0.to_device(cpu_dev)
        x1.to_device(cpu_dev)
        dy.to_device(cpu_dev)

        result = autograd.max(x0, x1)
        dx0, dx1 = result.creator.backward(dy.data)

        G = np.subtract(X0, X1)
        DX0 = np.where(G > 0, 1, G * 0)
        DX1 = np.where(G < 0, 1, G * 0)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx0)),
                                             DX0,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx1)),
                                             DX1,
                                             decimal=5)

    def test_max_cpu_3inputs(self):
        data_0 = np.array([3, 2, 1]).astype(np.float32)
        data_1 = np.array([1, 4, 4]).astype(np.float32)
        data_2 = np.array([2, 5, 3]).astype(np.float32)
        XT = np.array([3, 5, 4]).astype(np.float32)

        DY = np.array([1, 1, 1]).astype(np.float32)
        x0 = tensor.from_numpy(data_0)
        x1 = tensor.from_numpy(data_1)
        x2 = tensor.from_numpy(data_2)
        dy = tensor.from_numpy(DY)
        x0.to_device(cpu_dev)
        x1.to_device(cpu_dev)
        x2.to_device(cpu_dev)
        dy.to_device(cpu_dev)

        result = autograd.max(x0, x1, x2)
        dx0, dx1, dx2 = result.creator.backward(dy.data)

        DX0 = np.array([1, 0, 0]).astype(np.float32)
        DX1 = np.array([0, 0, 1]).astype(np.float32)
        DX2 = np.array([0, 1, 0]).astype(np.float32)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx0)),
                                             DX0,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx1)),
                                             DX1,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx2)),
                                             DX2,
                                             decimal=5)

    def test_max_cpu_1inputs(self):
        data_0 = np.array([3, 2, 1]).astype(np.float32)
        XT = np.array([3, 2, 1]).astype(np.float32)

        DY = np.array([1, 1, 1]).astype(np.float32)
        x0 = tensor.from_numpy(data_0)
        dy = tensor.from_numpy(DY)
        x0.to_device(cpu_dev)
        dy.to_device(cpu_dev)

        result = autograd.max(x0)
        dx0 = result.creator.backward(dy.data)

        DX0 = np.array([1, 1, 1]).astype(np.float32)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)

    def test_max_gpu(self):
        X0 = np.array([0.1, 0.2, 2.0, 0.0, 0.1,
                       0.2]).reshape(3, 2).astype(np.float32)
        X1 = np.array([1.0, 2.0, 1.0, 2.1, 0.0,
                       2.0]).reshape(3, 2).astype(np.float32)
        XT = np.maximum(X0, X1)

        DY = np.ones((3, 2), dtype=np.float32)
        x0 = tensor.from_numpy(X0)
        x1 = tensor.from_numpy(X1)
        dy = tensor.from_numpy(DY)
        x0.to_device(gpu_dev)
        x1.to_device(gpu_dev)
        dy.to_device(gpu_dev)

        result = autograd.max(x0, x1)
        dx0, dx1 = result.creator.backward(dy.data)

        G = np.subtract(X0, X1)
        DX0 = np.where(G > 0, 1, G * 0)
        DX1 = np.where(G < 0, 1, G * 0)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx0)),
                                             DX0,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx1)),
                                             DX1,
                                             decimal=5)

    def test_max_gpu_3inputs(self):
        data_0 = np.array([3, 2, 1]).astype(np.float32)
        data_1 = np.array([1, 4, 4]).astype(np.float32)
        data_2 = np.array([2, 5, 3]).astype(np.float32)
        XT = np.array([3, 5, 4]).astype(np.float32)

        DY = np.array([1, 1, 1]).astype(np.float32)
        x0 = tensor.from_numpy(data_0)
        x1 = tensor.from_numpy(data_1)
        x2 = tensor.from_numpy(data_2)
        dy = tensor.from_numpy(DY)
        x0.to_device(gpu_dev)
        x1.to_device(gpu_dev)
        x2.to_device(gpu_dev)
        dy.to_device(gpu_dev)

        result = autograd.max(x0, x1, x2)
        dx0, dx1, dx2 = result.creator.backward(dy.data)

        DX0 = np.array([1, 0, 0]).astype(np.float32)
        DX1 = np.array([0, 0, 1]).astype(np.float32)
        DX2 = np.array([0, 1, 0]).astype(np.float32)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx0)),
                                             DX0,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx1)),
                                             DX1,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx2)),
                                             DX2,
                                             decimal=5)

    def test_max_gpu_1inputs(self):
        data_0 = np.array([3, 2, 1]).astype(np.float32)
        XT = np.array([3, 2, 1]).astype(np.float32)

        DY = np.array([1, 1, 1]).astype(np.float32)
        x0 = tensor.from_numpy(data_0)
        dy = tensor.from_numpy(DY)
        x0.to_device(gpu_dev)
        dy.to_device(gpu_dev)

        result = autograd.max(x0)
        dx0 = result.creator.backward(dy.data)

        DX0 = np.array([1, 1, 1]).astype(np.float32)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)

    def test_Div_cpu(self):
        X0 = np.array([7, -5, 0.2, -0.1, 0.3, 4]).reshape(3,
                                                          2).astype(np.float32)
        X1 = np.array([0.6, -1.3, 0.1, -0.1, 0.4,
                       0.3]).reshape(3, 2).astype(np.float32)
        XT = np.divide(X0, X1)

        DY = np.ones((3, 2), dtype=np.float32)
        x0 = tensor.from_numpy(X0)
        x1 = tensor.from_numpy(X1)
        dy = tensor.from_numpy(DY)
        x0.to_device(cpu_dev)
        x1.to_device(cpu_dev)
        dy.to_device(cpu_dev)

        result = autograd.div(x0, x1)
        dx0, dx1 = result.creator.backward(dy.data)

        G0 = 1.0 / X1
        DX0 = np.multiply(G0, DY)
        G1 = np.divide(-X0, np.square(X1))
        DX1 = np.multiply(G1, DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx0)),
                                             DX0,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx1)),
                                             DX1,
                                             decimal=5)

    def test_Div_gpu(self):
        X0 = np.array([7, -5, 0.2, -0.1, 0.3, 4]).reshape(3,
                                                          2).astype(np.float32)
        X1 = np.array([0.6, -1.3, 0.1, -0.1, 0.4,
                       0.3]).reshape(3, 2).astype(np.float32)
        XT = np.divide(X0, X1)

        DY = np.ones((3, 2), dtype=np.float32)
        x0 = tensor.from_numpy(X0)
        x1 = tensor.from_numpy(X1)
        dy = tensor.from_numpy(DY)
        x0.to_device(gpu_dev)
        x1.to_device(gpu_dev)
        dy.to_device(gpu_dev)

        result = autograd.div(x0, x1)
        dx0, dx1 = result.creator.backward(dy.data)

        G0 = 1.0 / X1
        DX0 = np.multiply(G0, DY)
        G1 = np.divide(-X0, np.square(X1))
        DX1 = np.multiply(G1, DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx0)),
                                             DX0,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx1)),
                                             DX1,
                                             decimal=5)

    def test_squeeze(self):

        def squeeze_helper(gpu=False):
            x = np.random.randn(3, 1, 2, 1, 1)
            y = x.reshape(3, 2)
            dy = np.random.randn(3, 2)
            grad = dy.reshape(3, 1, 2, 1, 1)

            x = tensor.from_numpy(x)
            dy = tensor.from_numpy(dy)
            if (gpu):
                x.to_device(gpu_dev)
                dy.to_device(gpu_dev)

            result = autograd.squeeze(x, [1, 3, 4])
            dx = result.creator.backward(dy.data)

            np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                                 y,
                                                 decimal=5)
            np.testing.assert_array_almost_equal(tensor.to_numpy(
                tensor.from_raw_tensor(dx)),
                                                 grad,
                                                 decimal=5)

        squeeze_helper(False)
        squeeze_helper(True)

    def test_shape_cpu(self):
        x = np.array([0.1, -1.0, 0.4, 4.0, -0.9,
                      9.0]).reshape(3, 2).astype(np.float32)
        y = list(x.shape)
        dy = np.ones((3, 2), dtype=np.float32)
        grad = list(dy.shape)

        x = tensor.from_numpy(x)
        dy = tensor.from_numpy(dy)
        x.to_device(cpu_dev)
        dy.to_device(cpu_dev)

        result = autograd.shape(x)
        dx = result.creator.backward(dy.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)
        np.testing.assert_array_almost_equal(dx, grad, decimal=5)

    def test_shape_gpu(self):
        x = np.array([0.1, -1.0, 0.4, 4.0, -0.9,
                      9.0]).reshape(3, 2).astype(np.float32)
        y = list(x.shape)
        dy = np.ones((3, 2), dtype=np.float32)
        grad = list(dy.shape)

        x = tensor.from_numpy(x)
        dy = tensor.from_numpy(dy)
        x.to_device(gpu_dev)
        dy.to_device(gpu_dev)

        result = autograd.shape(x)
        dx = result.creator.backward(dy.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)
        np.testing.assert_array_almost_equal(dx, grad, decimal=5)

    def test_min_cpu(self):
        X0 = np.array([0.1, 0.2, 2.0, 0.0, 0.1,
                       0.2]).reshape(3, 2).astype(np.float32)
        X1 = np.array([1.0, 2.0, 1.0, 2.1, 0.0,
                       2.0]).reshape(3, 2).astype(np.float32)
        XT = np.minimum(X0, X1)

        DY = np.ones((3, 2), dtype=np.float32)
        x0 = tensor.from_numpy(X0)
        x1 = tensor.from_numpy(X1)
        dy = tensor.from_numpy(DY)
        x0.to_device(cpu_dev)
        x1.to_device(cpu_dev)
        dy.to_device(cpu_dev)

        result = autograd.min(x0, x1)
        dx0, dx1 = result.creator.backward(dy.data)

        G = np.subtract(X0, X1)
        DX0 = np.where(G < 0, 1, G * 0)
        DX1 = np.where(G > 0, 1, G * 0)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx0)),
                                             DX0,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx1)),
                                             DX1,
                                             decimal=5)

    def test_min_cpu_3inputs(self):
        data_0 = np.array([3, 2, 1]).astype(np.float32)
        data_1 = np.array([1, 4, 4]).astype(np.float32)
        data_2 = np.array([2, 5, 0]).astype(np.float32)
        XT = np.array([1, 2, 0]).astype(np.float32)

        DY = np.array([1, 1, 1]).astype(np.float32)
        x0 = tensor.from_numpy(data_0)
        x1 = tensor.from_numpy(data_1)
        x2 = tensor.from_numpy(data_2)
        dy = tensor.from_numpy(DY)
        x0.to_device(cpu_dev)
        x1.to_device(cpu_dev)
        x2.to_device(cpu_dev)
        dy.to_device(cpu_dev)

        result = autograd.min(x0, x1, x2)
        dx0, dx1, dx2 = result.creator.backward(dy.data)

        DX0 = np.array([0, 1, 0]).astype(np.float32)
        DX1 = np.array([1, 0, 0]).astype(np.float32)
        DX2 = np.array([0, 0, 1]).astype(np.float32)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx0)),
                                             DX0,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx1)),
                                             DX1,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx2)),
                                             DX2,
                                             decimal=5)

    def test_min_cpu_1inputs(self):
        data_0 = np.array([3, 2, 1]).astype(np.float32)
        XT = np.array([3, 2, 1]).astype(np.float32)

        DY = np.array([1, 1, 1]).astype(np.float32)
        x0 = tensor.from_numpy(data_0)
        dy = tensor.from_numpy(DY)
        x0.to_device(cpu_dev)
        dy.to_device(cpu_dev)

        result = autograd.min(x0)
        dx0 = result.creator.backward(dy.data)

        DX0 = np.array([1, 1, 1]).astype(np.float32)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx0)),
                                             DX0,
                                             decimal=5)

    def test_min_gpu(self):
        X0 = np.array([0.1, 0.2, 2.0, 0.0, 0.1,
                       0.2]).reshape(3, 2).astype(np.float32)
        X1 = np.array([1.0, 2.0, 1.0, 2.1, 0.0,
                       2.0]).reshape(3, 2).astype(np.float32)
        XT = np.minimum(X0, X1)

        DY = np.ones((3, 2), dtype=np.float32)
        x0 = tensor.from_numpy(X0)
        x1 = tensor.from_numpy(X1)
        dy = tensor.from_numpy(DY)
        x0.to_device(gpu_dev)
        x1.to_device(gpu_dev)
        dy.to_device(gpu_dev)

        result = autograd.min(x0, x1)
        dx0, dx1 = result.creator.backward(dy.data)

        G = np.subtract(X0, X1)
        DX0 = np.where(G < 0, 1, G * 0)
        DX1 = np.where(G > 0, 1, G * 0)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx0)),
                                             DX0,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx1)),
                                             DX1,
                                             decimal=5)

    def test_min_gpu_3inputs(self):
        data_0 = np.array([3, 2, 1]).astype(np.float32)
        data_1 = np.array([1, 4, 4]).astype(np.float32)
        data_2 = np.array([2, 5, 0]).astype(np.float32)
        XT = np.array([1, 2, 0]).astype(np.float32)

        DY = np.array([1, 1, 1]).astype(np.float32)
        x0 = tensor.from_numpy(data_0)
        x1 = tensor.from_numpy(data_1)
        x2 = tensor.from_numpy(data_2)
        dy = tensor.from_numpy(DY)
        x0.to_device(gpu_dev)
        x1.to_device(gpu_dev)
        x2.to_device(gpu_dev)
        dy.to_device(gpu_dev)

        result = autograd.min(x0, x1, x2)
        dx0, dx1, dx2 = result.creator.backward(dy.data)

        DX0 = np.array([0, 1, 0]).astype(np.float32)
        DX1 = np.array([1, 0, 0]).astype(np.float32)
        DX2 = np.array([0, 0, 1]).astype(np.float32)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx0)),
                                             DX0,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx1)),
                                             DX1,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx2)),
                                             DX2,
                                             decimal=5)

    def test_min_gpu_1inputs(self):
        data_0 = np.array([3, 2, 1]).astype(np.float32)
        XT = np.array([3, 2, 1]).astype(np.float32)

        DY = np.array([1, 1, 1]).astype(np.float32)
        x0 = tensor.from_numpy(data_0)
        dy = tensor.from_numpy(DY)
        x0.to_device(gpu_dev)
        dy.to_device(gpu_dev)

        result = autograd.min(x0)
        dx0 = result.creator.backward(dy.data)

        DX0 = np.array([1, 1, 1]).astype(np.float32)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx0)),
                                             DX0,
                                             decimal=5)

    def test_HardSigmoid(self):

        def test_helper(gpu=False):
            x = np.random.randn(3, 2)
            #y = max(0, min(1, alpha * x + gamma))
            a = 0.2
            g = 0.5
            y = np.clip(x * 0.2 + 0.5, 0, 1)
            dy = np.random.randn(3, 2)
            grad = (0 < (np.clip(x * 0.2 + 0.5, 0, 1)) *
                    (np.clip(x * 0.2 + 0.5, 0, 1) < 1)) * 0.2 * dy
            x = tensor.from_numpy(x)
            dy = tensor.from_numpy(dy)
            if (gpu):
                x.to_device(gpu_dev)
                dy.to_device(gpu_dev)
            result = autograd.hardsigmoid(x, a, g)
            dx = result.creator.backward(dy.data)
            np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                                 y,
                                                 decimal=5)
            np.testing.assert_array_almost_equal(tensor.to_numpy(
                tensor.from_raw_tensor(dx)),
                                                 grad,
                                                 decimal=5)

        test_helper(False)
        test_helper(True)

    def test_prelu(self):

        def test_helper(gpu):
            x = np.random.randn(3, 2)
            slope = np.random.randn(3, 2)
            y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * slope
            dy = np.random.randn(3, 2)
            x0 = x.copy()
            x0[x0 > 0] = 1
            x0[x0 < 1] = 0
            grad0 = (x0 + (1 - x0) * slope) * dy
            grad1 = (1 - x0) * x * dy
            x = tensor.from_numpy(x)
            slope = tensor.from_numpy(slope)
            dy = tensor.from_numpy(dy)
            if (gpu):
                x.to_device(gpu_dev)
                slope.to_device(gpu_dev)
                dy.to_device(gpu_dev)
            result = autograd.prelu(x, slope)
            dx0, dx1 = result.creator.backward(dy.data)
            np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                                 y,
                                                 decimal=5)
            np.testing.assert_array_almost_equal(tensor.to_numpy(
                tensor.from_raw_tensor(dx0)),
                                                 grad0,
                                                 decimal=5)
            np.testing.assert_array_almost_equal(tensor.to_numpy(
                tensor.from_raw_tensor(dx1)),
                                                 grad1,
                                                 decimal=5)

        test_helper(False)
        if (singa_wrap.USE_CUDA):
            test_helper(True)

    def test_SeLU(self):

        def test_helper(gpu):
            x = np.random.randn(3, 2)
            a = 0.2
            g = 0.3
            y = np.clip(
                x, 0, np.inf) * g + (np.exp(np.clip(x, -np.inf, 0)) - 1) * a * g
            dy = np.random.randn(3, 2)
            grad = (np.exp(np.clip(x, -np.inf, 0))) * g
            grad[x <= 0] = grad[x <= 0] * a
            grad *= dy
            x = tensor.from_numpy(x)

    def test_and_cpu(self):
        x0 = np.array([0, -0.3, -0.1, 0.1, 0.5,
                       0.9]).reshape(3, 2).astype(np.float32)
        x1 = np.array([0, -0.3, 0, 0.1, 0.5, 0.9]).reshape(3,
                                                           2).astype(np.float32)

        y = np.logical_and(x0, x1)
        x0 = tensor.from_numpy(x0)
        x1 = tensor.from_numpy(x1)
        x0.to_device(cpu_dev)
        x1.to_device(cpu_dev)

        result = autograd._and(x0, x1)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)

    def test_and_gpu(self):
        x0 = np.array([0, -0.3, -0.1, 0.1, 0.5,
                       0.9]).reshape(3, 2).astype(np.float32)
        x1 = np.array([0, -0.3, 0, 0.1, 0.5, 0.9]).reshape(3,
                                                           2).astype(np.float32)

        y = np.logical_and(x0, x1)
        x0 = tensor.from_numpy(x0)
        x1 = tensor.from_numpy(x1)
        x0.to_device(gpu_dev)
        x1.to_device(gpu_dev)

        result = autograd._and(x0, x1)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)

    def test_or_cpu(self):
        x0 = np.array([1.0, 1.0, 2.0, -3.0, 0,
                       -7.0]).reshape(3, 2).astype(np.float32)
        x1 = np.array([-1.0, 0, 2.0, 4.0, 0,
                       -7.0]).reshape(3, 2).astype(np.float32)

        y = np.logical_or(x0, x1)
        x0 = tensor.from_numpy(x0)
        x1 = tensor.from_numpy(x1)
        x0.to_device(cpu_dev)
        x1.to_device(cpu_dev)

        result = autograd._or(x0, x1)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)

    def test_or_gpu(self):
        x0 = np.array([1.0, 1.0, 2.0, -3.0, 0,
                       -7.0]).reshape(3, 2).astype(np.float32)
        x1 = np.array([-1.0, 0, 2.0, 4.0, 0,
                       -7.0]).reshape(3, 2).astype(np.float32)

        y = np.logical_or(x0, x1)
        x0 = tensor.from_numpy(x0)
        x1 = tensor.from_numpy(x1)
        x0.to_device(gpu_dev)
        x1.to_device(gpu_dev)

        result = autograd._or(x0, x1)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)

    def test_not_cpu(self):
        x = np.array([1.0, -1.0, 0, -0.1, 0,
                      -7.0]).reshape(3, 2).astype(np.float32)

        y = np.logical_not(x)
        x = tensor.from_numpy(x)
        x.to_device(cpu_dev)

        result = autograd._not(x)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)

    def test_not_gpu(self):
        x = np.array([1.0, -1.0, 0, -0.1, 0,
                      -7.0]).reshape(3, 2).astype(np.float32)

        y = np.logical_not(x)
        x = tensor.from_numpy(x)
        x.to_device(gpu_dev)

        result = autograd._not(x)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)

    def test_xor_cpu(self):
        x0 = np.array([0, -0.3, -0.1, 0.1, 0.5,
                       9.0]).reshape(3, 2).astype(np.float32)
        x1 = np.array([0, -0.3, 0, 0.1, 0, 0.9]).reshape(3,
                                                         2).astype(np.float32)

        y = np.logical_xor(x0, x1)
        x0 = tensor.from_numpy(x0)
        x1 = tensor.from_numpy(x1)
        x0.to_device(cpu_dev)
        x1.to_device(cpu_dev)

        result = autograd._xor(x0, x1)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)

    def test_xor_gpu(self):
        x0 = np.array([0, -0.3, -0.1, 0.1, 0.5,
                       9.0]).reshape(3, 2).astype(np.float32)
        x1 = np.array([0, -0.3, 0, 0.1, 0, 0.9]).reshape(3,
                                                         2).astype(np.float32)

        y = np.logical_xor(x0, x1)
        x0 = tensor.from_numpy(x0)
        x1 = tensor.from_numpy(x1)
        x0.to_device(gpu_dev)
        x1.to_device(gpu_dev)

        result = autograd._xor(x0, x1)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)

    def test_negative_cpu(self):
        X = np.array([0.1, 0, 0.4, 1. - 4, 0.9,
                      -2.0]).reshape(3, 2).astype(np.float32)
        XT = np.negative(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(cpu_dev)
        dy.to_device(cpu_dev)

        result = autograd.negative(x)
        dx = result.creator.backward(dy.data)
        DX = np.negative(DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_negative_gpu(self):
        X = np.array([0.1, 0, 0.4, 1. - 4, 0.9,
                      -2.0]).reshape(3, 2).astype(np.float32)
        XT = np.negative(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(gpu_dev)
        dy.to_device(gpu_dev)

        result = autograd.negative(x)
        dx = result.creator.backward(dy.data)
        DX = np.negative(DY)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_reciprocal_cpu(self):
        X = np.array([0.1, 0, 0.4, 1. - 4, 0.9,
                      -2.0]).reshape(3, 2).astype(np.float32)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(cpu_dev)
        dy.to_device(cpu_dev)

        result = autograd.reciprocal(x)
        dx = result.creator.backward(dy.data)
        #dy/dx = -1/x**2
        with np.errstate(divide='ignore'):
            XT = np.reciprocal(X)
            DX = -1 / np.square(X)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_reciprocal_gpu(self):
        X = np.array([0.1, 0, 0.4, 1. - 4, 0.9,
                      -2.0]).reshape(3, 2).astype(np.float32)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(gpu_dev)
        dy.to_device(gpu_dev)

        result = autograd.reciprocal(x)
        dx = result.creator.backward(dy.data)
        #dy/dx = -1/x**2
        with np.errstate(divide='ignore'):
            XT = np.reciprocal(X)
            DX = -1 / np.square(X)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_and_broadcast_gpu(self):
        dev = gpu_dev
        cases = [
            ([3, 4, 5], [5]),  # 3d vs 1d
            ([3, 4, 5], [4, 5]),  # 3d vs 2d
            ([3, 4, 5, 6], [5, 6]),  # 4d vs 2d
            ([3, 4, 5, 6], [4, 5, 6]),  # 4d vs 3d
            ([1, 4, 1, 6], [3, 1, 5, 6])  # 4d vs 4d
        ]
        for in1, in2 in cases:
            x = (np.random.randn(*in1) > 0).astype(np.float32)
            x1 = (np.random.randn(*in2) > 0).astype(np.float32)
            y = np.logical_and(x, x1)

            x = tensor.from_numpy(x)
            x1 = tensor.from_numpy(x1)
            x.to_device(dev)
            x1.to_device(dev)

            result = autograd._and(x, x1)
            np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                                 y,
                                                 decimal=5)

    def test_and_broadcast_cpu(self):
        dev = cpu_dev
        cases = [
            ([3, 4, 5], [5]),  # 3d vs 1d
            ([3, 4, 5], [4, 5]),  # 3d vs 2d
            ([3, 4, 5, 6], [5, 6]),  # 4d vs 2d
            ([3, 4, 5, 6], [4, 5, 6]),  # 4d vs 3d
            ([1, 4, 1, 6], [3, 1, 5, 6])  # 4d vs 4d
        ]
        for in1, in2 in cases:
            x = (np.random.randn(*in1) > 0).astype(np.float32)
            x1 = (np.random.randn(*in2) > 0).astype(np.float32)
            y = np.logical_and(x, x1)

            x = tensor.from_numpy(x)
            x1 = tensor.from_numpy(x1)
            x.to_device(dev)
            x1.to_device(dev)

            result = autograd._and(x, x1)
            np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                                 y,
                                                 decimal=5)

    def test_or_broadcast_gpu(self):
        dev = gpu_dev
        cases = [
            ([3, 4, 5], [5]),  # 3d vs 1d
            ([3, 4, 5], [4, 5]),  # 3d vs 2d
            ([3, 4, 5, 6], [5, 6]),  # 4d vs 2d
            ([3, 4, 5, 6], [4, 5, 6]),  # 4d vs 3d
            ([1, 4, 1, 6], [3, 1, 5, 6])  # 4d vs 4d
        ]
        for in1, in2 in cases:
            x = (np.random.randn(*in1) > 0).astype(np.float32)
            x1 = (np.random.randn(*in2) > 0).astype(np.float32)
            y = np.logical_or(x, x1)

            x = tensor.from_numpy(x)
            x1 = tensor.from_numpy(x1)
            x.to_device(dev)
            x1.to_device(dev)

            result = autograd._or(x, x1)
            np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                                 y,
                                                 decimal=5)

    def test_or_broadcast_cpu(self):
        dev = cpu_dev
        cases = [
            ([3, 4, 5], [5]),  # 3d vs 1d
            ([3, 4, 5], [4, 5]),  # 3d vs 2d
            ([3, 4, 5, 6], [5, 6]),  # 4d vs 2d
            ([3, 4, 5, 6], [4, 5, 6]),  # 4d vs 3d
            ([1, 4, 1, 6], [3, 1, 5, 6])  # 4d vs 4d
        ]
        for in1, in2 in cases:
            x = (np.random.randn(*in1) > 0).astype(np.float32)
            x1 = (np.random.randn(*in2) > 0).astype(np.float32)
            y = np.logical_or(x, x1)

            x = tensor.from_numpy(x)
            x1 = tensor.from_numpy(x1)
            x.to_device(dev)
            x1.to_device(dev)

            result = autograd._or(x, x1)
            np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                                 y,
                                                 decimal=5)

    def test_xor_broadcast_gpu(self):
        dev = gpu_dev
        cases = [
            ([3, 4, 5], [5]),  # 3d vs 1d
            ([3, 4, 5], [4, 5]),  # 3d vs 2d
            ([3, 4, 5, 6], [5, 6]),  # 4d vs 2d
            ([3, 4, 5, 6], [4, 5, 6]),  # 4d vs 3d
            ([1, 4, 1, 6], [3, 1, 5, 6])  # 4d vs 4d
        ]
        for in1, in2 in cases:
            x = (np.random.randn(*in1) > 0).astype(np.float32)
            x1 = (np.random.randn(*in2) > 0).astype(np.float32)
            y = np.logical_xor(x, x1)

            x = tensor.from_numpy(x)
            x1 = tensor.from_numpy(x1)
            x.to_device(dev)
            x1.to_device(dev)

            result = autograd._xor(x, x1)
            np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                                 y,
                                                 decimal=5)

    def test_xor_broadcast_cpu(self):
        dev = cpu_dev
        cases = [
            ([3, 4, 5], [5]),  # 3d vs 1d
            ([3, 4, 5], [4, 5]),  # 3d vs 2d
            ([3, 4, 5, 6], [5, 6]),  # 4d vs 2d
            ([3, 4, 5, 6], [4, 5, 6]),  # 4d vs 3d
            ([1, 4, 1, 6], [3, 1, 5, 6])  # 4d vs 4d
        ]
        for in1, in2 in cases:
            x = (np.random.randn(*in1) > 0).astype(np.float32)
            x1 = (np.random.randn(*in2) > 0).astype(np.float32)
            y = np.logical_xor(x, x1)

            x = tensor.from_numpy(x)
            x1 = tensor.from_numpy(x1)
            x.to_device(dev)
            x1.to_device(dev)

            result = autograd._xor(x, x1)
            np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                                 y,
                                                 decimal=5)

    def test_greater_broadcast_gpu(self):
        dev = gpu_dev
        cases = [
            ([3, 4, 5], [5]),  # 3d vs 1d
            ([3, 4, 5], [4, 5]),  # 3d vs 2d
            ([3, 4, 5, 6], [5, 6]),  # 4d vs 2d
            ([3, 4, 5, 6], [4, 5, 6]),  # 4d vs 3d
            ([1, 4, 1, 6], [3, 1, 5, 6])  # 4d vs 4d
        ]
        for in1, in2 in cases:
            x = np.random.randn(*in1).astype(np.float32)
            x1 = np.random.randn(*in2).astype(np.float32)
            y = np.greater(x, x1)

            x = tensor.from_numpy(x)
            x1 = tensor.from_numpy(x1)
            x.to_device(dev)
            x1.to_device(dev)

            result = autograd.greater(x, x1)
            np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                                 y,
                                                 decimal=5)

    def test_greater_broadcast_cpu(self):
        dev = cpu_dev
        cases = [
            ([3, 4, 5], [5]),  # 3d vs 1d
            ([3, 4, 5], [4, 5]),  # 3d vs 2d
            ([3, 4, 5, 6], [5, 6]),  # 4d vs 2d
            ([3, 4, 5, 6], [4, 5, 6]),  # 4d vs 3d
            ([1, 4, 1, 6], [3, 1, 5, 6])  # 4d vs 4d
        ]
        for in1, in2 in cases:
            x = np.random.randn(*in1).astype(np.float32)
            x1 = np.random.randn(*in2).astype(np.float32)
            y = np.greater(x, x1)

            x = tensor.from_numpy(x)
            x1 = tensor.from_numpy(x1)
            x.to_device(dev)
            x1.to_device(dev)

            result = autograd.greater(x, x1)
            np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                                 y,
                                                 decimal=5)

    def test_less_broadcast_cpu(self):
        dev = cpu_dev
        cases = [
            ([3, 4, 5], [5]),  # 3d vs 1d
            ([3, 4, 5], [4, 5]),  # 3d vs 2d
            ([3, 4, 5, 6], [5, 6]),  # 4d vs 2d
            ([3, 4, 5, 6], [4, 5, 6]),  # 4d vs 3d
            ([1, 4, 1, 6], [3, 1, 5, 6])  # 4d vs 4d
        ]
        for in1, in2 in cases:
            x = np.random.randn(*in1).astype(np.float32)
            x1 = np.random.randn(*in2).astype(np.float32)
            y = np.less(x, x1)

            x = tensor.from_numpy(x)
            x1 = tensor.from_numpy(x1)
            x.to_device(dev)
            x1.to_device(dev)

            result = autograd.less(x, x1)
            np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                                 y,
                                                 decimal=5)

    def test_less_broadcast_gpu(self):
        dev = gpu_dev
        cases = [
            ([3, 4, 5], [5]),  # 3d vs 1d
            ([3, 4, 5], [4, 5]),  # 3d vs 2d
            ([3, 4, 5, 6], [5, 6]),  # 4d vs 2d
            ([3, 4, 5, 6], [4, 5, 6]),  # 4d vs 3d
            ([1, 4, 1, 6], [3, 1, 5, 6])  # 4d vs 4d
        ]
        for in1, in2 in cases:
            x = np.random.randn(*in1).astype(np.float32)
            x1 = np.random.randn(*in2).astype(np.float32)
            y = np.less(x, x1)

            x = tensor.from_numpy(x)
            x1 = tensor.from_numpy(x1)
            x.to_device(dev)
            x1.to_device(dev)

            result = autograd.less(x, x1)
            np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                                 y,
                                                 decimal=5)

    def test_add_broadcast_gpu(self):
        dev = gpu_dev
        cases = [
            ([3, 4, 5], [5]),  # 3d vs 1d
            ([3, 4, 5], [4, 5]),  # 3d vs 2d
            ([3, 4, 5, 6], [5, 6]),  # 4d vs 2d
            ([3, 4, 5, 6], [4, 5, 6]),  # 4d vs 3d
            ([1, 4, 1, 6], [3, 1, 5, 6])  # 4d vs 4d
        ]
        for in1, in2 in cases:
            x = np.random.randn(*in1).astype(np.float32)
            x1 = np.random.randn(*in2).astype(np.float32)
            y = x + x1

            dy = np.random.randn(*y.shape)
            grad0 = np.sum(dy, axis=axis_helper(y.shape,
                                                x.shape)).reshape(x.shape)
            grad1 = np.sum(dy, axis=axis_helper(y.shape,
                                                x1.shape)).reshape(x1.shape)

            x = tensor.from_numpy(x)
            x1 = tensor.from_numpy(x1)
            dy = tensor.from_numpy(dy)
            x.to_device(dev)
            x1.to_device(dev)
            dy.to_device(dev)

            result = autograd.add(x, x1)
            dx0, dx1 = result.creator.backward(dy.data)
            np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                                 y,
                                                 decimal=5)
            np.testing.assert_array_almost_equal(tensor.to_numpy(
                tensor.from_raw_tensor(dx0)),
                                                 grad0,
                                                 decimal=5)
            np.testing.assert_array_almost_equal(tensor.to_numpy(
                tensor.from_raw_tensor(dx1)),
                                                 grad1,
                                                 decimal=5)

    def test_add_broadcast_cpu(self):
        dev = cpu_dev
        cases = [
            ([3, 4, 5], [5]),  # 3d vs 1d
            ([3, 4, 5], [4, 5]),  # 3d vs 2d
            ([3, 4, 5, 6], [5, 6]),  # 4d vs 2d
            ([3, 4, 5, 6], [4, 5, 6]),  # 4d vs 3d
            ([1, 4, 1, 6], [3, 1, 5, 6])  # 4d vs 4d
        ]
        for in1, in2 in cases:
            x = np.random.randn(*in1).astype(np.float32)
            x1 = np.random.randn(*in2).astype(np.float32)
            y = x + x1

            dy = np.random.randn(*y.shape)
            grad0 = np.sum(dy, axis=axis_helper(y.shape,
                                                x.shape)).reshape(x.shape)
            grad1 = np.sum(dy, axis=axis_helper(y.shape,
                                                x1.shape)).reshape(x1.shape)

            x = tensor.from_numpy(x)
            x1 = tensor.from_numpy(x1)
            dy = tensor.from_numpy(dy)
            x.to_device(dev)
            x1.to_device(dev)
            dy.to_device(dev)

            result = autograd.add(x, x1)
            dx0, dx1 = result.creator.backward(dy.data)
            np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                                 y,
                                                 decimal=5)
            np.testing.assert_array_almost_equal(tensor.to_numpy(
                tensor.from_raw_tensor(dx0)),
                                                 grad0,
                                                 decimal=5)
            np.testing.assert_array_almost_equal(tensor.to_numpy(
                tensor.from_raw_tensor(dx1)),
                                                 grad1,
                                                 decimal=5)

    def test_sub_broadcast_gpu(self):
        dev = gpu_dev
        cases = [
            ([3, 4, 5], [5]),  # 3d vs 1d
            ([3, 4, 5], [4, 5]),  # 3d vs 2d
            ([3, 4, 5, 6], [5, 6]),  # 4d vs 2d
            ([3, 4, 5, 6], [4, 5, 6]),  # 4d vs 3d
            ([1, 4, 1, 6], [3, 1, 5, 6])  # 4d vs 4d
        ]
        for in1, in2 in cases:
            x = np.random.randn(*in1).astype(np.float32)
            x1 = np.random.randn(*in2).astype(np.float32)
            y = x - x1

            dy = np.random.randn(*y.shape)
            grad0 = np.sum(dy, axis=axis_helper(y.shape,
                                                x.shape)).reshape(x.shape)
            grad1 = np.sum(-dy, axis=axis_helper(y.shape,
                                                 x1.shape)).reshape(x1.shape)

            x = tensor.from_numpy(x)
            x1 = tensor.from_numpy(x1)
            dy = tensor.from_numpy(dy)
            x.to_device(dev)
            x1.to_device(dev)
            dy.to_device(dev)

            result = autograd.sub(x, x1)
            dx0, dx1 = result.creator.backward(dy.data)
            np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                                 y,
                                                 decimal=5)
            np.testing.assert_array_almost_equal(tensor.to_numpy(
                tensor.from_raw_tensor(dx0)),
                                                 grad0,
                                                 decimal=5)
            np.testing.assert_array_almost_equal(tensor.to_numpy(
                tensor.from_raw_tensor(dx1)),
                                                 grad1,
                                                 decimal=5)

    def test_sub_broadcast_cpu(self):
        dev = cpu_dev
        cases = [
            ([3, 4, 5], [5]),  # 3d vs 1d
            ([3, 4, 5], [4, 5]),  # 3d vs 2d
            ([3, 4, 5, 6], [5, 6]),  # 4d vs 2d
            ([3, 4, 5, 6], [4, 5, 6]),  # 4d vs 3d
            ([1, 4, 1, 6], [3, 1, 5, 6])  # 4d vs 4d
        ]
        for in1, in2 in cases:
            x = np.random.randn(*in1).astype(np.float32)
            x1 = np.random.randn(*in2).astype(np.float32)
            y = x - x1

            dy = np.random.randn(*y.shape)
            grad0 = np.sum(dy, axis=axis_helper(y.shape,
                                                x.shape)).reshape(x.shape)
            grad1 = np.sum(-dy, axis=axis_helper(y.shape,
                                                 x1.shape)).reshape(x1.shape)

            x = tensor.from_numpy(x)
            x1 = tensor.from_numpy(x1)
            dy = tensor.from_numpy(dy)
            x.to_device(dev)
            x1.to_device(dev)
            dy.to_device(dev)

            result = autograd.sub(x, x1)
            dx0, dx1 = result.creator.backward(dy.data)
            np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                                 y,
                                                 decimal=5)
            np.testing.assert_array_almost_equal(tensor.to_numpy(
                tensor.from_raw_tensor(dx0)),
                                                 grad0,
                                                 decimal=5)
            np.testing.assert_array_almost_equal(tensor.to_numpy(
                tensor.from_raw_tensor(dx1)),
                                                 grad1,
                                                 decimal=5)

    def test_mul_broadcast_gpu(self):
        dev = gpu_dev
        cases = [
            ([3, 4, 5], [5]),  # 3d vs 1d
            ([3, 4, 5], [4, 5]),  # 3d vs 2d
            ([3, 4, 5, 6], [5, 6]),  # 4d vs 2d
            ([3, 4, 5, 6], [4, 5, 6]),  # 4d vs 3d
            ([1, 4, 1, 6], [3, 1, 5, 6])  # 4d vs 4d
        ]
        for in1, in2 in cases:
            x = np.random.randn(*in1).astype(np.float32)
            x1 = np.random.randn(*in2).astype(np.float32)
            y = x * x1

            dy = np.random.randn(*y.shape)
            grad0 = np.sum(x1 * dy, axis=axis_helper(y.shape,
                                                     x.shape)).reshape(x.shape)
            grad1 = np.sum(x * dy, axis=axis_helper(y.shape,
                                                    x1.shape)).reshape(x1.shape)

            x = tensor.from_numpy(x)
            x1 = tensor.from_numpy(x1)
            dy = tensor.from_numpy(dy)
            x.to_device(dev)
            x1.to_device(dev)
            dy.to_device(dev)

            result = autograd.mul(x, x1)
            dx0, dx1 = result.creator.backward(dy.data)
            np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                                 y,
                                                 decimal=5)
            np.testing.assert_array_almost_equal(tensor.to_numpy(
                tensor.from_raw_tensor(dx0)),
                                                 grad0,
                                                 decimal=5)
            np.testing.assert_array_almost_equal(tensor.to_numpy(
                tensor.from_raw_tensor(dx1)),
                                                 grad1,
                                                 decimal=5)

    def test_mul_broadcast_cpu(self):
        dev = cpu_dev
        cases = [
            ([3, 4, 5], [5]),  # 3d vs 1d
            ([3, 4, 5], [4, 5]),  # 3d vs 2d
            ([3, 4, 5, 6], [5, 6]),  # 4d vs 2d
            ([3, 4, 5, 6], [4, 5, 6]),  # 4d vs 3d
            ([1, 4, 1, 6], [3, 1, 5, 6])  # 4d vs 4d
        ]
        for in1, in2 in cases:
            x = np.random.randn(*in1).astype(np.float32)
            x1 = np.random.randn(*in2).astype(np.float32)
            y = x * x1

            dy = np.random.randn(*y.shape)
            grad0 = np.sum(x1 * dy, axis=axis_helper(y.shape,
                                                     x.shape)).reshape(x.shape)
            grad1 = np.sum(x * dy, axis=axis_helper(y.shape,
                                                    x1.shape)).reshape(x1.shape)

            x = tensor.from_numpy(x)
            x1 = tensor.from_numpy(x1)
            dy = tensor.from_numpy(dy)
            x.to_device(dev)
            x1.to_device(dev)
            dy.to_device(dev)

            result = autograd.mul(x, x1)
            dx0, dx1 = result.creator.backward(dy.data)
            np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                                 y,
                                                 decimal=5)
            np.testing.assert_array_almost_equal(tensor.to_numpy(
                tensor.from_raw_tensor(dx0)),
                                                 grad0,
                                                 decimal=5)
            np.testing.assert_array_almost_equal(tensor.to_numpy(
                tensor.from_raw_tensor(dx1)),
                                                 grad1,
                                                 decimal=5)

    def test_div_broadcast_gpu(self):
        dev = gpu_dev
        cases = [
            ([3, 4, 5], [5]),  # 3d vs 1d
            ([3, 4, 5], [4, 5]),  # 3d vs 2d
            ([3, 4, 5, 6], [5, 6]),  # 4d vs 2d
            ([3, 4, 5, 6], [4, 5, 6]),  # 4d vs 3d
            ([1, 4, 1, 6], [3, 1, 5, 6])  # 4d vs 4d
        ]
        for in1, in2 in cases:
            x = np.random.randn(*in1).astype(np.float32)
            x1 = np.random.randn(*in2).astype(np.float32) + 1.0
            y = x / x1

            dy = np.random.randn(*y.shape).astype(np.float32)
            grad0 = np.sum(np.power(x1, -1) * dy,
                           axis=axis_helper(y.shape, x.shape)).reshape(x.shape)
            grad1 = np.sum(x * -np.power(x1, -2) * dy,
                           axis=axis_helper(y.shape,
                                            x1.shape)).reshape(x1.shape)

            x = tensor.from_numpy(x)
            x1 = tensor.from_numpy(x1)
            dy = tensor.from_numpy(dy)
            x.to_device(dev)
            x1.to_device(dev)
            dy.to_device(dev)

            result = autograd.div(x, x1)
            dx0, dx1 = result.creator.backward(dy.data)
            np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                                 y,
                                                 decimal=2)
            np.testing.assert_array_almost_equal(tensor.to_numpy(
                tensor.from_raw_tensor(dx0)),
                                                 grad0,
                                                 decimal=2)
            np.testing.assert_array_almost_equal(tensor.to_numpy(
                tensor.from_raw_tensor(dx1)),
                                                 grad1,
                                                 decimal=2)
            break

    def test_div_broadcast_cpu(self):
        dev = cpu_dev
        cases = [
            ([3, 4, 5], [5]),  # 3d vs 1d
            ([3, 4, 5], [4, 5]),  # 3d vs 2d
            ([3, 4, 5, 6], [5, 6]),  # 4d vs 2d
            ([3, 4, 5, 6], [4, 5, 6]),  # 4d vs 3d
            ([1, 4, 1, 6], [3, 1, 5, 6])  # 4d vs 4d
        ]
        for in1, in2 in cases:
            x = np.random.randn(*in1).astype(np.float32)
            x1 = np.random.randn(*in2).astype(np.float32) + 1.0
            y = x / x1

            dy = np.random.randn(*y.shape).astype(np.float32)
            grad0 = np.sum(np.power(x1, -1) * dy,
                           axis=axis_helper(y.shape, x.shape)).reshape(x.shape)
            grad1 = np.sum(x * -np.power(x1, -2) * dy,
                           axis=axis_helper(y.shape,
                                            x1.shape)).reshape(x1.shape)

            x = tensor.from_numpy(x)
            x1 = tensor.from_numpy(x1)
            dy = tensor.from_numpy(dy)
            x.to_device(dev)
            x1.to_device(dev)
            dy.to_device(dev)

            result = autograd.div(x, x1)
            dx0, dx1 = result.creator.backward(dy.data)
            np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                                 y,
                                                 decimal=2)
            np.testing.assert_array_almost_equal(tensor.to_numpy(
                tensor.from_raw_tensor(dx0)),
                                                 grad0,
                                                 decimal=2)
            np.testing.assert_array_almost_equal(tensor.to_numpy(
                tensor.from_raw_tensor(dx1)),
                                                 grad1,
                                                 decimal=2)

    def test_pow_broadcast_gpu(self):
        dev = gpu_dev
        cases = [
            ([3, 4, 5], [5]),  # 3d vs 1d
            ([3, 4, 5], [4, 5]),  # 3d vs 2d
            ([3, 4, 5, 6], [5, 6]),  # 4d vs 2d
            ([3, 4, 5, 6], [4, 5, 6]),  # 4d vs 3d
            ([1, 4, 1, 6], [3, 1, 5, 6])  # 4d vs 4d
        ]
        for in1, in2 in cases:
            x = np.random.randint(1, 10, size=in1).astype(np.float32)
            x1 = np.random.randint(1, 5, size=in2).astype(np.float32)
            y = np.power(x, x1).astype(np.float32)

            dy = np.random.randn(*y.shape).astype(np.float32)
            grad0 = np.sum(x1 * np.power(x, x1 - 1) * dy,
                           axis=axis_helper(y.shape, x.shape)).reshape(x.shape)
            grad1 = np.sum(np.power(x, x1) * np.log(x) * dy,
                           axis=axis_helper(y.shape,
                                            x1.shape)).reshape(x1.shape)

            x = tensor.from_numpy(x)
            x1 = tensor.from_numpy(x1)
            dy = tensor.from_numpy(dy)
            x.to_device(dev)
            x1.to_device(dev)
            dy.to_device(dev)

            result = autograd.pow(x, x1)
            dx0, dx1 = result.creator.backward(dy.data)
            np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                                 y,
                                                 decimal=2)
            np.testing.assert_array_almost_equal(tensor.to_numpy(
                tensor.from_raw_tensor(dx0)),
                                                 grad0,
                                                 decimal=2)
            np.testing.assert_array_almost_equal(tensor.to_numpy(
                tensor.from_raw_tensor(dx1)),
                                                 grad1,
                                                 decimal=2)

    def test_pow_broadcast_cpu(self):
        dev = cpu_dev
        cases = [
            ([3, 4, 5], [5]),  # 3d vs 1d
            ([3, 4, 5], [4, 5]),  # 3d vs 2d
            ([3, 4, 5, 6], [5, 6]),  # 4d vs 2d
            ([3, 4, 5, 6], [4, 5, 6]),  # 4d vs 3d
            ([1, 4, 1, 6], [3, 1, 5, 6])  # 4d vs 4d
        ]
        for in1, in2 in cases:
            x = np.random.randint(1, 10, size=in1).astype(np.float32)
            x1 = np.random.randint(1, 5, size=in2).astype(np.float32)
            y = np.power(x, x1).astype(np.float32)

            dy = np.random.randn(*y.shape).astype(np.float32)
            grad0 = np.sum(x1 * np.power(x, x1 - 1) * dy,
                           axis=axis_helper(y.shape, x.shape)).reshape(x.shape)
            grad1 = np.sum(np.power(x, x1) * np.log(x) * dy,
                           axis=axis_helper(y.shape,
                                            x1.shape)).reshape(x1.shape)

            x = tensor.from_numpy(x)
            x1 = tensor.from_numpy(x1)
            dy = tensor.from_numpy(dy)
            x.to_device(dev)
            x1.to_device(dev)
            dy.to_device(dev)

            result = autograd.pow(x, x1)
            dx0, dx1 = result.creator.backward(dy.data)
            np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                                 y,
                                                 decimal=2)
            np.testing.assert_array_almost_equal(tensor.to_numpy(
                tensor.from_raw_tensor(dx0)),
                                                 grad0,
                                                 decimal=2)
            np.testing.assert_array_almost_equal(tensor.to_numpy(
                tensor.from_raw_tensor(dx1)),
                                                 grad1,
                                                 decimal=2)

    def test_prelu_broadcast_gpu(self):
        dev = gpu_dev
        cases = [
            ([3, 4, 5], [5]),  # 3d vs 1d
            ([3, 4, 5], [4, 5]),  # 3d vs 2d
            ([3, 4, 5, 6], [5, 6]),  # 4d vs 2d
            ([3, 4, 5, 6], [4, 5, 6]),  # 4d vs 3d
            ([1, 4, 1, 6], [3, 1, 5, 6])  # 4d vs 4d
        ]
        for in1, in2 in cases:
            x = np.random.randn(*in1).astype(np.float32)
            slope = np.random.randn(*in2).astype(np.float32)
            y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * slope

            dy = np.random.randn(*y.shape).astype(np.float32)
            x0 = x.copy()
            x0[x0 > 0] = 1
            x0[x0 < 1] = 0
            grad0 = np.sum((x0 + (1 - x0) * slope) * dy,
                           axis=axis_helper(y.shape, x.shape)).reshape(x.shape)
            grad1 = np.sum((1 - x0) * x * dy,
                           axis=axis_helper(y.shape,
                                            slope.shape)).reshape(slope.shape)

            x = tensor.from_numpy(x)
            slope = tensor.from_numpy(slope)
            dy = tensor.from_numpy(dy)
            x.to_device(dev)
            slope.to_device(dev)
            dy.to_device(dev)

            result = autograd.prelu(x, slope)
            dx0, dx1 = result.creator.backward(dy.data)
            np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                                 y,
                                                 decimal=5)
            np.testing.assert_array_almost_equal(tensor.to_numpy(
                tensor.from_raw_tensor(dx0)),
                                                 grad0,
                                                 decimal=5)
            np.testing.assert_array_almost_equal(tensor.to_numpy(
                tensor.from_raw_tensor(dx1)),
                                                 grad1,
                                                 decimal=5)

    def test_prelu_broadcast_cpu(self):
        dev = cpu_dev
        cases = [
            ([3, 4, 5], [5]),  # 3d vs 1d
            ([3, 4, 5], [4, 5]),  # 3d vs 2d
            ([3, 4, 5, 6], [5, 6]),  # 4d vs 2d
            ([3, 4, 5, 6], [4, 5, 6]),  # 4d vs 3d
            ([1, 4, 1, 6], [3, 1, 5, 6])  # 4d vs 4d
        ]
        for in1, in2 in cases:
            x = np.random.randn(*in1).astype(np.float32)
            slope = np.random.randn(*in2).astype(np.float32)
            y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * slope

            dy = np.random.randn(*y.shape).astype(np.float32)
            x0 = x.copy()
            x0[x0 > 0] = 1
            x0[x0 < 1] = 0
            grad0 = np.sum((x0 + (1 - x0) * slope) * dy,
                           axis=axis_helper(y.shape, x.shape)).reshape(x.shape)
            grad1 = np.sum((1 - x0) * x * dy,
                           axis=axis_helper(y.shape,
                                            slope.shape)).reshape(slope.shape)

            x = tensor.from_numpy(x)
            slope = tensor.from_numpy(slope)
            dy = tensor.from_numpy(dy)
            x.to_device(dev)
            slope.to_device(dev)
            dy.to_device(dev)

            result = autograd.prelu(x, slope)
            dx0, dx1 = result.creator.backward(dy.data)
            np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                                 y,
                                                 decimal=5)
            np.testing.assert_array_almost_equal(tensor.to_numpy(
                tensor.from_raw_tensor(dx0)),
                                                 grad0,
                                                 decimal=5)
            np.testing.assert_array_almost_equal(tensor.to_numpy(
                tensor.from_raw_tensor(dx1)),
                                                 grad1,
                                                 decimal=5)


    def gemm_test(self, dev):
        configs = [
            # alpha, beta, transA, transB, shapeA, shapeB, shapeC, shapeY
            [0.25, 0.35, 0, 0, (3, 4), (4, 5), (1, 5), (3, 5)],
            [0.25, 0.35, 0, 1, (3, 4), (5, 4), (1, 5), (3, 5)],
            [0.25, 0.35, 1, 0, (4, 3), (4, 5), (1, 5), (3, 5)],
            [0.25, 0.35, 1, 1, (4, 3), (5, 4), (1, 5), (3, 5)],
        ]
        for config in configs:
            alpha = config[0]
            beta = config[1]
            transA = config[2]
            transB = config[3]
            shapeA = config[4]
            shapeB = config[5]
            shapeC = config[6]
            shapeY = config[7]
            A = np.random.randn(*shapeA).astype(np.float32)
            B = np.random.randn(*shapeB).astype(np.float32)
            C = np.random.randn(*shapeC).astype(np.float32)
            DY = np.ones(shapeY, dtype=np.float32)

            a = tensor.from_numpy(A)
            a.to_device(dev)
            b = tensor.from_numpy(B)
            b.to_device(dev)
            c = tensor.from_numpy(C)
            c.to_device(dev)
            dy = tensor.from_numpy(DY)
            dy.to_device(dev)

            result = autograd.gemm(a, b, c, alpha, beta, transA, transB)
            da, db, dc = result.creator.backward(dy.data)

            # Y = alpha * A' * B' + beta * C
            _A = A if transA == 0 else A.T
            _B = B if transB == 0 else B.T
            C = C if C is not None else np.array(0)
            Y = alpha * np.dot(_A, _B) + beta * C

            DA = alpha * np.matmul(DY, _B.T)
            DA = DA if transA == 0 else DA.T
            DB = alpha * np.matmul(_A.T, DY)
            DB = DB if transB == 0 else DB.T
            DC = beta * np.sum(DY, axis=axis_helper(Y.shape, C.shape)).reshape(
                C.shape)

            np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                                 Y,
                                                 decimal=5)
            np.testing.assert_array_almost_equal(tensor.to_numpy(
                tensor.from_raw_tensor(da)),
                                                 DA,
                                                 decimal=5)
            np.testing.assert_array_almost_equal(tensor.to_numpy(
                tensor.from_raw_tensor(db)),
                                                 DB,
                                                 decimal=5)
            np.testing.assert_array_almost_equal(tensor.to_numpy(
                tensor.from_raw_tensor(dc)),
                                                 DC,
                                                 decimal=5)

    def test_gemm_cpu(self):
        self.gemm_test(cpu_dev)

    def test_gemm_gpu(self):
        self.gemm_test(gpu_dev)

    def globalaveragepool_channel_first(self, dev):
        X = np.array([[[
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]]]).astype(np.float32)
        XT = np.array([[[[5]]]]).astype(np.float32)
        DY = np.ones((1, 1, 1, 1), dtype=np.float32)

        x = tensor.from_numpy(X)
        x.to_device(dev)
        dy = tensor.from_numpy(DY)
        dy.to_device(dev)

        result = autograd.globalaveragepool(x)
        dx = result.creator.backward(dy.data)

        DX = np.ones(X.shape, dtype=np.float32)
        DX = np.multiply(DX, DY) / np.prod(X.shape[2:])

        np.testing.assert_array_almost_equal(tensor.to_numpy(result), XT, decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(tensor.from_raw_tensor(dx)), DX, decimal=5)

    def globalaveragepool_channel_last(self, dev):
        X = np.array([[
            [[1], [2], [3]],
            [[4], [5], [6]],
            [[7], [8], [9]],
        ]]).astype(np.float32)
        XT = np.array([[[[5]]]]).astype(np.float32)
        DY = np.ones((1, 1, 1, 1), dtype=np.float32)

        x = tensor.from_numpy(X)
        x.to_device(dev)
        dy = tensor.from_numpy(DY)
        dy.to_device(dev)

        result = autograd.globalaveragepool(x, 'channel_last')
        dx = result.creator.backward(dy.data)

        DX = np.ones(X.shape, dtype=np.float32)
        DX = np.multiply(DX, DY) / np.prod(X.shape[1:-1])

        np.testing.assert_array_almost_equal(tensor.to_numpy(result), XT, decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(tensor.from_raw_tensor(dx)), DX, decimal=5)

    def test_globalaveragepool_cpu(self):
        self.globalaveragepool_channel_first(cpu_dev)
        self.globalaveragepool_channel_last(cpu_dev)

    def test_globalaveragepool_gpu(self):
        self.globalaveragepool_channel_first(gpu_dev)
        self.globalaveragepool_channel_last(gpu_dev)
    def constantOfShape_test(self, dev):
        # float_ones
        X = np.array([4, 3, 2]).astype(np.int64)
        x = tensor.from_numpy(X)
        x.to_device(dev)

        y = np.ones(X, dtype=np.float32)
        result = autograd.constant_of_shape(x, 1.0)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)
        # int32_zeros
        X = np.array([10, 6]).astype(np.int64)
        x = tensor.from_numpy(X)
        x.to_device(dev)

        y = np.ones(X, dtype=np.int32)
        result = autograd.constant_of_shape(x, 1)
        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)

    def test_constantOfShape_cpu(self):
        self.constantOfShape_test(cpu_dev)

    def test_constantOfShape_gpu(self):
        self.constantOfShape_test(gpu_dev)

    def dropout_test(self, dev):
        X = np.random.randn(3, 4, 5).astype(np.float32)
        dy = np.random.randn(3, 4, 5).astype(np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(dy)
        x.to_device(dev)
        dy.to_device(dev)

        result = autograd.dropout(x, 0.5)
        dx = result.creator.backward(dy.data)
        self.check_shape(result.shape, (3, 4, 5))
        self.check_shape(dx.shape(), (3, 4, 5))

    def test_dropout_cpu(self):
        self.dropout_test(cpu_dev)

    def test_dropout_gpu(self):
        self.dropout_test(gpu_dev)

    def reduceSum_test(self, dev):
        shape = [3, 2, 2]
        cases = [
            (None, 1),
            ([1], 0),
            ([1], 1),
            ([-2], 1),
            ([1, 2], 1)
        ]
        for axes, keepdims in cases:
            X = np.random.uniform(-10, 10, shape).astype(np.float32)
            _axes = tuple(axes) if axes is not None else None
            y = np.sum(X, axis=_axes, keepdims=keepdims == 1)
            dy = np.random.randn(*y.shape).astype(np.float32)

            x = tensor.from_numpy(X)
            dy = tensor.from_numpy(dy)
            x.to_device(dev)
            dy.to_device(dev)

            result = autograd.reduce_sum(x, axes, keepdims)
            dx = result.creator.backward(dy.data)

            np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                                y,
                                                decimal=5)
            self.check_shape(dx.shape(), tuple(shape))

    def test_reduceSum_cpu(self):
        self.reduceSum_test(cpu_dev)

    def test_reduceSum_gpu(self):
        self.reduceSum_test(gpu_dev)

    def reduceMean_test(self, dev):
        shape = [3, 2, 2]
        cases = [
            (None, 1),
            ([1], 0),
            ([1], 1),
            ([-2], 1),
            ([1, 2], 1)
        ]
        for axes, keepdims in cases:
            X = np.random.uniform(-10, 10, shape).astype(np.float32)
            _axes = tuple(axes) if axes is not None else None
            y = np.mean(X, axis=_axes, keepdims=keepdims == 1)
            dy = np.random.randn(*y.shape).astype(np.float32)

            x = tensor.from_numpy(X)
            dy = tensor.from_numpy(dy)
            x.to_device(dev)
            dy.to_device(dev)

            result = autograd.reduce_mean(x, axes, keepdims)
            dx = result.creator.backward(dy.data)

            np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                                y,
                                                decimal=5)
            self.check_shape(dx.shape(), tuple(shape))

    def test_reduceMean_cpu(self):
        self.reduceMean_test(cpu_dev)

    def test_reduceMean_gpu(self):
        self.reduceMean_test(gpu_dev)

    def slice_test(self, dev):
        X = np.random.randn(20, 10, 5).astype(np.float32)
        indexes = np.array(range(20*10*5)).reshape(20, 10, 5)
        configs = [
            # starts, ends, axes, steps, y
            [[0, 0], [3, 10], [0, 1], [1, 1], X[0:3, 0:10], indexes[0:3, 0:10]],  # slice
            [[0, 0, 3], [20, 10, 4], None, None, X[:, :, 3:4], indexes[:, :, 3:4]],  # slice_default_axes
            [[1], [1000], [1], [1], X[:, 1:1000], indexes[:, 1:1000]],  # slice_end_out_of_bounds
            [[0], [-1], [1], [1], X[:, 0:-1], indexes[:, 0:-1]],  # slice_end_out_of_bounds
            [[20, 10, 4], [0, 0, 1], [0, 1, 2], [-1, -3, -2], X[20:0:-1, 10:0:-3, 4:1:-2],
                indexes[20:0:-1, 10:0:-3, 4:1:-2]],  # slice_neg_steps
            [[0, 0, 3], [20, 10, 4], [0, -2, -1], None, X[:, :, 3:4], indexes[:, :, 3:4]],  # slice_negative_axes
            # [[1000], [1000], [1], [1], X[:, 1000:1000], indexes[:, 1000:1000]], # slice_start_out_of_bounds # cannot support empty tensor
        ]
        for starts, ends, axes, steps, y, dx_idx in configs:
            dy = np.ones(y.shape).astype(np.float32)

            x = tensor.from_numpy(X)
            dy = tensor.from_numpy(dy)
            x.to_device(dev)
            dy.to_device(dev)

            result = autograd.slice(x, starts, ends, axes, steps)
            dx = result.creator.backward(dy.data)

            dx_idx = tuple(dx_idx.flatten().tolist())
            dX = np.array([1. if i in dx_idx else 0. for i in range(20*10*5)]).reshape(X.shape)

            np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                                y,
                                                decimal=5)
            np.testing.assert_array_almost_equal(tensor.to_numpy(tensor.from_raw_tensor(dx)),
                                                dX,
                                                decimal=5)

    def test_slice_cpu(self):
        self.slice_test(cpu_dev)

    def test_slice_gpu(self):
        self.slice_test(gpu_dev)

    def ceil_test(self, dev):
        X = np.array([-1.5, 1.2]).astype(np.float32)
        DY = np.ones((2), dtype=np.float32)
        y = np.ceil(X)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(dev)
        dy.to_device(dev)

        result = autograd.ceil(x)
        dx = result.creator.backward(dy.data)
        DX = np.zeros((2), dtype=np.float32)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_ceil_cpu(self):
        self.ceil_test(cpu_dev)

    def test_ceil_gpu(self):
        self.ceil_test(gpu_dev)

    def split_test(self, dev):
        X = np.array([1., 2., 3., 4., 5., 6.]).astype(np.float32)
        DY1 = np.ones((2), dtype=np.float32)
        DY2 = np.ones((4), dtype=np.float32)
        y = [
            np.array([1., 2.]).astype(np.float32),
            np.array([3., 4., 5., 6.]).astype(np.float32)
        ]

        x = tensor.from_numpy(X)
        dy1 = tensor.from_numpy(DY1)
        dy2 = tensor.from_numpy(DY2)
        x.to_device(dev)
        dy1.to_device(dev)
        dy2.to_device(dev)

        result = autograd.split(x, 0, (2, 4))
        dx = result[0].creator.backward(dy1.data, dy2.data)
        DX = np.ones((6), dtype=np.float32)

        for idx, _r in enumerate(result):
            np.testing.assert_array_almost_equal(
                tensor.to_numpy(_r), y[idx], decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_split_cpu(self):
        self.split_test(cpu_dev)

    def test_split_gpu(self):
        self.split_test(gpu_dev)

    def gather_test(self, dev):
        config = [
            ([0, 1, 3], 0), ([0, 1, 3], 1),
                  ([[0, 1], [1, 2], [2, 3]], 1),
                  ([0, -1, -2], 0)
                  ]  # (indices, axis)
        for indices, _axis in config:
            X = np.random.randn(5, 4, 3, 2).astype(np.float32)
            y = np.take(X, indices, axis=_axis)
            DY = np.ones(y.shape, dtype=np.float32)

            x = tensor.from_numpy(X)
            dy = tensor.from_numpy(DY)
            x.to_device(dev)
            dy.to_device(dev)

            result = autograd.gather(x, _axis, indices)
            dx = result.creator.backward(dy.data)

            np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                                 y,
                                                 decimal=5)
            self.check_shape(dx.shape(), tuple(X.shape))

    def test_gather_cpu(self):
        self.gather_test(cpu_dev)

    def test_gather_gpu(self):
        self.gather_test(gpu_dev)

    def tile_test(self, dev):
        config_repeats = [
            2,
            [2, 2],
            [2, 1, 2],
        ]
        for repeats in config_repeats:
            X = np.array([0, 1, 2]).astype(np.float32)
            y = np.tile(X, repeats)
            DY = np.copy(y)

            x = tensor.from_numpy(X)
            dy = tensor.from_numpy(DY)
            x.to_device(dev)
            dy.to_device(dev)

            result = autograd.tile(x, repeats)
            dx = result.creator.backward(dy.data)
            DX = np.multiply(X, np.prod(repeats))
            np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                                 y,
                                                 decimal=5)
            np.testing.assert_array_almost_equal(tensor.to_numpy(
                tensor.from_raw_tensor(dx)),
                                                 DX,
                                                 decimal=5)

    def test_tile_cpu(self):
        self.tile_test(cpu_dev)

    def test_tile_gpu(self):
        self.tile_test(gpu_dev)

    def noneZero_test(self, dev):
        X = np.array([[1, 0], [1, 1]]).astype(np.float32)
        y = np.array((np.nonzero(X)))

        x = tensor.from_numpy(X)
        x.to_device(dev)

        result = autograd.nonzero(x)
        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                                y,
                                                decimal=5)

    def test_noneZero_cpu(self):
        self.noneZero_test(cpu_dev)

    def test_noneZero_gpu(self):
        self.noneZero_test(gpu_dev)

    def cast_test(self, dev):
        config = [
            (np.float32, np.int32, tensor.int32),
            (np.int32, np.float32, tensor.float32),
        ]
        for t1, t2, t3 in config:
            X = np.array([[1, 0], [1, 1]]).astype(t1)
            y = np.array([[1, 0], [1, 1]]).astype(t2)

            x = tensor.from_numpy(X)
            x.to_device(dev)

            result = autograd.cast(x, t3)
            np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                                    y,
                                                    decimal=5)

    def test_cast_cpu(self):
        self.cast_test(cpu_dev)

    def test_cast_gpu(self):
        self.cast_test(gpu_dev)

if __name__ == '__main__':
    unittest.main()
