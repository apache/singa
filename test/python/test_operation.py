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
from singa import layer
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


def prepare_inputs_targets_for_rnn_test(dev):
    x_0 = np.random.random((2, 3)).astype(np.float32)
    x_1 = np.random.random((2, 3)).astype(np.float32)
    x_2 = np.random.random((2, 3)).astype(np.float32)

    h_0 = np.zeros((2, 2)).astype(np.float32)

    t_0 = np.random.random((2, 2)).astype(np.float32)
    t_1 = np.random.random((2, 2)).astype(np.float32)
    t_2 = np.random.random((2, 2)).astype(np.float32)

    x0 = tensor.Tensor(device=dev, data=x_0)
    x1 = tensor.Tensor(device=dev, data=x_1)
    x2 = tensor.Tensor(device=dev, data=x_2)

    h0 = tensor.Tensor(device=dev, data=h_0)

    t0 = tensor.Tensor(device=dev, data=t_0)
    t1 = tensor.Tensor(device=dev, data=t_1)
    t2 = tensor.Tensor(device=dev, data=t_2)

    inputs = [x0, x1, x2]
    targets = [t0, t1, t2]
    return inputs, targets, h0


class TestPythonOperation(unittest.TestCase):

    def check_shape(self, actual, expect):
        self.assertEqual(
            actual, expect, 'shape mismatch, actual shape is %s'
            ' exepcted is %s' %
            (_tuple_to_string(actual), _tuple_to_string(expect)))

    def _greater_helper(self, dev):
        x0 = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                       0.9]).reshape(3, 2).astype(np.float32)
        x1 = np.array([0, -0.3, 0, 0.1, 0, 0.9]).reshape(3,
                                                         2).astype(np.float32)
        y = np.greater(x0, x1)
        x0 = tensor.from_numpy(x0)
        x1 = tensor.from_numpy(x1)
        x0.to_device(dev)
        x1.to_device(dev)

        result = autograd.greater(x0, x1)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)

    def test_Greater_cpu(self):
        self._greater_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_Greater_gpu(self):
        self._greater_helper(gpu_dev)

    def _conv2d_helper(self, dev):
        # (out_channels, kernel_size)
        conv_0 = layer.Conv2d(1, 2)
        conv_without_bias_0 = layer.Conv2d(1, 2, bias=False)

        cpu_input_tensor = tensor.Tensor(shape=(2, 3, 3, 3), device=dev)
        cpu_input_tensor.gaussian(0.0, 1.0)

        dy = tensor.Tensor(shape=(2, 1, 2, 2), device=dev)
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

    def test_conv2d_cpu(self):
        self._conv2d_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_conv2d_gpu(self):
        self._conv2d_helper(gpu_dev)

    def _conv_same_pad(self, dev, pad_mode, is_2d):
        if is_2d:
            x_h, w_h, k_h, p_h = 32, 4, 4, 1
        else:
            x_h, w_h, k_h, p_h = 1, 1, 1, 0

        x = tensor.Tensor(shape=(3, 3, x_h, 32), device=dev)
        x.gaussian(0.0, 1.0)

        # with the same padding, the padding should be 3
        # for SAME_UPPER, is (1, 1) + (0, 1)
        # for SAME_LOWER, is (1, 1) + (1, 0)

        kernel = (k_h, 4)
        padding = (p_h, 1)
        stride = (1, 1)
        group = 1
        bias = False
        out_channels = 3

        conv_0 = layer.Conv2d(out_channels,
                              kernel,
                              stride=stride,
                              group=group,
                              bias=bias,
                              pad_mode=pad_mode)

        y = conv_0(x)
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

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_conv2d_same_pad_gpu(self):
        self._conv_same_pad(gpu_dev, "SAME_LOWER", True)
        self._conv_same_pad(gpu_dev, "SAME_UPPER", True)

    def test_conv1d_same_pad_cpu(self):
        self._conv_same_pad(cpu_dev, "SAME_LOWER", False)
        self._conv_same_pad(cpu_dev, "SAME_UPPER", False)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_conv1d_same_pad_gpu(self):
        self._conv_same_pad(gpu_dev, "SAME_LOWER", False)
        self._conv_same_pad(gpu_dev, "SAME_UPPER", False)

    def _pooling_same_pad(self, dev, pad_mode, is_2d):
        if is_2d:
            x_h, k_h, p_h = 32, 4, 1
        else:
            x_h, k_h, p_h = 1, 1, 0

        x = tensor.Tensor(shape=(3, 3, x_h, 32), device=dev)
        x.gaussian(0.0, 1.0)

        # with the same padding, the padding should be 3
        # for SAME_UPPER, is (1, 1) + (0, 1)
        # for SAME_LOWER, is (1, 1) + (1, 0)

        kernel = (k_h, 4)
        # we add 4 padding here and hope the conv and trim one padding then
        padding = (p_h, 1)
        stride = (1, 1)

        pooling = layer.Pooling2d(kernel, stride=stride, pad_mode=pad_mode)

        y = pooling(x)

        dy = np.ones((3, 3, x_h, 32), dtype=np.float32)
        dy = tensor.from_numpy(dy)
        dy.to_device(dev)

        dx = y.creator.backward(dy.data)
        self.check_shape(y.shape, (3, 3, x_h, 32))
        self.check_shape(dx.shape(), (3, 3, x_h, 32))

    def test_pooling2d_same_pad_cpu(self):
        self._pooling_same_pad(cpu_dev, "SAME_LOWER", True)
        self._pooling_same_pad(cpu_dev, "SAME_UPPER", True)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_pooling2d_same_pad_gpu(self):
        self._pooling_same_pad(gpu_dev, "SAME_LOWER", True)
        self._pooling_same_pad(gpu_dev, "SAME_UPPER", True)

    def test_pooling1d_same_pad_cpu(self):
        self._pooling_same_pad(cpu_dev, "SAME_LOWER", False)
        self._pooling_same_pad(cpu_dev, "SAME_UPPER", False)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_pooling1d_same_pad_gpu(self):
        self._pooling_same_pad(gpu_dev, "SAME_LOWER", False)
        self._pooling_same_pad(gpu_dev, "SAME_UPPER", False)

    def _sum_helper(self, dev):
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
        x.to_device(dev)
        x1.to_device(dev)
        dy.to_device(dev)

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

    def test_sum_cpu(self):
        self._sum_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_sum_gpu(self):
        self._sum_helper(gpu_dev)

    def _SeparableConv2d_helper(self, dev):
        # SeparableConv2d(in_channels, out_channels, kernel_size)
        if dev == cpu_dev:
            in_channels = 1
        else:
            in_channels = 8
        separ_conv = layer.SeparableConv2d(16, 3, padding=1)

        x = np.random.random((10, in_channels, 28, 28)).astype(np.float32)
        x = tensor.Tensor(device=dev, data=x)

        y = separ_conv(x)
        self.check_shape(y.shape, (10, 16, 28, 28))

        y1 = separ_conv.depthwise_conv(x)
        y2 = separ_conv.point_conv(y1)

        dy1, dW_depth = y2.creator.backward(y2.data)
        dx, dW_spacial = y1.creator.backward(dy1)

        self.check_shape(y2.shape, (10, 16, 28, 28))

        self.check_shape(dy1.shape(), (10, in_channels, 28, 28))
        self.check_shape(dW_depth.shape(), (16, in_channels, 1, 1))

        self.check_shape(dx.shape(), (10, in_channels, 28, 28))
        self.check_shape(dW_spacial.shape(), (in_channels, 1, 3, 3))

    def test_SeparableConv2d_cpu(self):
        self._SeparableConv2d_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_SeparableConv2d_gpu(self):
        self._SeparableConv2d_helper(gpu_dev)

    def _batchnorm2d_helper(self, dev):
        batchnorm_0 = layer.BatchNorm2d(3)

        cpu_input_tensor = tensor.Tensor(shape=(2, 3, 3, 3), device=dev)
        cpu_input_tensor.gaussian(0.0, 1.0)

        dy = cpu_input_tensor.clone().data

        y = batchnorm_0(cpu_input_tensor)
        dx, ds, db = y.creator.backward(dy)

        self.check_shape(y.shape, (2, 3, 3, 3))
        self.check_shape(dx.shape(), (2, 3, 3, 3))
        self.check_shape(ds.shape(), (3,))
        self.check_shape(db.shape(), (3,))

    def test_batchnorm2d_cpu(self):
        self._batchnorm2d_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_batchnorm2d_gpu(self):
        self._batchnorm2d_helper(gpu_dev)

    def gradients_check(self,
                        func,
                        param,
                        autograds,
                        h=0.0005,
                        df=1,
                        dev=cpu_dev):
        # param: PyTensor
        # autograds: numpy_tensor
        p = tensor.to_numpy(param)
        it = np.nditer(p, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            diff = np.zeros_like(p)
            diff[idx] += h
            diff = tensor.from_numpy(diff)
            diff.to_device(dev)

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

    def _vanillaRNN_gpu_tiny_ops_shape_check_helper(self, dev):
        # gradients shape check.
        inputs, target, h0 = prepare_inputs_targets_for_rnn_test(dev)
        rnn = layer.RNN(3, 2)

        hs, _ = rnn(inputs, h0)

        loss = autograd.softmax_cross_entropy(hs[0], target[0])
        for i in range(1, len(hs)):
            l = autograd.softmax_cross_entropy(hs[i], target[i])
            loss = autograd.add(loss, l)
        # d=autograd.infer_dependency(loss.creator)
        # print(d)
        for t, dt in autograd.backward(loss):
            self.check_shape(t.shape, dt.shape)

    def test_vanillaRNN_gpu_tiny_ops_shape_check_cpu(self):
        self._vanillaRNN_gpu_tiny_ops_shape_check_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_vanillaRNN_gpu_tiny_ops_shape_check_gpu(self):
        self._vanillaRNN_gpu_tiny_ops_shape_check_helper(gpu_dev)

    def _LSTM_gpu_tiny_ops_shape_check_helper(self, dev):
        # gradients shape check.
        inputs, target, h0 = prepare_inputs_targets_for_rnn_test(dev)
        c_0 = np.random.random((2, 1)).astype(np.float32)
        c0 = tensor.Tensor(device=dev, data=c_0)

        rnn = layer.LSTM(3, 2)

        hs, _, _ = rnn(inputs, (h0, c0))
        loss = autograd.softmax_cross_entropy(hs[0], target[0])

        for i in range(1, len(hs)):
            l = autograd.softmax_cross_entropy(hs[i], target[i])
            loss = autograd.add(loss, l)
        # d=autograd.infer_dependency(loss.creator)
        # print(d)
        for t, dt in autograd.backward(loss):
            self.check_shape(t.shape, dt.shape)

    def test_LSTM_gpu_tiny_ops_shape_check_cpu(self):
        self._LSTM_gpu_tiny_ops_shape_check_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_LSTM_gpu_tiny_ops_shape_check_gpu(self):
        self._LSTM_gpu_tiny_ops_shape_check_helper(gpu_dev)

    def _numerical_gradients_check_for_vallina_rnn_helper(self, dev):
        inputs, target, h0 = prepare_inputs_targets_for_rnn_test(dev)

        rnn = layer.RNN(3, 2)

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

        params = rnn.get_params()
        for key, param in params.items():
            auto_grad = tensor.to_numpy(auto_grads[id(param)])

            self.gradients_check(valinna_rnn_forward, param, auto_grad, dev=dev)

    def _gradient_check_cudnn_rnn(self, mode="vanilla", dev=gpu_dev):
        seq = 10
        bs = 2
        fea = 10
        hid = 10
        x = np.random.random((seq, bs, fea)).astype(np.float32)
        tx = tensor.Tensor(device=dev, data=x)
        y = np.random.random((seq, bs, hid)).astype(np.float32)
        y = np.reshape(y, (-1, hid))
        ty = tensor.Tensor(device=dev, data=y)
        rnn = layer.CudnnRNN(hid, rnn_mode=mode, return_sequences=True)

        def vanilla_rnn_forward():
            out = rnn(tx)
            out = autograd.reshape(out, (-1, hid))
            loss = autograd.softmax_cross_entropy(out, ty)
            return loss

        loss = vanilla_rnn_forward()
        auto_grads = autograd.gradients(loss)

        params = rnn.get_params()
        for key, param in params.items():
            auto_grad = tensor.to_numpy(auto_grads[id(param)])
            self.gradients_check(vanilla_rnn_forward, param, auto_grad, dev=dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_gradient_check_cudnn_rnn_vanilla(self):
        self._gradient_check_cudnn_rnn(mode="vanilla", dev=gpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_gradient_check_cudnn_rnn_lstm(self):
        self._gradient_check_cudnn_rnn(mode="lstm", dev=gpu_dev)

    # Cos Sim Gradient Check
    def _gradient_check_cossim(self, dev=gpu_dev):
        bs = 2
        vec = 3
        ta = tensor.random((bs, vec), dev)
        tb = tensor.random((bs, vec), dev)
        # treat ta, tb as params
        ta.stores_grad = True
        tb.stores_grad = True
        ty = tensor.random((bs,), dev)

        def _forward():
            out = autograd.cossim(ta, tb)
            loss = autograd.mse_loss(out, ty)
            return loss

        loss = _forward()
        auto_grads = autograd.gradients(loss)

        params = {id(ta): ta, id(tb): tb}

        for key, param in params.items():
            auto_grad = tensor.to_numpy(auto_grads[id(param)])
            self.gradients_check(_forward, param, auto_grad, dev=dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_gradient_check_cossim_gpu(self):
        self._gradient_check_cossim(dev=gpu_dev)

    def test_gradient_check_cossim_cpu(self):
        self._gradient_check_cossim(dev=cpu_dev)

    def test_numerical_gradients_check_for_vallina_rnn_cpu(self):
        self._numerical_gradients_check_for_vallina_rnn_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_numerical_gradients_check_for_vallina_rnn_gpu(self):
        self._numerical_gradients_check_for_vallina_rnn_helper(gpu_dev)

    def _numerical_gradients_check_for_lstm_helper(self, dev):
        inputs, target, h0 = prepare_inputs_targets_for_rnn_test(dev)
        c_0 = np.zeros((2, 2)).astype(np.float32)
        c0 = tensor.Tensor(device=dev, data=c_0)

        rnn = layer.LSTM(3, 2)

        def lstm_forward():
            hs, _, _ = rnn(inputs, (h0, c0))

            loss = autograd.softmax_cross_entropy(hs[0], target[0])
            for i in range(1, len(hs)):
                l = autograd.softmax_cross_entropy(hs[i], target[i])
                loss = autograd.add(loss, l)
            return loss

        loss1 = lstm_forward()
        auto_grads = autograd.gradients(loss1)

        params = rnn.get_params()
        for key, param in params.items():
            auto_grad = tensor.to_numpy(auto_grads[id(param)])

            self.gradients_check(lstm_forward, param, auto_grad, dev=dev)

    def test_numerical_gradients_check_for_lstm_cpu(self):
        self._numerical_gradients_check_for_lstm_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_numerical_gradients_check_for_lstm_gpu(self):
        self._numerical_gradients_check_for_lstm_helper(gpu_dev)

    def _MeanSquareError_helper(self, dev):
        X = np.array([4.3, 5.4, 3.3, 3.6, 5.7,
                      6.0]).reshape(3, 2).astype(np.float32)
        T = np.array([4.4, 5.3, 3.2, 3.7, 5.4,
                      6.3]).reshape(3, 2).astype(np.float32)
        x = tensor.from_numpy(X)
        t = tensor.from_numpy(T)
        x.to_device(dev)
        t.to_device(dev)

        loss = autograd.mse_loss(x, t)
        dx = loss.creator.backward()

        loss_np = tensor.to_numpy(loss)[0]
        self.assertAlmostEqual(loss_np, 0.0366666, places=4)
        self.check_shape(dx.shape(), (3, 2))

    def test_MeanSquareError_cpu(self):
        self._MeanSquareError_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_MeanSquareError_gpu(self):
        self._MeanSquareError_helper(gpu_dev)

    def _Abs_helper(self, dev):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5,
                      0.5]).reshape(3, 2).astype(np.float32)
        XT = np.array([0.8, 1.2, 3.3, 3.6, 0.5,
                       0.5]).reshape(3, 2).astype(np.float32)
        x = tensor.from_numpy(X)
        x.to_device(dev)

        result = autograd.abs(x)
        dx = result.creator.backward(x.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result), XT)
        self.check_shape(dx.shape(), (3, 2))

    def test_Abs_cpu(self):
        self._Abs_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_Abs_gpu(self):
        self._Abs_helper(gpu_dev)

    def _Mean_helper(self, dev):
        x0 = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                       0.9]).reshape(3, 2).astype(np.float32)
        x1 = np.array([0, -0.3, 0, 0.1, 0, 0.9]).reshape(3,
                                                         2).astype(np.float32)
        y = (x0 + x1) / 2
        grad = np.ones(x0.shape) / 2
        x0 = tensor.from_numpy(x0)
        x1 = tensor.from_numpy(x1)
        x0.to_device(dev)
        x1.to_device(dev)

        result = autograd.mean(x0, x1)
        dy = tensor.from_numpy(np.ones((3, 2)).astype(np.float32))
        dy.to_device(dev)
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
        self._Mean_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_Mean_gpu(self):
        self._Mean_helper(gpu_dev)

    def _Exp_helper(self, dev):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5,
                      0.5]).reshape(3, 2).astype(np.float32)
        XT = np.exp(X)
        x = tensor.from_numpy(X)
        x.to_device(dev)

        result = autograd.exp(x)
        dx = result.creator.backward(x.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        self.check_shape(dx.shape(), (3, 2))

    def test_Exp_cpu(self):
        self._Exp_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_Exp_gpu(self):
        self._Exp_helper(gpu_dev)

    def _Identity_helper(self, dev):
        x = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                      0.9]).reshape(3, 2).astype(np.float32)
        y = x.copy()
        grad = np.ones(x.shape)
        x = tensor.from_numpy(x)
        x.to_device(dev)

        result = autograd.identity(x)
        dy = tensor.from_numpy(np.ones((3, 2)).astype(np.float32))
        dy.to_device(dev)
        dx = result.creator.backward(dy.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             grad,
                                             decimal=5)
        self.check_shape(dx.shape(), (3, 2))

    def test_Identity_cpu(self):
        self._Identity_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_Identity_gpu(self):
        self._Identity_helper(gpu_dev)

    def _LeakyRelu_helper(self, dev):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5,
                      0.5]).reshape(3, 2).astype(np.float32)
        XT = np.array([0.8, -0.012, 3.3, -0.036, -0.005,
                       0.5]).reshape(3, 2).astype(np.float32)
        x = tensor.from_numpy(X)
        x.to_device(dev)

        result = autograd.leakyrelu(x)

        dx = result.creator.backward(x.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result), XT)
        self.check_shape(dx.shape(), (3, 2))

    def test_LeakyRelu_cpu(self):
        self._LeakyRelu_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_LeakyRelu_gpu(self):
        self._LeakyRelu_helper(gpu_dev)

    def _Relu_helper(self, dev):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5,
                      0.5]).reshape(3, 2).astype(np.float32)
        XT = np.maximum(X, 0)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(dev)
        dy.to_device(dev)

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

    def test_Relu_cpu(self):
        self._Relu_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_Relu_gpu(self):
        self._Relu_helper(gpu_dev)

    def _Cos_helper(self, dev):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5,
                      0.5]).reshape(3, 2).astype(np.float32)
        XT = np.cos(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(dev)
        dy.to_device(dev)

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

    def test_Cos_cpu(self):
        self._Cos_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_Cos_gpu(self):
        self._Cos_helper(gpu_dev)

    def _Cosh_helper(self, dev):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5,
                      0.5]).reshape(3, 2).astype(np.float32)
        XT = np.cosh(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(dev)
        dy.to_device(dev)

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

    def test_Cosh_cpu(self):
        self._Cosh_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_Cosh_gpu(self):
        self._Cosh_helper(gpu_dev)

    def _Acos_helper(self, dev):
        X = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                      0.9]).reshape(3, 2).astype(np.float32)
        XT = np.arccos(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(dev)
        dy.to_device(dev)

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

    def test_Acos_cpu(self):
        self._Acos_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_Acos_gpu(self):
        self._Acos_helper(gpu_dev)

    def _Acosh_helper(self, dev):
        X = np.array([1.1, 1.5, 1.9, 2.2, 2.5,
                      2.8]).reshape(3, 2).astype(np.float32)
        XT = np.arccosh(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(dev)
        dy.to_device(dev)

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

    def test_Acosh_cpu(self):
        self._Acosh_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_Acosh_gpu(self):
        self._Acosh_helper(gpu_dev)

    def _Sin_helper(self, dev):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5,
                      0.5]).reshape(3, 2).astype(np.float32)
        XT = np.sin(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(dev)
        dy.to_device(dev)

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

    def test_Sin_cpu(self):
        self._Sin_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_Sin_gpu(self):
        self._Sin_helper(gpu_dev)

    def _Sinh_helper(self, dev):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5,
                      0.5]).reshape(3, 2).astype(np.float32)
        XT = np.sinh(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(dev)
        dy.to_device(dev)

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

    def test_Sinh_cpu(self):
        self._Sinh_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_Sinh_gpu(self):
        self._Sinh_helper(gpu_dev)

    def _Asin_helper(self, dev):
        X = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                      0.9]).reshape(3, 2).astype(np.float32)
        XT = np.arcsin(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(dev)
        dy.to_device(dev)

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

    def test_Asin_cpu(self):
        self._Asin_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_Asin_gpu(self):
        self._Asin_helper(gpu_dev)

    def _Asinh_helper(self, dev):
        X = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                      0.9]).reshape(3, 2).astype(np.float32)
        XT = np.arcsinh(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(dev)
        dy.to_device(dev)

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

    def test_Asinh_cpu(self):
        self._Asinh_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_Asinh_gpu(self):
        self._Asinh_helper(gpu_dev)

    def _Tan_helper(self, dev):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5,
                      0.5]).reshape(3, 2).astype(np.float32)
        XT = np.tan(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(dev)
        dy.to_device(dev)

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

    def test_Tan_cpu(self):
        self._Tan_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_Tan_gpu(self):
        self._Tan_helper(gpu_dev)

    def _Tanh_helper(self, dev):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5,
                      0.5]).reshape(3, 2).astype(np.float32)
        XT = np.tanh(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(dev)
        dy.to_device(dev)

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

    def test_Tanh_cpu(self):
        self._Tanh_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_Tanh_gpu(self):
        self._Tanh_helper(gpu_dev)

    def _Atan_helper(self, dev):
        X = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                      0.9]).reshape(3, 2).astype(np.float32)
        XT = np.arctan(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(dev)
        dy.to_device(dev)

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

    def test_Atan_cpu(self):
        self._Atan_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_Atan_gpu(self):
        self._Atan_helper(gpu_dev)

    def _Atanh_helper(self, dev):
        X = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                      0.9]).reshape(3, 2).astype(np.float32)
        XT = np.arctanh(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(dev)
        dy.to_device(dev)

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

    def test_Atanh_cpu(self):
        self._Atanh_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_Atanh_gpu(self):
        self._Atanh_helper(gpu_dev)

    def _Less_helper(self, dev):
        x0 = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                       0.9]).reshape(3, 2).astype(np.float32)
        x1 = np.array([0, -0.3, 0, 0.1, 0, 0.9]).reshape(3,
                                                         2).astype(np.float32)
        y = np.less(x0, x1)
        x0 = tensor.from_numpy(x0)
        x1 = tensor.from_numpy(x1)
        x0.to_device(dev)
        x1.to_device(dev)

        result = autograd.less(x0, x1)
        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)

    def test_Less_cpu(self):
        self._Less_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_Less_gpu(self):
        self._Less_helper(gpu_dev)

    def _Sub_helper(self, dev):
        X0 = np.array([7, -5, 0.2, -0.1, 0.3, 4]).reshape(3,
                                                          2).astype(np.float32)
        X1 = np.array([0.6, -1.3, 0.1, -0.1, 0.4,
                       0.3]).reshape(3, 2).astype(np.float32)
        XT = np.subtract(X0, X1)

        DY = np.ones((3, 2), dtype=np.float32)
        x0 = tensor.from_numpy(X0)
        x1 = tensor.from_numpy(X1)
        dy = tensor.from_numpy(DY)
        x0.to_device(dev)
        x1.to_device(dev)
        dy.to_device(dev)

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

    def test_Sub_cpu(self):
        self._Sub_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_Sub_gpu(self):
        self._Sub_helper(gpu_dev)

    def _Pow_helper(self, dev):
        X0 = np.array([7, 5, 0.2, 0.1, 0.3, 4]).reshape(3, 2).astype(np.float32)
        X1 = np.array([-1.0, 2.0, -1.0, -2.1, 1.0,
                       -2.0]).reshape(3, 2).astype(np.float32)
        XT = np.power(X0, X1)

        DY = np.ones((3, 2), dtype=np.float32)
        x0 = tensor.from_numpy(X0)
        x1 = tensor.from_numpy(X1)
        dy = tensor.from_numpy(DY)
        x0.to_device(dev)
        x1.to_device(dev)
        dy.to_device(dev)

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

    def test_Pow_cpu(self):
        self._Pow_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_Pow_gpu(self):
        self._Pow_helper(gpu_dev)

    def _SoftSign_helper(self, dev):
        # y = x / (1 + np.abs(x))
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5,
                      0.5]).reshape(3, 2).astype(np.float32)
        XT = X / (1 + np.absolute(X))
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(dev)
        dy.to_device(dev)

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

    def test_SoftSign_cpu(self):
        self._SoftSign_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_SoftSign_gpu(self):
        self._SoftSign_helper(gpu_dev)

    def _SoftPlus_helper(self, dev):
        #y = np.log(np.exp(x) + 1)
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5,
                      0.5]).reshape(3, 2).astype(np.float32)
        XT = np.log(np.exp(X) + 1)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(dev)
        dy.to_device(dev)

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

    def test_SoftPlus_cpu(self):
        self._SoftPlus_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_SoftPlus_gpu(self):
        self._SoftPlus_helper(gpu_dev)

    def _unsqueeze_helper(self, dev):
        data = [0.1, -1.0, 0.4, 4.0, -0.9, 9.0]

        x = np.array(data).reshape(1, 2, 3).astype(np.float32)
        y = x.reshape(1, 1, 2, 3, 1)
        dy = np.ones((1, 1, 2, 3, 1), dtype=np.float32)
        grad = dy.reshape(1, 2, 3)

        x = tensor.from_numpy(x)
        dy = tensor.from_numpy(dy)
        x.to_device(dev)
        dy.to_device(dev)

        result = autograd.unsqueeze(x, [0, 4])
        dx = result.creator.backward(dy.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             grad,
                                             decimal=5)

    def test_unsqueeze_cpu(self):
        self._unsqueeze_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_unsqueeze_gpu(self):
        self._unsqueeze_helper(gpu_dev)

    def _Sqrt_helper(self, dev):
        X = np.array([0.1, 1.0, 0.4, 4.0, 0.9,
                      9.0]).reshape(3, 2).astype(np.float32)
        XT = np.sqrt(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(dev)
        dy.to_device(dev)

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

    def test_Sqrt_cpu(self):
        self._Sqrt_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_Sqrt_gpu(self):
        self._Sqrt_helper(gpu_dev)

    def _transpose_helper(self, dev):
        x = np.random.randn(3, 2, 1)
        y = x.transpose(1, 2, 0)
        dy = np.random.randn(*(y.shape))
        grad = dy.transpose((2, 0, 1))

        x = tensor.from_numpy(x)
        dy = tensor.from_numpy(dy)
        x.to_device(dev)
        dy.to_device(dev)

        result = autograd.transpose(x, (1, 2, 0))
        dx = result.creator.backward(dy.data)
        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             grad,
                                             decimal=5)

    def test_transpose_cpu(self):
        self._transpose_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_transpose_gpu(self):
        self._transpose_helper(gpu_dev)

    def _Sign_helper(self, dev):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5,
                      0.5]).reshape(3, 2).astype(np.float32)
        XT = np.sign(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(dev)
        dy.to_device(dev)
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

    def test_Sign_cpu(self):
        self._Sign_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_Sign_gpu(self):
        self._Sign_helper(gpu_dev)

    def _Log_helper(self, dev):
        X = np.array([0.1, 1.0, 0.4, 1.4, 0.9,
                      2.0]).reshape(3, 2).astype(np.float32)
        XT = np.log(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(dev)
        dy.to_device(dev)
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

    def test_Log_cpu(self):
        self._Log_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_Log_gpu(self):
        self._Log_helper(gpu_dev)

    def _mul_helper(self, dev):
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
        x.to_device(dev)
        slope.to_device(dev)
        dy.to_device(dev)

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

    def test_mul_cpu(self):
        self._mul_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_mul_gpu(self):
        self._mul_helper(gpu_dev)

    def _reshape_helper(self, dev):
        x = np.array([0.1, -1.0, 0.4, 4.0, -0.9,
                      9.0]).reshape(3, 2).astype(np.float32)
        y = x.reshape(2, 3)
        dy = np.array([1, 2, 3, 4, 5, 6]).reshape(2, 3).astype(np.float32)
        grad = dy.reshape(3, 2)

        x = tensor.from_numpy(x)
        dy = tensor.from_numpy(dy)
        x.to_device(dev)
        dy.to_device(dev)

        result = autograd.reshape(x, (2, 3))
        dx = result.creator.backward(dy.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             grad,
                                             decimal=5)

    def test_reshape_cpu(self):
        self._reshape_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_reshape_gpu(self):
        self._reshape_helper(gpu_dev)

    def _max_helper(self, dev):
        X0 = np.array([0.1, 0.2, 2.0, 0.0, 0.1,
                       0.2]).reshape(3, 2).astype(np.float32)
        X1 = np.array([1.0, 2.0, 1.0, 2.1, 0.0,
                       2.0]).reshape(3, 2).astype(np.float32)
        XT = np.maximum(X0, X1)

        DY = np.ones((3, 2), dtype=np.float32)
        x0 = tensor.from_numpy(X0)
        x1 = tensor.from_numpy(X1)
        dy = tensor.from_numpy(DY)
        x0.to_device(dev)
        x1.to_device(dev)
        dy.to_device(dev)

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

    def test_max_cpu(self):
        self._max_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_max_gpu(self):
        self._max_helper(gpu_dev)

    def _max_3inputs_helper(self, dev):
        data_0 = np.array([3, 2, 1]).astype(np.float32)
        data_1 = np.array([1, 4, 4]).astype(np.float32)
        data_2 = np.array([2, 5, 3]).astype(np.float32)
        XT = np.array([3, 5, 4]).astype(np.float32)

        DY = np.array([1, 1, 1]).astype(np.float32)
        x0 = tensor.from_numpy(data_0)
        x1 = tensor.from_numpy(data_1)
        x2 = tensor.from_numpy(data_2)
        dy = tensor.from_numpy(DY)
        x0.to_device(dev)
        x1.to_device(dev)
        x2.to_device(dev)
        dy.to_device(dev)

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

    def test_max_3inputs_cpu(self):
        self._max_3inputs_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_max_3inputs_gpu(self):
        self._max_3inputs_helper(gpu_dev)

    def _max_1inputs_helper(self, dev):
        data_0 = np.array([3, 2, 1]).astype(np.float32)
        XT = np.array([3, 2, 1]).astype(np.float32)

        DY = np.array([1, 1, 1]).astype(np.float32)
        x0 = tensor.from_numpy(data_0)
        dy = tensor.from_numpy(DY)
        x0.to_device(dev)
        dy.to_device(dev)

        result = autograd.max(x0)
        dx0 = result.creator.backward(dy.data)

        DX0 = np.array([1, 1, 1]).astype(np.float32)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)

    def test_max_1inputs_cpu(self):
        self._max_1inputs_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_max_1inputs_gpu(self):
        self._max_1inputs_helper(gpu_dev)

    def _Div_helper(self, dev):
        X0 = np.array([7, -5, 0.2, -0.1, 0.3, 4]).reshape(3,
                                                          2).astype(np.float32)
        X1 = np.array([0.6, -1.3, 0.1, -0.1, 0.4,
                       0.3]).reshape(3, 2).astype(np.float32)
        XT = np.divide(X0, X1)

        DY = np.ones((3, 2), dtype=np.float32)
        x0 = tensor.from_numpy(X0)
        x1 = tensor.from_numpy(X1)
        dy = tensor.from_numpy(DY)
        x0.to_device(dev)
        x1.to_device(dev)
        dy.to_device(dev)

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

    def test_Div_cpu(self):
        self._Div_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_Div_gpu(self):
        self._Div_helper(gpu_dev)

    def _squeeze_helper(self, dev):
        x = np.random.randn(3, 1, 2, 1, 1)
        y = x.reshape(3, 2)
        dy = np.random.randn(3, 2)
        grad = dy.reshape(3, 1, 2, 1, 1)

        x = tensor.from_numpy(x)
        dy = tensor.from_numpy(dy)
        x.to_device(dev)
        dy.to_device(dev)

        result = autograd.squeeze(x, [1, 3, 4])
        dx = result.creator.backward(dy.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             grad,
                                             decimal=5)

    def test_squeeze_cpu(self):
        self._squeeze_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_squeeze_gpu(self):
        self._squeeze_helper(gpu_dev)

    def _shape_helper(self, dev):
        x = np.array([0.1, -1.0, 0.4, 4.0, -0.9,
                      9.0]).reshape(3, 2).astype(np.float32)
        y = list(x.shape)
        dy = np.ones((3, 2), dtype=np.float32)
        grad = list(dy.shape)

        x = tensor.from_numpy(x)
        dy = tensor.from_numpy(dy)
        x.to_device(dev)
        dy.to_device(dev)

        result = autograd.shape(x)
        dx = result.creator.backward(dy.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)
        np.testing.assert_array_almost_equal(dx, grad, decimal=5)

    def test_shape_cpu(self):
        self._shape_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_shape_gpu(self):
        self._shape_helper(gpu_dev)

    def _min_helper(self, dev):
        X0 = np.array([0.1, 0.2, 2.0, 0.0, 0.1,
                       0.2]).reshape(3, 2).astype(np.float32)
        X1 = np.array([1.0, 2.0, 1.0, 2.1, 0.0,
                       2.0]).reshape(3, 2).astype(np.float32)
        XT = np.minimum(X0, X1)

        DY = np.ones((3, 2), dtype=np.float32)
        x0 = tensor.from_numpy(X0)
        x1 = tensor.from_numpy(X1)
        dy = tensor.from_numpy(DY)
        x0.to_device(dev)
        x1.to_device(dev)
        dy.to_device(dev)

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

    def test_min_cpu(self):
        self._min_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_min_gpu(self):
        self._min_helper(gpu_dev)

    def _min_3inputs_helper(self, dev):
        data_0 = np.array([3, 2, 1]).astype(np.float32)
        data_1 = np.array([1, 4, 4]).astype(np.float32)
        data_2 = np.array([2, 5, 0]).astype(np.float32)
        XT = np.array([1, 2, 0]).astype(np.float32)

        DY = np.array([1, 1, 1]).astype(np.float32)
        x0 = tensor.from_numpy(data_0)
        x1 = tensor.from_numpy(data_1)
        x2 = tensor.from_numpy(data_2)
        dy = tensor.from_numpy(DY)
        x0.to_device(dev)
        x1.to_device(dev)
        x2.to_device(dev)
        dy.to_device(dev)

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

    def test_min_3inputs_cpu(self):
        self._min_3inputs_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_min_3inputs_gpu(self):
        self._min_3inputs_helper(gpu_dev)

    def _min_1inputs_helper(self, dev):
        data_0 = np.array([3, 2, 1]).astype(np.float32)
        XT = np.array([3, 2, 1]).astype(np.float32)

        DY = np.array([1, 1, 1]).astype(np.float32)
        x0 = tensor.from_numpy(data_0)
        dy = tensor.from_numpy(DY)
        x0.to_device(dev)
        dy.to_device(dev)

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

    def test_min_1inputs_cpu(self):
        self._min_1inputs_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_min_1inputs_gpu(self):
        self._min_1inputs_helper(gpu_dev)

    def _HardSigmoid_helper(self, dev):
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
        x.to_device(dev)
        dy.to_device(dev)

        result = autograd.hardsigmoid(x, a, g)
        dx = result.creator.backward(dy.data)
        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             grad,
                                             decimal=5)

    def test_HardSigmoid_cpu(self):
        self._HardSigmoid_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_HardSigmoid_gpu(self):
        self._HardSigmoid_helper(gpu_dev)

    def _prelu_helper(self, dev):
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

    def test_prelu_cpu(self):
        self._prelu_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_prelu_gpu(self):
        self._prelu_helper(gpu_dev)

    def _SeLU_helper(self, dev):
        x = np.random.randn(3, 2)
        a = 0.2
        g = 0.3
        y = np.clip(x, 0,
                    np.inf) * g + (np.exp(np.clip(x, -np.inf, 0)) - 1) * a * g
        dy = np.random.randn(3, 2)
        grad = (np.exp(np.clip(x, -np.inf, 0))) * g
        grad[x <= 0] = grad[x <= 0] * a
        grad *= dy

        x = tensor.from_numpy(x)
        dy = tensor.from_numpy(dy)
        x.to_device(dev)
        dy.to_device(dev)
        result = autograd.selu(x, a, g)
        dx = result.creator.backward(dy.data)
        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             grad,
                                             decimal=5)

    def test_SeLU_cpu(self):
        self._SeLU_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_SeLU_gpu(self):
        self._SeLU_helper(gpu_dev)

    def _and_helper(self, dev):
        x0 = np.array([0, -0.3, -0.1, 0.1, 0.5,
                       0.9]).reshape(3, 2).astype(np.float32)
        x1 = np.array([0, -0.3, 0, 0.1, 0.5, 0.9]).reshape(3,
                                                           2).astype(np.float32)

        y = np.logical_and(x0, x1)
        x0 = tensor.from_numpy(x0)
        x1 = tensor.from_numpy(x1)
        x0.to_device(dev)
        x1.to_device(dev)

        result = autograd._and(x0, x1)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)

    def test_and_cpu(self):
        self._and_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_and_gpu(self):
        self._and_helper(gpu_dev)

    def _or_helper(self, dev):
        x0 = np.array([1.0, 1.0, 2.0, -3.0, 0,
                       -7.0]).reshape(3, 2).astype(np.float32)
        x1 = np.array([-1.0, 0, 2.0, 4.0, 0,
                       -7.0]).reshape(3, 2).astype(np.float32)

        y = np.logical_or(x0, x1)
        x0 = tensor.from_numpy(x0)
        x1 = tensor.from_numpy(x1)
        x0.to_device(dev)
        x1.to_device(dev)

        result = autograd._or(x0, x1)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)

    def test_or_cpu(self):
        self._or_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_or_gpu(self):
        self._or_helper(gpu_dev)

    def _not_helper(self, dev):
        x = np.array([1.0, -1.0, 0, -0.1, 0,
                      -7.0]).reshape(3, 2).astype(np.float32)

        y = np.logical_not(x)
        x = tensor.from_numpy(x)
        x.to_device(dev)

        result = autograd._not(x)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)

    def test_not_cpu(self):
        self._not_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_not_gpu(self):
        self._not_helper(gpu_dev)

    def _xor_helper(self, dev):
        x0 = np.array([0, -0.3, -0.1, 0.1, 0.5,
                       9.0]).reshape(3, 2).astype(np.float32)
        x1 = np.array([0, -0.3, 0, 0.1, 0, 0.9]).reshape(3,
                                                         2).astype(np.float32)

        y = np.logical_xor(x0, x1)
        x0 = tensor.from_numpy(x0)
        x1 = tensor.from_numpy(x1)
        x0.to_device(dev)
        x1.to_device(dev)

        result = autograd._xor(x0, x1)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)

    def test_xor_cpu(self):
        self._xor_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_xor_gpu(self):
        self._xor_helper(gpu_dev)

    def _negative_helper(self, dev):
        X = np.array([0.1, 0, 0.4, 1. - 4, 0.9,
                      -2.0]).reshape(3, 2).astype(np.float32)
        XT = np.negative(X)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(dev)
        dy.to_device(dev)

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

    def test_negative_cpu(self):
        self._negative_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_negative_gpu(self):
        self._negative_helper(gpu_dev)

    def _reciprocal_helper(self, dev):
        X = np.array([0.1, 0, 0.4, 1. - 4, 0.9,
                      -2.0]).reshape(3, 2).astype(np.float32)
        DY = np.ones((3, 2), dtype=np.float32)

        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(dev)
        dy.to_device(dev)

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

    def test_reciprocal_cpu(self):
        self._reciprocal_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_reciprocal_gpu(self):
        self._reciprocal_helper(gpu_dev)

    def _and_broadcast_helper(self, dev):
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
        self._and_broadcast_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_and_broadcast_gpu(self):
        self._and_broadcast_helper(gpu_dev)

    def _or_broadcast_helper(self, dev):
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
        self._or_broadcast_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_or_broadcast_gpu(self):
        self._or_broadcast_helper(gpu_dev)

    def _xor_broadcast_helper(self, dev):
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
        self._xor_broadcast_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_xor_broadcast_gpu(self):
        self._xor_broadcast_helper(gpu_dev)

    def _greater_broadcast_helper(self, dev):
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
        self._greater_broadcast_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_greater_broadcast_gpu(self):
        self._greater_broadcast_helper(gpu_dev)

    def _less_broadcast_helper(self, dev):
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

    def test_less_broadcast_cpu(self):
        self._less_broadcast_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_less_broadcast_gpu(self):
        self._less_broadcast_helper(gpu_dev)

    def _add_broadcast_helper(self, dev):
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
        self._add_broadcast_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_add_broadcast_gpu(self):
        self._add_broadcast_helper(gpu_dev)

    def _sub_broadcast_helper(self, dev):
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
        self._sub_broadcast_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_sub_broadcast_gpu(self):
        self._sub_broadcast_helper(gpu_dev)

    def _mul_broadcast_helper(self, dev):
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
        self._mul_broadcast_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_mul_broadcast_gpu(self):
        self._mul_broadcast_helper(gpu_dev)

    def _div_broadcast_helper(self, dev):
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
            # use realtive and total error instead of demical number
            np.testing.assert_allclose(tensor.to_numpy(result),
                                       y,
                                       rtol=1e-4,
                                       atol=1e-4)
            np.testing.assert_allclose(tensor.to_numpy(
                tensor.from_raw_tensor(dx0)),
                                       grad0,
                                       rtol=1e-4,
                                       atol=1e-4)
            np.testing.assert_allclose(tensor.to_numpy(
                tensor.from_raw_tensor(dx1)),
                                       grad1,
                                       rtol=1e-4,
                                       atol=1e-4)

    def test_div_broadcast_cpu(self):
        self._div_broadcast_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_div_broadcast_gpu(self):
        self._div_broadcast_helper(gpu_dev)

    def _pow_broadcast_helper(self, dev):
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
        self._pow_broadcast_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_pow_broadcast_gpu(self):
        self._pow_broadcast_helper(gpu_dev)

    def _prelu_broadcast_helper(self, dev):
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
        self._prelu_broadcast_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_prelu_broadcast_gpu(self):
        self._prelu_broadcast_helper(gpu_dev)

    def _gemm_helper(self, dev):
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
            DY = np.ones(shapeY, dtype=np.float32)

            if transB == 0:
                out_features = shapeB[1]
            else:
                out_features = shapeB[0]

            a = tensor.from_numpy(A)
            a.to_device(dev)
            dy = tensor.from_numpy(DY)
            dy.to_device(dev)

            gemm = layer.Gemm(out_features, alpha, beta, transA == 1,
                              transB == 1)
            result = gemm(a)

            params = gemm.get_params()
            B = tensor.to_numpy(params['W'])
            C = tensor.to_numpy(params['b'])

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
        self._gemm_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_gemm_gpu(self):
        self._gemm_helper(gpu_dev)

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

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

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

        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             XT,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_globalaveragepool_cpu(self):
        self.globalaveragepool_channel_first(cpu_dev)
        self.globalaveragepool_channel_last(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
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

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
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

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_dropout_gpu(self):
        self.dropout_test(gpu_dev)

    def reduceSum_test(self, dev):
        shape = [3, 2, 2]
        cases = [(None, 1), ([1], 0), ([1], 1), ([-2], 1), ([1, 2], 1)]
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

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_reduceSum_gpu(self):
        self.reduceSum_test(gpu_dev)

    def reduceMean_test(self, dev):
        shape = [3, 2, 2]
        cases = [(None, 1), ([1], 0), ([1], 1), ([-2], 1), ([1, 2], 1)]
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

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_reduceMean_gpu(self):
        self.reduceMean_test(gpu_dev)

    def slice_test(self, dev):
        X = np.random.randn(20, 10, 5).astype(np.float32)
        indexes = np.array(range(20 * 10 * 5)).reshape(20, 10, 5)
        configs = [
            # starts, ends, axes, steps, y
            [[0, 0], [3, 10], [0, 1], [1, 1], X[0:3, 0:10],
             indexes[0:3, 0:10]],  # slice
            [[0, 0, 3], [20, 10, 4], None, None, X[:, :, 3:4],
             indexes[:, :, 3:4]],  # slice_default_axes
            [[1], [1000], [1], [1], X[:, 1:1000],
             indexes[:, 1:1000]],  # slice_end_out_of_bounds
            [[0], [-1], [1], [1], X[:, 0:-1],
             indexes[:, 0:-1]],  # slice_end_out_of_bounds
            [[20, 10, 4], [0, 0, 1], [0, 1, 2], [-1, -3, -2],
             X[20:0:-1, 10:0:-3, 4:1:-2], indexes[20:0:-1, 10:0:-3,
                                                  4:1:-2]],  # slice_neg_steps
            [[0, 0, 3], [20, 10, 4], [0, -2, -1], None, X[:, :, 3:4],
             indexes[:, :, 3:4]],  # slice_negative_axes
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
            dX = np.array([
                1. if i in dx_idx else 0. for i in range(20 * 10 * 5)
            ]).reshape(X.shape)

            np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                                 y,
                                                 decimal=5)
            np.testing.assert_array_almost_equal(tensor.to_numpy(
                tensor.from_raw_tensor(dx)),
                                                 dX,
                                                 decimal=5)

    def test_slice_cpu(self):
        self.slice_test(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
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

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_ceil_gpu(self):
        self.ceil_test(gpu_dev)

    def floor_test(self, dev):
        X = np.array([-1.9, 1.2]).astype(np.float32)
        DY = np.ones((2), dtype=np.float32)
        y = np.floor(X)
        x = tensor.from_numpy(X)
        dy = tensor.from_numpy(DY)
        x.to_device(dev)
        dy.to_device(dev)

        result = autograd.floor(x)
        dx = result.creator.backward(dy.data)
        DX = np.zeros((2), dtype=np.float32)
        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_floor_cpu(self):
        self.floor_test(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_floor_gpu(self):
        self.floor_test(gpu_dev)

    def _test_scatter_elements(self, dev):
        # testing witout axis
        data = np.zeros((3, 3), dtype=np.float32)
        indices = np.array([[1, 0, 2], [0, 2, 1]], dtype=np.int32)
        updates = np.array([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]], dtype=np.float32)
        output = np.array([[2.0, 1.1, 0.0], [1.0, 0.0, 2.2], [0.0, 2.1, 1.2]],
                          dtype=np.float32)

        data = tensor.from_numpy(data)
        indices = tensor.from_numpy(indices)
        updates = tensor.from_numpy(updates)
        data.to_device(dev)
        indices.to_device(dev)
        updates.to_device(dev)

        result = autograd.scatter_elements(data, indices, updates)
        dy = tensor.from_numpy(np.ones(data.shape, dtype=np.float32))
        dx = result.creator.backward(dy.data)
        np.testing.assert_almost_equal(tensor.to_numpy(result),
                                       output,
                                       decimal=5)
        self.check_shape(dx.shape(), data.shape)

        # testing with axis
        data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
        indices = np.array([[1, 3]], dtype=np.int32)
        updates = np.array([[1.1, 2.1]], dtype=np.float32)
        output = np.array([[1.0, 1.1, 3.0, 2.1, 5.0]], dtype=np.float32)

        data = tensor.from_numpy(data)
        indices = tensor.from_numpy(indices)
        updates = tensor.from_numpy(updates)
        data.to_device(dev)
        indices.to_device(dev)
        updates.to_device(dev)

        result = autograd.scatter_elements(data, indices, updates, axis=1)
        dy = tensor.from_numpy(np.ones(data.shape, dtype=np.float32))
        dx = result.creator.backward(dy.data)
        np.testing.assert_almost_equal(tensor.to_numpy(result),
                                       output,
                                       decimal=5)
        self.check_shape(dx.shape(), data.shape)

        # testing with negative indices:
        data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
        indices = np.array([[1, -3]], dtype=np.int64)
        updates = np.array([[1.1, 2.1]], dtype=np.float32)
        output = np.array([[1.0, 1.1, 2.1, 4.0, 5.0]], dtype=np.float32)

        data = tensor.from_numpy(data)
        indices = tensor.from_numpy(indices)
        updates = tensor.from_numpy(updates)
        data.to_device(dev)
        indices.to_device(dev)
        updates.to_device(dev)

        result = autograd.scatter_elements(data, indices, updates, axis=1)
        dy = tensor.from_numpy(np.ones(data.shape, dtype=np.float32))
        dx = result.creator.backward(dy.data)
        np.testing.assert_almost_equal(tensor.to_numpy(result),
                                       output,
                                       decimal=5)
        self.check_shape(dx.shape(), data.shape)

    def test_cpu_scatter_elements(self):
        self._test_scatter_elements(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_gpu_scatter_elements(self):
        self._test_scatter_elements(gpu_dev)

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
            np.testing.assert_array_almost_equal(tensor.to_numpy(_r),
                                                 y[idx],
                                                 decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(
            tensor.from_raw_tensor(dx)),
                                             DX,
                                             decimal=5)

    def test_split_cpu(self):
        self.split_test(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_split_gpu(self):
        self.split_test(gpu_dev)

    def gather_test(self, dev):
        config = [([0, 1, 3], 0), ([0, 1, 3], 1), ([[0, 1], [1, 2], [2, 3]], 1),
                  ([0, -1, -2], 0)]  # (indices, axis)
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

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
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

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
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

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
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
            result_np = tensor.to_numpy(result)
            assert result_np.dtype == y.dtype, "type %s != %s." % (
                result_np.dtype, y.dtype)
            np.testing.assert_array_almost_equal(result_np, y, decimal=5)

    def test_cast_cpu(self):
        self.cast_test(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_cast_gpu(self):
        self.cast_test(gpu_dev)

    def onehot_test(self, dev):

        def one_hot(indices, depth, axis=-1, dtype=np.float32):  # type: ignore
            ''' Compute one hot from indices at a specific axis '''
            values = np.asarray(indices)
            rank = len(values.shape)
            depth_range = np.arange(depth)
            if axis < 0:
                axis += (rank + 1)
            ls = values.shape[0:axis]
            rs = values.shape[axis:rank]
            targets = np.reshape(depth_range, (1,) * len(ls) +
                                 depth_range.shape + (1,) * len(rs))
            values = np.reshape(np.mod(values, depth), ls + (1,) + rs)
            return np.asarray(targets == values, dtype=dtype)

        axisValue = 1
        on_value = 3
        off_value = 1
        output_type = np.float32
        indices = np.array([[1, 9], [2, 4]], dtype=np.float32)
        depth = np.array([10], dtype=np.float32)
        values = np.array([off_value, on_value], dtype=output_type)
        y = one_hot(indices, depth, axis=axisValue, dtype=output_type)
        y = y * (on_value - off_value) + off_value

        x = tensor.from_numpy(indices)
        x.to_device(dev)

        result = autograd.onehot(axisValue, x, depth, values)
        np.testing.assert_array_almost_equal(tensor.to_numpy(result),
                                             y,
                                             decimal=5)

    def test_onehot_cpu(self):
        self.onehot_test(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_onehot_gpu(self):
        self.onehot_test(gpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_cudnn_rnn_operation(self, dev=gpu_dev):
        # init params, inputs
        hidden_size = 7
        seq_length = 5
        batch_size = 6
        feature_size = 3
        directions = 2
        num_layers = 2

        for mode in [0, 1, 2, 3]:  # 0-relu, 1-tanh, 2-lstm, 3-gru
            x = tensor.Tensor(shape=(seq_length, batch_size, feature_size),
                              device=dev).gaussian(0, 1)
            hx = tensor.Tensor(shape=(num_layers * directions, batch_size,
                                      hidden_size),
                               device=dev).gaussian(0, 1)
            cx = tensor.Tensor(shape=(num_layers * directions, batch_size,
                                      hidden_size),
                               device=dev).gaussian(0, 1)
            dy = tensor.Tensor(shape=(seq_length, batch_size,
                                      directions * hidden_size),
                               device=dev).gaussian(0, 1)

            # init cudnn rnn op
            rnn_handle = singa.CudnnRNNHandle(x.data,
                                              hidden_size,
                                              mode,
                                              num_layers=num_layers,
                                              dropout=0.1,
                                              bidirectional=1)

            w = tensor.Tensor(shape=(rnn_handle.weights_size,),
                              device=dev).gaussian(0, 1)

            # return sequence, y shape = {seq, bs, hidden}
            # init operator/operation
            _rnn = autograd._RNN(rnn_handle, return_sequences=True)

            # forward
            y = _rnn(x, hx, cx, w)[0]
            assert y.shape == dy.shape
            # print(ys)

            # backward
            dx, dhx, dcx, dw = _rnn.backward(dy.data)

            # return no sequence, y shape = {bs, hidden}
            _rnn = autograd._RNN(rnn_handle, return_sequences=False)
            dy = tensor.Tensor(shape=(batch_size, directions * hidden_size),
                               device=dev).gaussian(0, 1)
            y = _rnn(x, hx, cx, w)[0]

            assert y.shape == dy.shape
            # backward
            dx, dhx, dcx, dw = _rnn.backward(dy.data)

    def cossim_helper(self, dev):
        A = np.random.randn(*[3, 10]).astype(np.float32)
        B = np.random.randn(*[3, 10]).astype(np.float32)

        a = tensor.from_numpy(A)
        a.to_device(dev)
        b = tensor.from_numpy(B)
        b.to_device(dev)

        DY = np.random.randn(3).astype(np.float32)
        dy = tensor.from_numpy(DY)
        dy.to_device(dev)

        y = autograd.cossim(a, b)
        da, db = y.creator.backward(dy.data)  # CTensor

        self.check_shape(y.shape, (3,))
        self.check_shape(da.shape(), (3, 10))
        self.check_shape(db.shape(), (3, 10))

    def test_cossim_cpu(self):
        self.cossim_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_cossim_gpu(self):
        self.cossim_helper(gpu_dev)

    def expand_helper(self, dev):
        shape = [3, 1]
        X = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32),
                       shape)
        x = tensor.from_numpy(X)
        x.to_device(dev)

        # dim_changed
        new_shape = [2, 1, 6]
        y_t = X * np.ones(new_shape, dtype=np.float32)
        dy = tensor.from_numpy(y_t)
        dy.to_device(dev)
        y = autograd.expand(x, new_shape)
        dx = y.creator.backward(dy.data)
        np.testing.assert_array_almost_equal(tensor.to_numpy(y), y_t)
        self.check_shape(dx.shape(), tuple(shape))

        # dim_unchanged
        new_shape_2 = [3, 4]
        y_t2 = np.tile(X, 4)
        dy2 = tensor.from_numpy(y_t2)
        dy2.to_device(dev)
        y2 = autograd.expand(x, new_shape_2)
        dx2 = y2.creator.backward(dy2.data)
        np.testing.assert_array_almost_equal(tensor.to_numpy(y2), y_t2)
        self.check_shape(dx2.shape(), tuple(shape))

    def test_expand_cpu(self):
        self.expand_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_expand_gpu(self):
        self.expand_helper(gpu_dev)

    def pad_helper(self, dev):
        X = np.array([
            [1.0, 1.2],
            [2.3, 3.4],
            [4.5, 5.7],
        ]).astype(np.float32)
        Y1 = np.array([
            [0.0, 0.0, 1.0, 1.2],
            [0.0, 0.0, 2.3, 3.4],
            [0.0, 0.0, 4.5, 5.7],
        ],).astype(np.float32)
        Y2 = np.array([
            [1.0, 1.2, 1.0, 1.2],
            [2.3, 3.4, 2.3, 3.4],
            [4.5, 5.7, 4.5, 5.7],
        ],).astype(np.float32)
        Y3 = np.array([
            [1.0, 1.0, 1.0, 1.2],
            [2.3, 2.3, 2.3, 3.4],
            [4.5, 4.5, 4.5, 5.7],
        ],).astype(np.float32)

        x = tensor.from_numpy(X)
        x.to_device(dev)
        pads = [0, 2, 0, 0]

        DY = np.random.randn(3, 4).astype(np.float32)
        dy = tensor.from_numpy(DY)
        dy.to_device(dev)

        y1 = autograd.pad(x, "constant", pads)
        y2 = autograd.pad(x, "reflect", pads)
        y3 = autograd.pad(x, "edge", pads)
        dx1 = y1.creator.backward(dy.data)
        dx2 = y2.creator.backward(dy.data)
        dx3 = y3.creator.backward(dy.data)
        pad_width = []
        half_width = len(pads) // 2
        for i in range(half_width):
            pad_width += [[pads[i], pads[i + half_width]]]

        np.testing.assert_array_almost_equal(tensor.to_numpy(y1),
                                             np.pad(
                                                 X,
                                                 pad_width=pad_width,
                                                 mode="constant",
                                                 constant_values=0.,
                                             ),
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(y2),
                                             np.pad(
                                                 X,
                                                 pad_width=pad_width,
                                                 mode="reflect",
                                             ),
                                             decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(y3),
                                             np.pad(
                                                 X,
                                                 pad_width=pad_width,
                                                 mode="edge",
                                             ),
                                             decimal=5)
        self.check_shape(dx1.shape(), (3, 2))
        self.check_shape(dx2.shape(), (3, 2))
        self.check_shape(dx3.shape(), (3, 2))

    def test_pad_cpu(self):
        self.pad_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_pad_gpu(self):
        self.pad_helper(gpu_dev)

    def upsample_helper(self, dev):
        X = np.array([[[
            [1, 2],
            [3, 4],
        ]]], dtype=np.float32)
        x = tensor.from_numpy(X)
        x.to_device(dev)

        scales = np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float32)
        y_t = np.array([[[
            [1, 1, 1, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
            [3, 3, 3, 4, 4, 4],
            [3, 3, 3, 4, 4, 4],
        ]]],
                       dtype=np.float32)
        dy = tensor.from_numpy(y_t)
        dy.to_device(dev)

        y = autograd.upsample(x, "nearest", scales)
        dx = y.creator.backward(dy.data)
        np.testing.assert_array_almost_equal(tensor.to_numpy(y), y_t)
        self.check_shape(dx.shape(), tuple(X.shape))

    def test_upsample_cpu(self):
        self.upsample_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_upsample_gpu(self):
        self.upsample_helper(gpu_dev)

    def depth_space_helper(self, dev):
        # (1, 8, 2, 3) input tensor
        X = np.array(
            [[[[0., 1., 2.], [3., 4., 5.]], [[9., 10., 11.], [12., 13., 14.]],
              [[18., 19., 20.], [21., 22., 23.]],
              [[27., 28., 29.], [30., 31., 32.]],
              [[36., 37., 38.], [39., 40., 41.]],
              [[45., 46., 47.], [48., 49., 50.]],
              [[54., 55., 56.], [57., 58., 59.]],
              [[63., 64., 65.], [66., 67., 68.]]]],
            dtype=np.float32)
        x = tensor.from_numpy(X)
        x.to_device(dev)

        # (1, 2, 4, 6) output tensor
        y_t = np.array(
            [[[[0., 18., 1., 19., 2., 20.], [36., 54., 37., 55., 38., 56.],
               [3., 21., 4., 22., 5., 23.], [39., 57., 40., 58., 41., 59.]],
              [[9., 27., 10., 28., 11., 29.], [45., 63., 46., 64., 47., 65.],
               [12., 30., 13., 31., 14., 32.], [48., 66., 49., 67., 50., 68.]]]
            ],
            dtype=np.float32)
        dy = tensor.from_numpy(y_t)
        dy.to_device(dev)
        y = autograd.depth_to_space(x, 2, "DCR")
        dx = y.creator.backward(dy.data)
        np.testing.assert_array_almost_equal(tensor.to_numpy(y), y_t)
        np.testing.assert_array_almost_equal(
            tensor.to_numpy(tensor.from_raw_tensor(dx)), X)

        y = autograd.space_to_depth(dy, 2, "DCR")
        dx = y.creator.backward(x.data)
        np.testing.assert_array_almost_equal(tensor.to_numpy(y), X)
        np.testing.assert_array_almost_equal(
            tensor.to_numpy(tensor.from_raw_tensor(dx)), y_t)

        y_t = np.array(
            [[[[0., 9., 1., 10., 2., 11.], [18., 27., 19., 28., 20., 29.],
               [3., 12., 4., 13., 5., 14.], [21., 30., 22., 31., 23., 32.]],
              [[36., 45., 37., 46., 38., 47.], [54., 63., 55., 64., 56., 65.],
               [39., 48., 40., 49., 41., 50.], [57., 66., 58., 67., 59., 68.]]]
            ],
            dtype=np.float32)
        dy = tensor.from_numpy(y_t)
        dy.to_device(dev)
        y = autograd.depth_to_space(x, 2, "CRD")
        dx = y.creator.backward(dy.data)
        np.testing.assert_array_almost_equal(tensor.to_numpy(y), y_t)
        np.testing.assert_array_almost_equal(
            tensor.to_numpy(tensor.from_raw_tensor(dx)), X)

        y = autograd.space_to_depth(dy, 2, "CRD")
        dx = y.creator.backward(x.data)
        np.testing.assert_array_almost_equal(tensor.to_numpy(y), X)
        np.testing.assert_array_almost_equal(
            tensor.to_numpy(tensor.from_raw_tensor(dx)), y_t)

    def test_depth_space_cpu(self):
        self.depth_space_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_depth_space_gpu(self):
        self.depth_space_helper(gpu_dev)

    def test_invalid_inputs(self, dev=cpu_dev):
        _1d = tensor.Tensor((10,), dev)
        _2d = tensor.Tensor((10, 10), dev)
        _3d = tensor.Tensor((10, 10, 10), dev)
        self.assertRaises(AssertionError, autograd.softmax_cross_entropy, _2d,
                          _3d)
        self.assertRaises(AssertionError, autograd.mse_loss, _2d, _3d)
        self.assertRaises(AssertionError, autograd.add_bias, _2d, _1d, 3)
        self.assertRaises(AssertionError, autograd.ranking_loss, _2d, _1d)

    def where_helper(self, dev):
        X = np.array([[1, 2], [3, 4]], dtype=np.float32)
        x = tensor.from_numpy(X)
        x.to_device(dev)

        X2 = np.array([[9, 8], [7, 6]], dtype=np.float32)
        x2 = tensor.from_numpy(X2)
        x2.to_device(dev)

        condition = [[True, False], [True, True]]
        y_t = np.where(condition, X, X2)
        dx1_t = np.array([[1, 0], [3, 4]], dtype=np.float32)
        dx2_t = np.array([[0, 8], [0, 0]], dtype=np.float32)
        dy = tensor.from_numpy(y_t)
        dy.to_device(dev)

        y = autograd.where(x, x2, condition)
        dx1, dx2 = y.creator.backward(dy.data)
        np.testing.assert_array_almost_equal(tensor.to_numpy(y), y_t)
        np.testing.assert_array_almost_equal(
            tensor.to_numpy(tensor.from_raw_tensor(dx1)), dx1_t)
        np.testing.assert_array_almost_equal(
            tensor.to_numpy(tensor.from_raw_tensor(dx2)), dx2_t)

    def test_where_cpu(self):
        self.where_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_where_gpu(self):
        self.where_helper(gpu_dev)

    def rounde_helper(self, dev):
        X = np.array([
            0.1, 0.5, 0.9, 1.2, 1.5, 1.8, 2.3, 2.5, 2.7, -1.1, -1.5, -1.9, -2.2,
            -2.5, -2.8
        ]).astype(np.float32)
        x = tensor.from_numpy(X)
        x.to_device(dev)

        y_t = np.array(
            [0., 0., 1., 1., 2., 2., 2., 2., 3., -1., -2., -2., -2., -2.,
             -3.]).astype(np.float32)
        dy = tensor.from_numpy(y_t)
        dy.to_device(dev)

        y = autograd.rounde(x)
        np.testing.assert_array_almost_equal(tensor.to_numpy(y), y_t)

    def test_rounde_cpu(self):
        self.rounde_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_rounde_gpu(self):
        self.rounde_helper(gpu_dev)

    def round_helper(self, dev):
        X = np.array([
            0.1, 0.5, 0.9, 1.2, 1.5, 1.8, 2.3, 2.5, 2.7, -1.1, -1.5, -1.9, -2.2,
            -2.5, -2.8
        ]).astype(np.float32)
        x = tensor.from_numpy(X)
        x.to_device(dev)

        y_t = np.array(
            [0., 1., 1., 1., 2., 2., 2., 3., 3., -1., -2., -2., -2., -3.,
             -3.]).astype(np.float32)
        dy = tensor.from_numpy(y_t)
        dy.to_device(dev)

        y = autograd.round(x)
        np.testing.assert_array_almost_equal(tensor.to_numpy(y), y_t)

    def test_round_cpu(self):
        self.round_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_round_gpu(self):
        self.round_helper(gpu_dev)

    def embedding_helper(self, dev):
        embedding = layer.Embedding(10, 3)

        X = np.array([[0, 1, 2, 3], [9, 8, 7, 6]])
        x = tensor.from_numpy(X)
        x.to_device(dev)

        dy = tensor.Tensor(shape=(2, 4, 3), device=dev)
        dy.gaussian(0.0, 1.0)

        y = embedding(x)  # PyTensor
        dx, dW = y.creator.backward(dy.data)  # CTensor

        self.check_shape(y.shape, (2, 4, 3))
        self.check_shape(dx.shape(), (2, 4))
        self.check_shape(dW.shape(), (10, 3))

    def test_embedding_cpu(self):
        self.embedding_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_embedding_gpu(self):
        self.embedding_helper(gpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def _cossim_value(self, dev=gpu_dev):
        # numpy val
        np.random.seed(0)
        bs = 1000
        vec_s = 1200
        a = np.random.random((bs, vec_s)).astype(np.float32)
        b = np.random.random((bs, vec_s)).astype(np.float32)
        dy = np.random.random((bs,)).astype(np.float32)

        # singa tensor
        ta = tensor.from_numpy(a)
        tb = tensor.from_numpy(b)
        tdy = tensor.from_numpy(dy)
        ta.to_device(dev)
        tb.to_device(dev)
        tdy.to_device(dev)

        # singa forward and backward
        ty = autograd.cossim(ta, tb)
        tda, tdb = ty.creator.backward(tdy.data)

        np_forward = list()
        for i in range(len(a)):
            a_norm = np.linalg.norm(a[i])
            b_norm = np.linalg.norm(b[i])
            ab_dot = np.dot(a[i], b[i])
            out = ab_dot / (a_norm * b_norm)
            np_forward.append(out)

        np_backward_a = list()
        np_backward_b = list()
        for i in range(len(a)):
            a_norm = np.linalg.norm(a[i])
            b_norm = np.linalg.norm(b[i])
            da = dy[i] * (b[i] / (a_norm * b_norm) - (np_forward[i] * a[i]) /
                          (a_norm * a_norm))
            db = dy[i] * (a[i] / (a_norm * b_norm) - (np_forward[i] * b[i]) /
                          (b_norm * b_norm))
            np_backward_a.append(da)
            np_backward_b.append(db)

        np.testing.assert_array_almost_equal(tensor.to_numpy(ty),
                                             np.array(np_forward))
        np.testing.assert_array_almost_equal(
            tensor.to_numpy(tensor.from_raw_tensor(tda)), np_backward_a)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_cossim_value_gpu(self):
        self._cossim_value(gpu_dev)

    def test_cossim_value_cpu(self):
        self._cossim_value(cpu_dev)

    def test_mse_loss_value(self, dev=cpu_dev):
        y = np.random.random((1000, 1200)).astype(np.float32)
        tar = np.random.random((1000, 1200)).astype(np.float32)
        # get singa value
        sy = tensor.from_numpy(y, dev)
        starget = tensor.from_numpy(tar, dev)
        sloss = autograd.mse_loss(sy, starget)
        sgrad = sloss.creator.backward()
        # get np value result
        np_loss = np.mean(np.square(tar - y))
        np_grad = -2 * (tar - y) / np.prod(tar.shape)
        # value check
        np.testing.assert_array_almost_equal(
            tensor.to_numpy(tensor.from_raw_tensor(sgrad)), np_grad)
        np.testing.assert_array_almost_equal(tensor.to_numpy(sloss), np_loss)

    def erf_helper(self, dev):
        X = np.array([
            0.1, 0.5, 0.9, 1.2, 1.5, 1.8, 2.3, 2.5, 2.7, -1.1, -1.5, -1.9, -2.2,
            -2.5, -2.8
        ]).astype(np.float32)
        x = tensor.from_numpy(X)
        x.to_device(dev)

        import math

        y_t = np.vectorize(math.erf)(X)
        dy = tensor.from_numpy(y_t)
        dy.to_device(dev)
        dx_t = 2. / np.pi**0.5 * np.exp(-np.power(y_t, 2))

        y = autograd.erf(x)
        dx = y.creator.backward(dy.data)
        np.testing.assert_array_almost_equal(tensor.to_numpy(y), y_t)
        np.testing.assert_array_almost_equal(
            tensor.to_numpy(tensor.from_raw_tensor(dx)), dx_t)

    def test_erf_cpu(self):
        self.erf_helper(cpu_dev)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_erf_gpu(self):
        self.erf_helper(gpu_dev)


if __name__ == '__main__':
    unittest.main()
