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

from cuda_helper import gpu_dev, cpu_dev

import numpy as np

autograd.training = True

CTensor = singa.Tensor

dy = CTensor([2, 1, 2, 2])
singa.Gaussian(0.0, 1.0, dy)


def _tuple_to_string(t):
    lt = [str(x) for x in t]
    return '(' + ', '.join(lt) + ')'


def prepare_inputs_targets_for_rnn_test():
    x_0 = np.random.random((2, 3)).astype(np.float32)
    x_1 = np.random.random((2, 3)).astype(np.float32)
    x_2 = np.random.random((2, 3)).astype(np.float32)

    h_0 = np.zeros((2, 2)).astype(
        np.float32)

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
        self.assertEqual(actual, expect, 'shape mismatch, actual shape is %s'
                         ' exepcted is %s' % (_tuple_to_string(actual),
                                              _tuple_to_string(expect))
                         )

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

    def test_conv2d_cpu(self):
        # (in_channels, out_channels, kernel_size)
        conv_1 = autograd.Conv2d(3, 1, 2)
        conv_without_bias_1 = autograd.Conv2d(3, 1, 2, bias=False)

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

    def test_SeparableConv2d_gpu(self):
        # SeparableConv2d(in_channels, out_channels, kernel_size)
        separ_conv=autograd.SeparableConv2d(8, 16, 3, padding=1)

        x=np.random.random((10,8,28,28)).astype(np.float32)
        x=tensor.Tensor(device=gpu_dev, data=x)

        y1 = separ_conv.depthwise_conv(x)
        y2 = separ_conv.point_conv(y1)

        dy1, dW_depth, _ = y2.creator.backward(y2.data)
        dx, dW_spacial, _ = y1.creator.backward(dy1)

        self.check_shape(y2.shape, (10, 16, 28, 28))

        self.check_shape(dy1.shape(), (10, 8, 28, 28))
        self.check_shape(dW_depth.shape(), (16, 8, 1, 1))

        self.check_shape(dx.shape(), (10, 8, 28, 28))
        self.check_shape(dW_spacial.shape(), (8, 1, 3, 3))

        y = separ_conv(x)
        self.check_shape(y.shape, (10, 16, 28, 28))


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
            self.assertAlmostEqual(
                autograds[idx] - numerical_grad, 0., places=2)

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
        X=np.array([4.3,5.4,3.3,3.6,5.7,6.0]).reshape(3,2).astype(np.float32)
        T=np.array([4.4,5.3,3.2,3.7,5.4,6.3]).reshape(3,2).astype(np.float32)
        x=tensor.from_numpy(X)
        t=tensor.from_numpy(T)
        x.to_device(gpu_dev)
        t.to_device(gpu_dev)

        loss= autograd.mse_loss(x,t)
        dx=loss.creator.backward()[0]

        loss_np=tensor.to_numpy(loss)[0]
        self.assertAlmostEqual(loss_np, 0.0366666, places=4)
        self.check_shape(dx.shape(), (3, 2))

    def test_Abs(self):
        X=np.array([0.8,-1.2,3.3,-3.6,-0.5,0.5]).reshape(3,2).astype(np.float32)
        XT=np.array([0.8,1.2,3.3,3.6,0.5,0.5]).reshape(3,2).astype(np.float32)
        x=tensor.from_numpy(X)
        x.to_device(gpu_dev)

        result=autograd.abs(x)
        dx=result.creator.backward(x.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result), XT)
        self.check_shape(dx.shape(), (3, 2))

    def test_Exp(self):
        X=np.array([0.8,-1.2,3.3,-3.6,-0.5,0.5]).reshape(3,2).astype(np.float32)
        XT=np.exp(X)
        x=tensor.from_numpy(X)
        x.to_device(gpu_dev)

        result=autograd.exp(x)
        dx=result.creator.backward(x.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result), XT, decimal=5)
        self.check_shape(dx.shape(), (3, 2))

    def test_LeakyRelu(self):
        X=np.array([0.8,-1.2,3.3,-3.6,-0.5,0.5]).reshape(3,2).astype(np.float32)
        XT=np.array([0.8,-0.012,3.3,-0.036,-0.005,0.5]).reshape(3,2).astype(np.float32)
        x=tensor.from_numpy(X)
        x.to_device(gpu_dev)

        result=autograd.leakyrelu(x)

        dx=result.creator.backward(x.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result), XT)
        self.check_shape(dx.shape(), (3, 2))

    def test_Cos_cpu(self):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5, 0.5]).reshape(3, 2).astype(np.float32)
        XT = np.cos(X)
        x = tensor.from_numpy(X)
        x.to_device(cpu_dev)

        result = autograd.cos(x)
        dx = result.creator.backward(x.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result), XT, decimal=5)
        self.check_shape(dx.shape(), (3, 2))

    def test_Cos_gpu(self):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5, 0.5]).reshape(3, 2).astype(np.float32)
        XT = np.cos(X)
        x = tensor.from_numpy(X)
        x.to_device(gpu_dev)

        result = autograd.cos(x)
        dx = result.creator.backward(x.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result), XT, decimal=5)
        self.check_shape(dx.shape(), (3, 2))

    def test_Cosh_cpu(self):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5, 0.5]).reshape(3, 2).astype(np.float32)
        XT = np.cosh(X)
        x = tensor.from_numpy(X)
        x.to_device(cpu_dev)

        result = autograd.cosh(x)
        dx = result.creator.backward(x.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result), XT, decimal=5)
        self.check_shape(dx.shape(), (3, 2))

    def test_Cosh_gpu(self):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5, 0.5]).reshape(3, 2).astype(np.float32)
        XT = np.cosh(X)
        x = tensor.from_numpy(X)
        x.to_device(gpu_dev)

        result = autograd.cosh(x)
        dx = result.creator.backward(x.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result), XT, decimal=5)
        self.check_shape(dx.shape(), (3, 2))

    def test_Acos_cpu(self):
        X = np.array([-0.9, -0.3, -0.1, 0.1, 0.5, 0.9]).reshape(3, 2).astype(np.float32)
        XT = np.arccos(X)
        x = tensor.from_numpy(X)
        x.to_device(cpu_dev)

        result = autograd.acos(x)
        dx = result.creator.backward(x.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result), XT, decimal=5)
        self.check_shape(dx.shape(), (3, 2))

    def test_Acos_gpu(self):
        X = np.array([-0.9, -0.3, -0.1, 0.1, 0.5, 0.9]).reshape(3, 2).astype(np.float32)
        XT = np.arccos(X)
        x = tensor.from_numpy(X)
        x.to_device(gpu_dev)

        result = autograd.acos(x)
        dx = result.creator.backward(x.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result), XT, decimal=5)
        self.check_shape(dx.shape(), (3, 2))

    def test_Acosh_cpu(self):
        X = np.array([1.1, 1.5, 1.9, 2.2, 2.5, 2.8]).reshape(3, 2).astype(np.float32)
        XT = np.arccosh(X)
        x = tensor.from_numpy(X)
        x.to_device(cpu_dev)

        result = autograd.acosh(x)
        dx = result.creator.backward(x.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result), XT, decimal=5)
        self.check_shape(dx.shape(), (3, 2))

    def test_Acosh_gpu(self):
        X = np.array([1.1, 1.5, 1.9, 2.2, 2.5, 2.8]).reshape(3, 2).astype(np.float32)
        XT = np.arccosh(X)
        x = tensor.from_numpy(X)
        x.to_device(gpu_dev)

        result = autograd.acosh(x)
        dx = result.creator.backward(x.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result), XT, decimal=5)
        self.check_shape(dx.shape(), (3, 2))

    def test_Sin_cpu(self):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5, 0.5]).reshape(3, 2).astype(np.float32)
        XT = np.sin(X)
        x = tensor.from_numpy(X)
        x.to_device(cpu_dev)

        result = autograd.sin(x)
        dx = result.creator.backward(x.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result), XT, decimal=5)
        self.check_shape(dx.shape(), (3, 2))

    def test_Sin_gpu(self):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5, 0.5]).reshape(3, 2).astype(np.float32)
        XT = np.sin(X)
        x = tensor.from_numpy(X)
        x.to_device(gpu_dev)

        result = autograd.sin(x)
        dx = result.creator.backward(x.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result), XT, decimal=5)
        self.check_shape(dx.shape(), (3, 2))

    def test_Sinh_cpu(self):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5, 0.5]).reshape(3, 2).astype(np.float32)
        XT = np.sinh(X)
        x = tensor.from_numpy(X)
        x.to_device(cpu_dev)

        result = autograd.sinh(x)
        dx = result.creator.backward(x.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result), XT, decimal=5)
        self.check_shape(dx.shape(), (3, 2))

    def test_Sinh_gpu(self):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5, 0.5]).reshape(3, 2).astype(np.float32)
        XT = np.sinh(X)
        x = tensor.from_numpy(X)
        x.to_device(gpu_dev)

        result = autograd.sinh(x)
        dx = result.creator.backward(x.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result), XT, decimal=5)
        self.check_shape(dx.shape(), (3, 2))

    def test_Asin_cpu(self):
        X = np.array([-0.9, -0.3, -0.1, 0.1, 0.5, 0.9]).reshape(3, 2).astype(np.float32)
        XT = np.arcsin(X)
        x = tensor.from_numpy(X)
        x.to_device(cpu_dev)

        result = autograd.asin(x)
        dx = result.creator.backward(x.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result), XT, decimal=5)
        self.check_shape(dx.shape(), (3, 2))

    def test_Asin_gpu(self):
        X = np.array([-0.9, -0.3, -0.1, 0.1, 0.5, 0.9]).reshape(3, 2).astype(np.float32)
        XT = np.arcsin(X)
        x = tensor.from_numpy(X)
        x.to_device(gpu_dev)

        result = autograd.asin(x)
        dx = result.creator.backward(x.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result), XT, decimal=5)
        self.check_shape(dx.shape(), (3, 2))

    def test_Asinh_cpu(self):
        X = np.array([-0.9, -0.3, -0.1, 0.1, 0.5, 0.9]).reshape(3, 2).astype(np.float32)
        XT = np.arcsinh(X)
        x = tensor.from_numpy(X)
        x.to_device(cpu_dev)

        result = autograd.asinh(x)
        dx = result.creator.backward(x.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result), XT, decimal=5)
        self.check_shape(dx.shape(), (3, 2))

    def test_Asinh_gpu(self):
        X = np.array([-0.9, -0.3, -0.1, 0.1, 0.5, 0.9]).reshape(3, 2).astype(np.float32)
        XT = np.arcsinh(X)
        x = tensor.from_numpy(X)
        x.to_device(gpu_dev)

        result = autograd.asinh(x)
        dx = result.creator.backward(x.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result), XT, decimal=5)
        self.check_shape(dx.shape(), (3, 2))

    def test_Tan_cpu(self):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5, 0.5]).reshape(3, 2).astype(np.float32)
        XT = np.tan(X)
        x = tensor.from_numpy(X)
        x.to_device(cpu_dev)

        result = autograd.tan(x)
        dx = result.creator.backward(x.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result), XT, decimal=5)
        self.check_shape(dx.shape(), (3, 2))

    def test_Tan_gpu(self):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5, 0.5]).reshape(3, 2).astype(np.float32)
        XT = np.tan(X)
        x = tensor.from_numpy(X)
        x.to_device(gpu_dev)

        result = autograd.tan(x)
        dx = result.creator.backward(x.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result), XT, decimal=5)
        self.check_shape(dx.shape(), (3, 2))

    def test_Tanh_cpu(self):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5, 0.5]).reshape(3, 2).astype(np.float32)
        XT = np.tanh(X)
        x = tensor.from_numpy(X)
        x.to_device(cpu_dev)

        result = autograd.tanh(x)
        dx = result.creator.backward(x.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result), XT, decimal=5)
        self.check_shape(dx.shape(), (3, 2))

    def test_Tanh_gpu(self):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5, 0.5]).reshape(3, 2).astype(np.float32)
        XT = np.tanh(X)
        x = tensor.from_numpy(X)
        x.to_device(gpu_dev)

        result = autograd.tanh(x)
        dx = result.creator.backward(x.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result), XT, decimal=5)
        self.check_shape(dx.shape(), (3, 2))

    def test_Atan_cpu(self):
        X = np.array([-0.9, -0.3, -0.1, 0.1, 0.5, 0.9]).reshape(3, 2).astype(np.float32)
        XT = np.arctan(X)
        x = tensor.from_numpy(X)
        x.to_device(cpu_dev)

        result = autograd.atan(x)
        dx = result.creator.backward(x.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result), XT, decimal=5)
        self.check_shape(dx.shape(), (3, 2))

    def test_Atan_gpu(self):
        X = np.array([-0.9, -0.3, -0.1, 0.1, 0.5, 0.9]).reshape(3, 2).astype(np.float32)
        XT = np.arctan(X)
        x = tensor.from_numpy(X)
        x.to_device(gpu_dev)

        result = autograd.atan(x)
        dx = result.creator.backward(x.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result), XT, decimal=5)
        self.check_shape(dx.shape(), (3, 2))

    def test_Atanh_cpu(self):
        X = np.array([-0.9, -0.3, -0.1, 0.1, 0.5, 0.9]).reshape(3, 2).astype(np.float32)
        XT = np.arctanh(X)
        x = tensor.from_numpy(X)
        x.to_device(cpu_dev)

        result = autograd.atanh(x)
        dx = result.creator.backward(x.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result), XT, decimal=5)
        self.check_shape(dx.shape(), (3, 2))

    def test_Atanh_gpu(self):
        X = np.array([-0.9, -0.3, -0.1, 0.1, 0.5, 0.9]).reshape(3, 2).astype(np.float32)
        XT = np.arctanh(X)
        x = tensor.from_numpy(X)
        x.to_device(gpu_dev)

        result = autograd.atanh(x)
        dx = result.creator.backward(x.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(result), XT, decimal=5)
        self.check_shape(dx.shape(), (3, 2))

    def test_Div_cpu(self):
        X0 = np.array([7, -5, 0.2, -0.1, 0.3, 4]).reshape(3, 2).astype(np.float32)
        X1 = np.array([0.6, -1.3, 0.1, -0.1, 0.4, 0.3]).reshape(3, 2).astype(np.float32)
        XT = np.divide(X0, X1)

        DY = np.ones((3, 2), dtype = np.float32)
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

        np.testing.assert_array_almost_equal(tensor.to_numpy(result), XT, decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(tensor.from_raw_tensor(dx0)), DX0, decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(tensor.from_raw_tensor(dx1)), DX1, decimal=5)

    def test_Div_gpu(self):
        X0 = np.array([7, -5, 0.2, -0.1, 0.3, 4]).reshape(3, 2).astype(np.float32)
        X1 = np.array([0.6, -1.3, 0.1, -0.1, 0.4, 0.3]).reshape(3, 2).astype(np.float32)
        XT = np.divide(X0, X1)

        DY = np.ones((3, 2), dtype = np.float32)
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

        np.testing.assert_array_almost_equal(tensor.to_numpy(result), XT, decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(tensor.from_raw_tensor(dx0)), DX0, decimal=5)
        np.testing.assert_array_almost_equal(tensor.to_numpy(tensor.from_raw_tensor(dx1)), DX1, decimal=5)


if __name__ == '__main__':
    unittest.main()
