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

from singa import singa_wrap as singa_api
from singa import tensor
from singa import singa_wrap as singa
from singa import autograd
from singa import layer
from singa import sonnx
from singa import opt

import onnx
from onnx import (defs, checker, helper, numpy_helper, mapping, ModelProto,
                  GraphProto, NodeProto, AttributeProto, TensorProto,
                  OperatorSetIdProto)
from onnx.helper import make_tensor, make_tensor_value_info, make_node, make_graph

from cuda_helper import gpu_dev, cpu_dev

import numpy as np

autograd.training = True


def _tuple_to_string(t):
    lt = [str(x) for x in t]
    return '(' + ', '.join(lt) + ')'


class TestPythonOnnx(unittest.TestCase):

    def check_shape(self, actual, expect):
        self.assertEqual(
            actual, expect, 'shape mismatch, actual shape is %s'
            ' exepcted is %s' %
            (_tuple_to_string(actual), _tuple_to_string(expect)))

    def _conv2d_helper(self, dev):
        x = tensor.Tensor(shape=(2, 3, 3, 3), device=dev)
        x.gaussian(0.0, 1.0)
        y = layer.Conv2d(1, 2)(x)

        # frontend
        model = sonnx.to_onnx([x], [y])
        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_conv2d_cpu(self):
        self._conv2d_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_conv2d_gpu(self):
        self._conv2d_helper(gpu_dev)

    def _relu_helper(self, dev):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5,
                      0.5]).reshape(3, 2).astype(np.float32)
        XT = np.array([0.8, 0, 3.3, 0, 0, 0.5]).reshape(3, 2).astype(np.float32)
        x = tensor.from_numpy(X)
        x.to_device(dev)
        y = autograd.ReLU()(x)[0]

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])
        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_relu_cpu(self):
        self._relu_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_relu_gpu(self):
        self._relu_helper(gpu_dev)

    def _avg_pool_helper(self, dev):
        x = tensor.Tensor(shape=(2, 3, 3, 3), device=dev)
        x.gaussian(0.0, 1.0)
        y = layer.AvgPool2d(3, 1, 2)(x)

        # frontend
        model = sonnx.to_onnx([x], [y])

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_avg_pool_cpu(self):
        self._avg_pool_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_avg_pool_gpu(self):
        self._avg_pool_helper(gpu_dev)

    def _softmax_helper(self, dev):
        X = np.array([[-1, 0, 1]]).astype(np.float32)
        x = tensor.from_numpy(X)
        x.to_device(dev)
        y = autograd.SoftMax()(x)[0]

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])
        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_softmax_cpu(self):
        self._softmax_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_softmax_gpu(self):
        self._softmax_helper(gpu_dev)

    def _sigmoid_helper(self, dev):
        X = np.array([[-1, 0, 1]]).astype(np.float32)
        x = tensor.from_numpy(X)
        x.to_device(dev)
        y = autograd.Sigmoid()(x)[0]

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])
        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_sigmoid_cpu(self):
        self._sigmoid_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_sigmoid_gpu(self):
        self._sigmoid_helper(gpu_dev)

    def _add_helper(self, dev):
        X1 = np.random.randn(3, 4, 5).astype(np.float32)
        X2 = np.random.randn(3, 4, 5).astype(np.float32)

        x1 = tensor.from_numpy(X1)
        x2 = tensor.from_numpy(X2)
        x1.to_device(dev)
        x2.to_device(dev)
        y = autograd.Add()(x1, x2)[0]

        # frontend
        model = sonnx.to_onnx([x1, x2], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x1, x2])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_add_cpu(self):
        self._add_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_add_gpu(self):
        self._add_helper(gpu_dev)

    def _concat_helper(self, dev):
        X1 = np.random.randn(3, 4, 5).astype(np.float32)
        X2 = np.random.randn(3, 4, 5).astype(np.float32)

        x1 = tensor.from_numpy(X1)
        x2 = tensor.from_numpy(X2)
        x1.to_device(dev)
        x2.to_device(dev)
        y = autograd.Concat()(x1, x2)[0]

        # frontend
        model = sonnx.to_onnx([x1, x2], [y])

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x1, x2])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_concat_cpu(self):
        self._concat_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_concat_gpu(self):
        self._concat_helper(gpu_dev)

    def _matmul_helper(self, dev):
        X1 = np.random.randn(4, 5).astype(np.float32)
        X2 = np.random.randn(5, 4).astype(np.float32)

        x1 = tensor.from_numpy(X1)
        x2 = tensor.from_numpy(X2)
        x1.to_device(dev)
        x2.to_device(dev)

        y = autograd.Matmul()(x1, x2)[0]

        # frontend
        model = sonnx.to_onnx([x1, x2], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x1, x2])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_matmul_cpu(self):
        self._matmul_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_matmul_gpu(self):
        self._matmul_helper(gpu_dev)

    def _max_pool_helper(self, dev):
        x = tensor.Tensor(shape=(2, 3, 4, 4), device=dev)
        x.gaussian(0.0, 1.0)
        y = layer.MaxPool2d(2, 2, 0)(x)

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_max_pool_cpu(self):
        self._max_pool_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_max_pool_gpu(self):
        self._max_pool_helper(gpu_dev)

    def _batch_norm_helper(self, dev):
        x = np.array([[[[-1, 0, 1]], [[2, 3, 4]]]]).astype(np.float32)
        s = np.array([1.0, 1.5]).astype(np.float32)
        bias = np.array([0, 1]).astype(np.float32)
        mean = np.array([0, 3]).astype(np.float32)
        var = np.array([1, 1.5]).astype(np.float32)

        x = tensor.from_numpy(x)
        x.to_device(dev)
        s = tensor.from_numpy(s)
        s.to_device(dev)

        bias = tensor.from_numpy(bias)
        mean = tensor.from_numpy(mean)
        var = tensor.from_numpy(var)

        bias.to_device(dev)
        mean.to_device(dev)
        var.to_device(dev)
        if dev == cpu_dev:
            handle = singa.BatchNormHandle(0.9, x.data)
        else:
            handle = singa.CudnnBatchNormHandle(0.9, x.data)
        y = autograd.batchnorm_2d(handle, x, s, bias, mean, var)

        # frontend
        model = sonnx.to_onnx([x, s, bias, mean, var], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x, s, bias])  # mean and var has been stored in graph

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_batch_norm_cpu(self):
        self._batch_norm_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_batch_norm_gpu(self):
        self._batch_norm_helper(gpu_dev)

    def _linear_helper(self, dev):
        x = tensor.Tensor(shape=(2, 20), device=dev)
        x.gaussian(0.0, 1.0)
        x1 = x.clone()
        y = layer.Linear(20, 1, bias=False)(x)

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x1])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_linear_cpu(self):
        self._linear_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_linear_gpu(self):
        self._linear_helper(gpu_dev)

    def _gemm_helper(self, dev):
        A = np.random.randn(2, 3).astype(np.float32)
        B = np.random.rand(3, 4).astype(np.float32)
        C = np.random.rand(2, 4).astype(np.float32)
        alpha = 1.0
        beta = 2.0

        tA = tensor.from_numpy(A)
        tB = tensor.from_numpy(B)
        tC = tensor.from_numpy(C)
        tA.to_device(dev)
        tB.to_device(dev)
        tC.to_device(dev)
        y = autograd.Gemm(alpha, beta, 0, 0)(tA, tB, tC)[0]

        # frontend
        model = sonnx.to_onnx([tA, tB, tC], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([tA, tB, tC])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_gemm_cpu(self):
        self._gemm_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_gemm_gpu(self):
        self._gemm_helper(gpu_dev)

    def _reshape_helper(self, dev):
        x = np.array([0.1, -1.0, 0.4, 4.0, -0.9,
                      9.0]).reshape(3, 2).astype(np.float32)
        x = tensor.from_numpy(x)
        x.to_device(dev)
        y = autograd.Reshape((2, 3))(x)[0]

        # frontend
        model = sonnx.to_onnx([x, (2, 3)], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])  # shape has been stored in graph

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_reshape_cpu(self):
        self._reshape_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_reshape_gpu(self):
        self._reshape_helper(gpu_dev)

    def _sum_helper(self, dev):
        x = np.array([0.1, -1.0, 0.4, 4.0, -0.9,
                      9.0]).reshape(3, 2).astype(np.float32)
        x1 = np.array([0.1, 1.0, 0.4, 4.0, 0.9,
                       9.0]).reshape(3, 2).astype(np.float32)
        x = tensor.from_numpy(x)
        x1 = tensor.from_numpy(x1)
        y = autograd.Sum()(x, x1)[0]

        # frontend
        model = sonnx.to_onnx([x, x1], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x, x1])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_sum_cpu(self):
        self._sum_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_sum_gpu(self):
        self._sum_helper(gpu_dev)

    def _Cos_helper(self, dev):
        x = np.array([0.1, -1.0, 0.4, 4.0, -0.9,
                      9.0]).reshape(3, 2).astype(np.float32)
        x = tensor.from_numpy(x)
        y = autograd.Cos()(x)[0]

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_Cos_cpu(self):
        self._Cos_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_Cos_gpu(self):
        self._Cos_helper(gpu_dev)

    def _Cosh_helper(self, dev):
        x = np.array([0.1, -1.0, 0.4, 4.0, -0.9,
                      9.0]).reshape(3, 2).astype(np.float32)
        x = tensor.from_numpy(x)
        y = autograd.Cosh()(x)[0]

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_Cosh_cpu(self):
        self._Cosh_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_Cosh_gpu(self):
        self._Cosh_helper(gpu_dev)

    def _Sin_helper(self, dev):
        x = np.array([0.1, -1.0, 0.4, 4.0, -0.9,
                      9.0]).reshape(3, 2).astype(np.float32)
        x = tensor.from_numpy(x)
        y = autograd.Sin()(x)[0]

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_Sin_cpu(self):
        self._Sin_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_Sin_gpu(self):
        self._Sin_helper(gpu_dev)

    def _Sinh_helper(self, dev):
        x = np.array([0.1, -1.0, 0.4, 4.0, -0.9,
                      9.0]).reshape(3, 2).astype(np.float32)
        x = tensor.from_numpy(x)
        y = autograd.Sinh()(x)[0]

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_Sinh_cpu(self):
        self._Sinh_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_Sinh_gpu(self):
        self._Sinh_helper(gpu_dev)

    def _Tan_helper(self, dev):
        x = np.array([0.1, -1.0, 0.4, 4.0, -0.9,
                      9.0]).reshape(3, 2).astype(np.float32)
        x = tensor.from_numpy(x)
        y = autograd.Tan()(x)[0]

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_Tan_cpu(self):
        self._Tan_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_Tan_gpu(self):
        self._Tan_helper(gpu_dev)

    def _Tanh_helper(self, dev):
        x = np.array([0.1, -1.0, 0.4, 4.0, -0.9,
                      9.0]).reshape(3, 2).astype(np.float32)
        x = tensor.from_numpy(x)
        y = autograd.Tanh()(x)[0]

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_Tanh_cpu(self):
        self._Tanh_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_Tanh_gpu(self):
        self._Tanh_helper(gpu_dev)

    def _Acos_helper(self, dev):
        x = np.array([0.1, -1.0, 0.4, 4.0, -0.9,
                      9.0]).reshape(3, 2).astype(np.float32)
        x = tensor.from_numpy(x)
        y = autograd.Acos()(x)[0]

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_Acos_cpu(self):
        self._Acos_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_Acos_gpu(self):
        self._Acos_helper(gpu_dev)

    def _Acosh_helper(self, dev):
        x = np.array([0.1, -1.0, 0.4, 4.0, -0.9,
                      9.0]).reshape(3, 2).astype(np.float32)
        x = tensor.from_numpy(x)
        y = autograd.Acosh()(x)[0]

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_Acosh_cpu(self):
        self._Acosh_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_Acosh_gpu(self):
        self._Acosh_helper(gpu_dev)

    def _Asin_helper(self, dev):
        x = np.array([0.1, -1.0, 0.4, 4.0, -0.9,
                      9.0]).reshape(3, 2).astype(np.float32)
        x = tensor.from_numpy(x)
        y = autograd.Asin()(x)[0]

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_Asin_cpu(self):
        self._Asin_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_Asin_gpu(self):
        self._Asin_helper(gpu_dev)

    def _Asinh_helper(self, dev):
        x = np.array([0.1, -1.0, 0.4, 4.0, -0.9,
                      9.0]).reshape(3, 2).astype(np.float32)
        x = tensor.from_numpy(x)
        y = autograd.Asinh()(x)[0]

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_Asinh_cpu(self):
        self._Asinh_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_Asinh_gpu(self):
        self._Asinh_helper(gpu_dev)

    def _Atan_helper(self, dev):
        x = np.array([0.1, -1.0, 0.4, 4.0, -0.9,
                      9.0]).reshape(3, 2).astype(np.float32)
        x = tensor.from_numpy(x)
        y = autograd.Atan()(x)[0]

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_Atan_cpu(self):
        self._Atan_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_Atan_gpu(self):
        self._Atan_helper(gpu_dev)

    def _Atanh_helper(self, dev):
        x = np.array([0.1, -1.0, 0.4, 4.0, -0.9,
                      9.0]).reshape(3, 2).astype(np.float32)
        x = tensor.from_numpy(x)
        y = autograd.Atanh()(x)[0]

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_Atanh_cpu(self):
        self._Atanh_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_Atanh_gpu(self):
        self._Atanh_helper(gpu_dev)

    def _SeLu_helper(self, dev):
        x = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                      0.9]).reshape(3, 2).astype(np.float32)
        #y = gamma * (alpha * e^x - alpha) for x <= 0, y = gamma * x for x > 0
        a = 1.67326
        g = 1.0507
        x = tensor.from_numpy(x)
        x.to_device(dev)

        y = autograd.selu(x, a, g)

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_SeLu_cpu(self):
        self._SeLu_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_SeLu_gpu(self):
        self._SeLu_helper(gpu_dev)

    def _ELu_helper(self, dev):
        x = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                      0.9]).reshape(3, 2).astype(np.float32)
        #y = gamma * (alpha * e^x - alpha) for x <= 0, y = gamma * x for x > 0
        a = 1.
        x = tensor.from_numpy(x)
        x.to_device(dev)

        y = autograd.elu(x, a)

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_ELu_cpu(self):
        self._ELu_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_ELu_gpu(self):
        self._ELu_helper(gpu_dev)

    # No Op registered for equal with domain_version of 11
    # def _Equal_helper(self, dev):
    #     x0 = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
    #                    0.9]).reshape(3, 2).astype(np.float32)
    #     x1 = np.array([0, -0.3, 0, 0.1, 0, 0.9]).reshape(3,
    #                                                      2).astype(np.float32)
    #     x0 = tensor.from_numpy(x0)
    #     x1 = tensor.from_numpy(x1)
    #     x0.to_device(dev)
    #     x1.to_device(dev)

    #     y = autograd.equal(x0, x1)

    #     # frontend
    #     model = sonnx.to_onnx([x0, x1], [y])
    #     # print('The model is:\n{}'.format(model))

    #     # backend
    #     sg_ir = sonnx.prepare(model, device=dev)
    #     sg_ir.is_graph = True
    #     y_t = sg_ir.run([x0, x1])

    #     np.testing.assert_array_almost_equal(tensor.to_numpy(y),
    #                                          tensor.to_numpy(y_t[0]),
    #                                          decimal=5)

    # def test_Equal_cpu(self):
    #     self._Equal_helper(cpu_dev)

    # @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    # def test_Equal_gpu(self):
    #     self._Equal_helper(gpu_dev)

    def _Less_helper(self, dev):
        x0 = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                       0.9]).reshape(3, 2).astype(np.float32)
        x1 = np.array([0, -0.3, 0, 0.1, 0, 0.9]).reshape(3,
                                                         2).astype(np.float32)
        x0 = tensor.from_numpy(x0)
        x1 = tensor.from_numpy(x1)
        x0.to_device(dev)
        x1.to_device(dev)

        y = autograd.less(x0, x1)

        # frontend
        model = sonnx.to_onnx([x0, x1], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x0, x1])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_Less_cpu(self):
        self._Less_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_Less_gpu(self):
        self._Less_helper(gpu_dev)

    def _Sign_helper(self, dev):
        x = np.array([0.8, -1.2, 3.3, -3.6, -0.5,
                      0.5]).reshape(3, 2).astype(np.float32)
        x = tensor.from_numpy(x)
        y = autograd.sign(x)

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_Sign_cpu(self):
        self._Sign_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_Sign_gpu(self):
        self._Sign_helper(gpu_dev)

    def _Div_helper(self, dev):
        x0 = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                       0.9]).reshape(3, 2).astype(np.float32)
        x1 = np.array([0, -0.3, 0, 0.1, 0, 0.9]).reshape(3,
                                                         2).astype(np.float32)
        x0 = tensor.from_numpy(x0)
        x1 = tensor.from_numpy(x1)
        x0.to_device(dev)
        x1.to_device(dev)

        y = autograd.div(x0, x1)

        # frontend
        model = sonnx.to_onnx([x0, x1], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x0, x1])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_Div_cpu(self):
        self._Div_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_Div_gpu(self):
        self._Div_helper(gpu_dev)

    def _Sub_helper(self, dev):
        x0 = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                       0.9]).reshape(3, 2).astype(np.float32)
        x1 = np.array([0, -0.3, 0, 0.1, 0, 0.9]).reshape(3,
                                                         2).astype(np.float32)
        x0 = tensor.from_numpy(x0)
        x1 = tensor.from_numpy(x1)
        x0.to_device(dev)
        x1.to_device(dev)

        y = autograd.sub(x0, x1)

        # frontend
        model = sonnx.to_onnx([x0, x1], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x0, x1])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_Sub_cpu(self):
        self._Sub_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_Sub_gpu(self):
        self._Sub_helper(gpu_dev)

    def _Sqrt_helper(self, dev):
        X = np.array([0.1, 1.0, 0.4, 4.0, 0.9,
                      9.0]).reshape(3, 2).astype(np.float32)
        x = tensor.from_numpy(X)
        x.to_device(dev)
        y = autograd.sqrt(x)

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev, init_inputs=X)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_Sqrt_cpu(self):
        self._Sqrt_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_Sqrt_gpu(self):
        self._Sqrt_helper(gpu_dev)

    def _Greater_helper(self, dev):
        x0 = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                       0.9]).reshape(3, 2).astype(np.float32)
        x1 = np.array([0, -0.3, 0, 0.1, 0, 0.9]).reshape(3,
                                                         2).astype(np.float32)
        x0 = tensor.from_numpy(x0)
        x1 = tensor.from_numpy(x1)
        x0.to_device(cpu_dev)
        x1.to_device(cpu_dev)

        y = autograd.greater(x0, x1)

        # frontend
        model = sonnx.to_onnx([x0, x1], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x0, x1])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_Greater_cpu(self):
        self._Greater_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_Greater_gpu(self):
        self._Greater_helper(gpu_dev)

    def _HardSigmoid_helper(self, dev):
        x = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                      0.9]).reshape(3, 2).astype(np.float32)
        a = 0.2
        g = 0.5

        x = tensor.from_numpy(x)
        x.to_device(dev)
        y = autograd.hardsigmoid(x, a, g)

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_HardSigmoid_cpu(self):
        self._HardSigmoid_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_HardSigmoid_gpu(self):
        self._HardSigmoid_helper(gpu_dev)

    def _identity_helper(self, dev):
        x = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                      0.9]).reshape(3, 2).astype(np.float32)
        x = tensor.from_numpy(x)
        x.to_device(dev)

        y = autograd.identity(x)

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_identity_cpu(self):
        self._identity_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_identity_gpu(self):
        self._identity_helper(gpu_dev)

    def _softplus_helper(self, dev):
        x = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                      0.9]).reshape(3, 2).astype(np.float32)
        x = tensor.from_numpy(x)
        x.to_device(dev)

        y = autograd.softplus(x)

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_softplus_cpu(self):
        self._softplus_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_softplus_gpu(self):
        self._softplus_helper(gpu_dev)

    def _softsign_helper(self, dev):
        x = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                      0.9]).reshape(3, 2).astype(np.float32)
        x = tensor.from_numpy(x)
        x.to_device(dev)

        y = autograd.softsign(x)

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_softsign_cpu(self):
        self._softsign_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_softsign_gpu(self):
        self._softsign_helper(gpu_dev)

    def _mean_helper(self, dev):
        x0 = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                       0.9]).reshape(3, 2).astype(np.float32)
        x1 = np.array([0, -0.3, 0, 0.1, 0, 0.9]).reshape(3,
                                                         2).astype(np.float32)
        x0 = tensor.from_numpy(x0)
        x1 = tensor.from_numpy(x1)
        x0.to_device(dev)
        x1.to_device(dev)

        y = autograd.mean(x0, x1)

        # frontend
        model = sonnx.to_onnx([x0, x1], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x0, x1])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_mean_cpu(self):
        self._mean_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_mean_gpu(self):
        self._mean_helper(gpu_dev)

    def _pow_helper(self, dev):
        x0 = np.array([7, 5, 0.2, 0.1, 0.3, 4]).reshape(3, 2).astype(np.float32)
        x1 = np.array([-1.0, 2.0, -1.0, -2.1, 1.0,
                       -2.0]).reshape(3, 2).astype(np.float32)
        x0 = tensor.from_numpy(x0)
        x1 = tensor.from_numpy(x1)
        x0.to_device(dev)
        x1.to_device(dev)

        y = autograd.mean(x0, x1)

        # frontend
        model = sonnx.to_onnx([x0, x1], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x0, x1])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_pow_cpu(self):
        self._pow_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_pow_gpu(self):
        self._pow_helper(gpu_dev)

    def _clip_helper(self, dev):
        x = np.array([-0.9, -0.3, -0.1, 0.1, 0.5,
                      0.9]).reshape(3, 2).astype(np.float32)

        x = tensor.from_numpy(x)
        min = -0.5
        max = 0.5
        x.to_device(dev)

        y = autograd.clip(x, min, max)

        # frontend
        model = sonnx.to_onnx([x, min, max], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])  # min, max has been stored in model

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_clip_cpu(self):
        self._clip_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_clip_gpu(self):
        self._clip_helper(gpu_dev)

    def _prelu_helper(self, dev):
        x = np.array([0.1, -1.0, -0.4, 4.0, -0.9,
                      9.0]).reshape(3, 2).astype(np.float32)
        slope = np.array([0.1, 1.0, 0.4, 4.0, 0.9,
                          9.0]).reshape(3, 2).astype(np.float32)

        x = tensor.from_numpy(x)
        slope = tensor.from_numpy(slope)
        x.to_device(dev)
        slope.to_device(dev)

        y = autograd.prelu(x, slope)

        # frontend
        model = sonnx.to_onnx([x, slope], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x, slope])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_prelu_cpu(self):
        self._prelu_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_prelu_gpu(self):
        self._prelu_helper(gpu_dev)

    def _mul_helper(self, dev):
        x = np.array([0.1, -1.0, 0.4, 4.0, -0.9,
                      9.0]).reshape(3, 2).astype(np.float32)
        x1 = np.array([0.1, 1.0, 0.4, 4.0, 0.9,
                       9.0]).reshape(3, 2).astype(np.float32)
        x = tensor.from_numpy(x)
        x1 = tensor.from_numpy(x1)
        x.to_device(dev)
        x1.to_device(dev)
        y = autograd.mul(x, x1)

        # frontend
        model = sonnx.to_onnx([x, x1], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x, x1])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_mul_cpu(self):
        self._mul_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_mul_gpu(self):
        self._mul_helper(gpu_dev)

    def _transpose_helper(self, dev):
        x = np.random.randn(3, 2, 1)
        y = x.transpose(1, 2, 0)

        x = tensor.from_numpy(x)
        x.to_device(cpu_dev)

        y = autograd.transpose(x, (1, 2, 0))

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_transpose_cpu(self):
        self._transpose_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_transpose_gpu(self):
        self._transpose_helper(gpu_dev)

    def _max_helper(self, dev):
        X0 = np.array([0.1, 0.2, 2.0, 0.0, 0.1,
                       0.2]).reshape(3, 2).astype(np.float32)
        X1 = np.array([1.0, 2.0, 1.0, 2.1, 0.0,
                       2.0]).reshape(3, 2).astype(np.float32)
        x0 = tensor.from_numpy(X0)
        x1 = tensor.from_numpy(X1)
        x0.to_device(dev)
        x1.to_device(dev)

        y = autograd.max(x0, x1)

        # frontend
        model = sonnx.to_onnx([x0, x1], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x0, x1])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_max_cpu(self):
        self._max_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_max_gpu(self):
        self._max_helper(gpu_dev)

    def _min_helper(self, dev):
        X0 = np.array([0.1, 0.2, 2.0, 0.0, 0.1,
                       0.2]).reshape(3, 2).astype(np.float32)
        X1 = np.array([1.0, 2.0, 1.0, 2.1, 0.0,
                       2.0]).reshape(3, 2).astype(np.float32)
        x0 = tensor.from_numpy(X0)
        x1 = tensor.from_numpy(X1)
        x0.to_device(dev)
        x1.to_device(dev)

        y = autograd.min(x0, x1)

        # frontend
        model = sonnx.to_onnx([x0, x1], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x0, x1])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_min_cpu(self):
        self._min_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_min_gpu(self):
        self._min_helper(gpu_dev)

    def _shape_helper(self, dev):
        x = np.array([0.1, -1.0, 0.4, 4.0, -0.9,
                      9.0]).reshape(3, 2).astype(np.float32)
        x = tensor.from_numpy(x)
        x.to_device(dev)

        y = autograd.shape(x)

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_shape_cpu(self):
        self._shape_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_shape_gpu(self):
        self._shape_helper(gpu_dev)

    def _and_helper(self, dev):
        x0 = np.array([0, -0.3, -0.1, 0.1, 0.5,
                       0.9]).reshape(3, 2).astype(np.float32)
        x1 = np.array([0, -0.3, 0, 0.1, 0.5, 0.9]).reshape(3,
                                                           2).astype(np.float32)

        x0 = tensor.from_numpy(x0)
        x1 = tensor.from_numpy(x1)
        x0.to_device(dev)
        x1.to_device(dev)

        y = autograd._and(x0, x1)

        # frontend
        model = sonnx.to_onnx([x0, x1], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x0, x1])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_and_cpu(self):
        self._and_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_and_gpu(self):
        self._and_helper(gpu_dev)

    def _or_helper(self, dev):
        x0 = np.array([1.0, 1.0, 2.0, -3.0, 0,
                       -7.0]).reshape(3, 2).astype(np.float32)
        x1 = np.array([-1.0, 0, 2.0, 4.0, 0,
                       -7.0]).reshape(3, 2).astype(np.float32)

        x0 = tensor.from_numpy(x0)
        x1 = tensor.from_numpy(x1)
        x0.to_device(dev)
        x1.to_device(dev)

        y = autograd._or(x0, x1)
        # frontend
        model = sonnx.to_onnx([x0, x1], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x0, x1])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_or_cpu(self):
        self._or_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_or_gpu(self):
        self._or_helper(gpu_dev)

    def _xor_helper(self, dev):
        x0 = np.array([0, -0.3, -0.1, 0.1, 0.5,
                       9.0]).reshape(3, 2).astype(np.float32)
        x1 = np.array([0, -0.3, 0, 0.1, 0, 0.9]).reshape(3,
                                                         2).astype(np.float32)

        x0 = tensor.from_numpy(x0)
        x1 = tensor.from_numpy(x1)
        x0.to_device(dev)
        x1.to_device(dev)

        y = autograd._xor(x0, x1)

        # frontend
        model = sonnx.to_onnx([x0, x1], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x0, x1])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_xor_cpu(self):
        self._xor_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_xor_gpu(self):
        self._xor_helper(gpu_dev)

    def _not_helper(self, dev):
        x = np.array([1.0, -1.0, 0, -0.1, 0,
                      -7.0]).reshape(3, 2).astype(np.float32)
        x = tensor.from_numpy(x)
        x.to_device(dev)

        y = autograd._not(x)

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_not_cpu(self):
        self._not_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_not_gpu(self):
        self._not_helper(gpu_dev)

    def _negative_helper(self, dev):
        X = np.array([0.1, 0, 0.4, 1. - 4, 0.9,
                      -2.0]).reshape(3, 2).astype(np.float32)
        x = tensor.from_numpy(X)
        x.to_device(dev)

        y = autograd.negative(x)

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_negative_cpu(self):
        self._negative_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_negative_gpu(self):
        self._negative_helper(gpu_dev)

    def _reciprocal_helper(self, dev):
        X = np.array([0.1, 0, 0.4, 1. - 4, 0.9,
                      -2.0]).reshape(3, 2).astype(np.float32)
        x = tensor.from_numpy(X)
        x.to_device(cpu_dev)

        y = autograd.reciprocal(x)
        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_reciprocal_cpu(self):
        self._reciprocal_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_reciprocal_gpu(self):
        self._reciprocal_helper(gpu_dev)

    def _constantOfShape_helper(self, dev):
        X = np.array([4, 3, 2]).astype(np.int64)
        x = tensor.from_numpy(X)
        x.to_device(cpu_dev)

        y = autograd.constant_of_shape(x, 1.)
        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev, init_inputs=[X])
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_constantOfShape_cpu(self):
        self._constantOfShape_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_constantOfShape_gpu(self):
        self._constantOfShape_helper(gpu_dev)

    def _dropout_helper(self, dev):
        X = np.random.randn(3, 4, 5).astype(np.float32)

        x = tensor.from_numpy(X)
        x.to_device(dev)
        y = autograd.dropout(x, 0.5)

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        self.check_shape(
            tensor.to_numpy(y).shape,
            tensor.to_numpy(y_t[0]).shape)

    def test_dropout_cpu(self):
        self._dropout_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_dropout_gpu(self):
        self._dropout_helper(gpu_dev)

    def _reduceSum_helper(self, dev):
        X = np.random.randn(3, 4, 5).astype(np.float32)

        x = tensor.from_numpy(X)
        x.to_device(dev)
        y = autograd.reduce_sum(x, None, 1)

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(
            tensor.to_numpy(y).shape,
            tensor.to_numpy(y_t[0]).shape)

    def test_reduceSum_cpu(self):
        self._reduceSum_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_reduceSum_gpu(self):
        self._reduceSum_helper(gpu_dev)

    def _reduceMean_helper(self, dev):
        X = np.random.randn(3, 4, 5).astype(np.float32)

        x = tensor.from_numpy(X)
        x.to_device(dev)
        y = autograd.reduce_mean(x, None, 1)

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(
            tensor.to_numpy(y).shape,
            tensor.to_numpy(y_t[0]).shape)

    def test_reduceMean_cpu(self):
        self._reduceMean_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_reduceMean_gpu(self):
        self._reduceMean_helper(gpu_dev)

    def _squeeze_helper(self, dev):
        X = np.random.randn(3, 1, 2, 1, 1)

        x = tensor.from_numpy(X)
        x.to_device(dev)
        y = autograd.squeeze(x, [1, 3, 4])

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(
            tensor.to_numpy(y).shape,
            tensor.to_numpy(y_t[0]).shape)

    def test_squeeze_cpu(self):
        self._squeeze_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_squeeze_gpu(self):
        self._squeeze_helper(gpu_dev)

    def _unsqueeze_helper(self, dev):
        X = np.random.randn(3, 2)

        x = tensor.from_numpy(X)
        x.to_device(dev)
        y = autograd.unsqueeze(x, [2, 4, 5])

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(
            tensor.to_numpy(y).shape,
            tensor.to_numpy(y_t[0]).shape)

    def test_unsqueeze_cpu(self):
        self._unsqueeze_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_unsqueeze_gpu(self):
        self._unsqueeze_helper(gpu_dev)

    def _slice_helper(self, dev):
        X = np.random.randn(20, 10, 5).astype(np.float32)
        starts, ends, axes, steps = [0, 0], [3, 10], [0, 1], [1, 1]
        x = tensor.from_numpy(X)
        x.to_device(dev)
        y = autograd.slice(x, starts, ends, axes, steps)

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(
            tensor.to_numpy(y).shape,
            tensor.to_numpy(y_t[0]).shape)

    def test_slice_cpu(self):
        self._slice_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_slice_gpu(self):
        self._slice_helper(gpu_dev)

    # # todo, we don't support muli outputs
    # def _split_helper(self, dev):
    #       X = np.array([1., 2., 3., 4., 5., 6.]).astype(np.float32)
    #       x = tensor.from_numpy(X)
    #       x.to_device(dev)
    #       y = autograd.split(x, 0, (2, 4))

    #       # frontend
    #       model = sonnx.to_onnx([x], [*y])
    #       # print('The model is:\n{}'.format(model))

    #       # backend
    #       sg_ir = sonnx.prepare(model, device=dev)
    #       sg_ir.is_graph = True
    #       y_t = sg_ir.run([x])[0]

    #       np.testing.assert_array_almost_equal(tensor.to_numpy(y).shape, tensor.to_numpy(y_t).shape)

    # def test_split_cpu(self):
    #     self._split_helper(cpu_dev)

    # @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    # def test_split_gpu(self):
    #     self._split_helper(gpu_dev)

    def _gather_helper(self, dev):
        X = np.array([0, 1, 2]).astype(np.float32)
        x = tensor.from_numpy(X)
        x.to_device(dev)
        y = autograd.gather(x, 0, [0, 1, 3])

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(
            tensor.to_numpy(y).shape,
            tensor.to_numpy(y_t[0]).shape)

    def test_gather_cpu(self):
        self._gather_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_gather_gpu(self):
        self._gather_helper(gpu_dev)

    def _tile_helper(self, dev):
        X = np.array([0, 1, 2]).astype(np.float32)
        x = tensor.from_numpy(X)
        x.to_device(dev)
        y = autograd.tile(x, [2, 2])

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(
            tensor.to_numpy(y).shape,
            tensor.to_numpy(y_t[0]).shape)

    def test_tile_cpu(self):
        self._tile_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_tile_gpu(self):
        self._tile_helper(gpu_dev)

    def _nonzero_helper(self, dev):
        X = np.array([[1, 0], [1, 1]]).astype(np.float32)
        x = tensor.from_numpy(X)
        x.to_device(dev)
        y = autograd.nonzero(x)

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(
            tensor.to_numpy(y).shape,
            tensor.to_numpy(y_t[0]).shape)

    def test_nonzero_cpu(self):
        self._nonzero_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_nonzero_gpu(self):
        self._nonzero_helper(gpu_dev)

    def _cast_helper(self, dev):
        X = np.array([[1, 0], [1, 1]]).astype(np.float32)
        x = tensor.from_numpy(X)
        x.to_device(dev)
        y = autograd.cast(x, tensor.int32)

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(
            tensor.to_numpy(y).shape,
            tensor.to_numpy(y_t[0]).shape)

    def test_cast_cpu(self):
        self._cast_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_cast_gpu(self):
        self._cast_helper(gpu_dev)

    def _onehot_helper(self, dev):
        axisValue = 1
        on_value = 3
        off_value = 1
        output_type = np.float32
        indices = np.array([[1, 9], [2, 4]], dtype=np.float32)
        depth = np.array([10], dtype=np.float32)
        values = np.array([off_value, on_value], dtype=output_type)

        x = tensor.from_numpy(indices)
        x.to_device(dev)
        y = autograd.onehot(axisValue, x, depth, values)

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x])

        self.check_shape(
            tensor.to_numpy(y).shape,
            tensor.to_numpy(y_t[0]).shape)

    def test_onehot_cpu(self):
        self._onehot_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_onehot_gpu(self):
        self._onehot_helper(gpu_dev)

    def _inference_helper(self, dev):
        x = tensor.Tensor(shape=(2, 3, 3, 3), device=dev)
        x.gaussian(0.0, 1.0)

        conv1 = layer.Conv2d(1, 2)
        conv2 = layer.Conv2d(1, 2)

        class MyLayer(layer.Layer):

            def __init__(self, conv1, conv2):
                super(MyLayer, self).__init__()
                self.conv1 = conv1
                self.conv2 = conv2

            def forward(self, inputs):
                x = self.conv1(inputs)
                x = self.conv2(x)
                return x

        y = MyLayer(conv1, conv2)(x)
        x1 = conv1(x)
        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        y_t = sg_ir.run([x], last_layers=-1)

        np.testing.assert_array_almost_equal(tensor.to_numpy(x1),
                                             tensor.to_numpy(y_t[0]),
                                             decimal=5)

    def test_inference_cpu(self):
        self._inference_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_inference_gpu(self):
        self._inference_helper(gpu_dev)

    def _retraining_helper(self, dev):
        # forward
        x = tensor.Tensor(shape=(2, 3, 3, 3), device=dev)
        x.gaussian(0.0, 1.0)

        class MyLayer(layer.Layer):

            def __init__(self):
                super(MyLayer, self).__init__()
                self.conv1 = layer.Conv2d(1, 2)
                self.conv2 = layer.Conv2d(1, 2)

            def forward(self, inputs):
                x = self.conv1(inputs)
                x = self.conv2(x)
                x = autograd.flatten(x)
                return x

        y = MyLayer()(x)
        y_t = tensor.Tensor(shape=(2, 1), device=dev)
        y_t.gaussian(0.0, 1.0)
        loss = autograd.MeanSquareError(y_t)(y)[0]
        # backward
        sgd = opt.SGD(lr=0.01)
        for p, gp in autograd.backward(loss):
            sgd.apply(p.name, p, gp)
        sgd.step()

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True
        # forward
        y_o = sg_ir.run([x])[0]
        # backward
        loss = autograd.MeanSquareError(y_t)(y_o)[0]
        sgd = opt.SGD(lr=0.01)
        for p, gp in autograd.backward(loss):
            sgd.apply(p.name, p, gp)
        sgd.step()

    def test_retraining_cpu(self):
        self._retraining_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_retraining_gpu(self):
        self._retraining_helper(gpu_dev)

    def _transfer_learning_helper(self, dev):
        # forward
        x = tensor.Tensor(shape=(2, 3, 3, 3), device=dev)
        x.gaussian(0.0, 1.0)

        class MyLayer(layer.Layer):

            def __init__(self):
                super(MyLayer, self).__init__()
                self.conv1 = layer.Conv2d(1, 2)

            def forward(self, inputs):
                x = self.conv1(inputs)
                x = autograd.flatten(x)
                return x

        y = MyLayer()(x)
        y_t = tensor.Tensor(shape=(2, 4), device=dev)
        y_t.gaussian(0.0, 1.0)
        loss = autograd.MeanSquareError(y_t)(y)[0]
        # backward
        sgd = opt.SGD(lr=0.01)
        for p, gp in autograd.backward(loss):
            sgd.apply(p.name, p, gp)
        sgd.step()

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=dev)
        sg_ir.is_graph = True

        # forward
        class MyLayer2(layer.Layer):

            def __init__(self, sg_ir):
                super(MyLayer2, self).__init__()
                self.sg_ir = sg_ir
                for node, operator in self.sg_ir.layers:
                    self.__dict__[node.name] = operator
                self.conv2 = layer.Conv2d(1, 2)

            def forward(self, inputs):
                x = self.sg_ir.run(inputs, last_layers=-1)[0]
                x = self.conv2(inputs)
                x = autograd.flatten(x)
                return x

        y_o = MyLayer()(x)
        # backward
        y_ot = tensor.Tensor(shape=(2, 1), device=dev)
        y_ot.gaussian(0.0, 1.0)
        loss = autograd.MeanSquareError(y_ot)(y_o)[0]
        sgd = opt.SGD(lr=0.01)
        for p, gp in autograd.backward(loss):
            sgd.apply(p.name, p, gp)
        sgd.step()

    def test_transfer_learning_cpu(self):
        self._transfer_learning_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_transfer_learning_gpu(self):
        self._transfer_learning_helper(gpu_dev)


if __name__ == '__main__':
    unittest.main()
