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
from singa import sonnx
from singa import opt

import onnx
from onnx import (defs, checker, helper, numpy_helper, mapping,
                  ModelProto, GraphProto, NodeProto, AttributeProto, TensorProto, OperatorSetIdProto)
from onnx.helper import make_tensor, make_tensor_value_info, make_node, make_graph

from cuda_helper import gpu_dev, cpu_dev

import numpy as np

autograd.training = True


class TestPythonOnnxFrontend(unittest.TestCase):

    def test_conv2d(self):
        x = tensor.Tensor(shape=(2, 3, 3, 3), device=gpu_dev)
        x.gaussian(0.0, 1.0)
        y = autograd.Conv2d(3, 1, 2)(x)

        # frontend
        model = sonnx.to_onnx([x], [y])
        # backend
        sg_ir = sonnx.prepare(model, device=gpu_dev)
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y), tensor.to_numpy(y_t[0]), decimal=5)

    def test_relu(self):
        X = np.array([0.8, -1.2, 3.3, -3.6, -0.5, 0.5]).reshape(3, 2).astype(np.float32)
        XT = np.array([0.8, 0, 3.3, 0, 0, 0.5]).reshape(3, 2).astype(np.float32)
        x = tensor.from_numpy(X)
        x.to_device(gpu_dev)
        y = autograd.ReLU()(x)[0]

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=gpu_dev)
        y_t = sg_ir.run([x])
        np.testing.assert_array_almost_equal(tensor.to_numpy(y), tensor.to_numpy(y_t[0]), decimal=5)

    def test_avg_pool(self):
        x = tensor.Tensor(shape=(2, 3, 3, 3), device=gpu_dev)
        x.gaussian(0.0, 1.0)
        y = autograd.AvgPool2d(3, 1, 2)(x)

        # frontend
        model = sonnx.to_onnx([x], [y])

        # backend
        sg_ir = sonnx.prepare(model, device=gpu_dev)
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y), tensor.to_numpy(y_t[0]), decimal=5)

    def test_softmax(self):
        X = np.array([[-1, 0, 1]]).astype(np.float32)
        x = tensor.from_numpy(X)
        x.to_device(gpu_dev)
        y = autograd.SoftMax()(x)[0]

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=gpu_dev)
        y_t = sg_ir.run([x])
        np.testing.assert_array_almost_equal(tensor.to_numpy(y), tensor.to_numpy(y_t[0]), decimal=5)

    def test_sigmoid(self):
        X = np.array([[-1, 0, 1]]).astype(np.float32)
        x = tensor.from_numpy(X)
        x.to_device(gpu_dev)
        y = autograd.Sigmoid()(x)[0]

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=gpu_dev)
        y_t = sg_ir.run([x])
        np.testing.assert_array_almost_equal(tensor.to_numpy(y), tensor.to_numpy(y_t[0]), decimal=5)

    def test_add(self):
        X1 = np.random.randn(3, 4, 5).astype(np.float32)
        X2 = np.random.randn(3, 4, 5).astype(np.float32)

        x1 = tensor.from_numpy(X1)
        x2 = tensor.from_numpy(X2)
        x1.to_device(gpu_dev)
        x2.to_device(gpu_dev)
        y = autograd.Add()(x1, x2)[0]

        # frontend
        model = sonnx.to_onnx([x1, x2], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=gpu_dev)
        y_t = sg_ir.run([x1, x2])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y), tensor.to_numpy(y_t[0]), decimal=5)

    def test_concat(self):
        X1 = np.random.randn(3, 4, 5).astype(np.float32)
        X2 = np.random.randn(3, 4, 5).astype(np.float32)

        x1 = tensor.from_numpy(X1)
        x2 = tensor.from_numpy(X2)
        x1.to_device(gpu_dev)
        x2.to_device(gpu_dev)
        y = autograd.Concat()(x1, x2)[0]

        # frontend
        model = sonnx.to_onnx([x1, x2], [y])

        # backend
        sg_ir = sonnx.prepare(model, device=gpu_dev)
        y_t = sg_ir.run([x1, x2])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y), tensor.to_numpy(y_t[0]), decimal=5)

    def test_matmul(self):
        X1 = np.random.randn(4, 5).astype(np.float32)
        X2 = np.random.randn(5, 4).astype(np.float32)

        x1 = tensor.from_numpy(X1)
        x2 = tensor.from_numpy(X2)
        x1.to_device(gpu_dev)
        x2.to_device(gpu_dev)

        y = autograd.Matmul()(x1, x2)[0]

        # frontend
        model = sonnx.to_onnx([x1, x2], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=gpu_dev)
        y_t = sg_ir.run([x1, x2])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y), tensor.to_numpy(y_t[0]), decimal=5)

    def test_max_pool(self):
        x = tensor.Tensor(shape=(2, 3, 4, 4), device=gpu_dev)
        x.gaussian(0.0, 1.0)
        y = autograd.MaxPool2d(2, 2, 0)(x)

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=gpu_dev)
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y), tensor.to_numpy(y_t[0]), decimal=5)

    def test_batch_norm(self):
        x = tensor.Tensor(shape=(2, 3, 3, 3), device=gpu_dev)
        x.gaussian(0.0, 1.0)
        y = autograd.BatchNorm2d(3)(x)

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=gpu_dev)
        y_t = sg_ir.run([x])

        np.testing.assert_array_almost_equal(tensor.to_numpy(y), tensor.to_numpy(y_t[0]), decimal=5)

    def test_inference(self):
        x = tensor.Tensor(shape=(2, 3, 3, 3), device=gpu_dev)
        x.gaussian(0.0, 1.0)
        x1 = autograd.Conv2d(3, 1, 2)(x)
        y = autograd.Conv2d(1, 1, 2)(x1)

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=gpu_dev)
        y_t = sg_ir.run([x], last_layers=-1)

        np.testing.assert_array_almost_equal(tensor.to_numpy(x1), tensor.to_numpy(y_t[0]), decimal=5)

    def test_retraining(self):
        # forward
        x = tensor.Tensor(shape=(2, 3, 3, 3), device=gpu_dev)
        x.gaussian(0.0, 1.0)
        x1 = autograd.Conv2d(3, 1, 2)(x)
        x2 = autograd.Conv2d(1, 1, 2)(x1)
        y = autograd.Flatten()(x2)[0]
        y_t = tensor.Tensor(shape=(2, 1), device=gpu_dev)
        y_t.gaussian(0.0, 1.0)
        loss = autograd.MeanSquareError()(y, y_t)[0]
        # backward
        sgd = opt.SGD(lr=0.01)
        for p, gp in autograd.backward(loss):
            sgd.update(p, gp)
        sgd.step()

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=gpu_dev)
        for idx, tens in sg_ir.tensor_map.items():
            tens.stores_grad = False
            sg_ir.tensor_map[idx] = tens
        # forward
        y_o = sg_ir.run([x])[0]
        # backward
        loss = autograd.MeanSquareError()(y_o, y_t)[0]
        sgd = opt.SGD(lr=0.01)
        for p, gp in autograd.backward(loss):
            sgd.update(p, gp)
        sgd.step()


    def test_transfer_learning(self):
        # forward
        x = tensor.Tensor(shape=(2, 3, 3, 3), device=gpu_dev)
        x.gaussian(0.0, 1.0)
        x1 = autograd.Conv2d(3, 1, 2)(x)
        y = autograd.Flatten()(x1)[0]
        y_t = tensor.Tensor(shape=(2, 4), device=gpu_dev)
        y_t.gaussian(0.0, 1.0)
        loss = autograd.MeanSquareError()(y, y_t)[0]
        # backward
        sgd = opt.SGD(lr=0.01)
        for p, gp in autograd.backward(loss):
            sgd.update(p, gp)
        sgd.step()

        # frontend
        model = sonnx.to_onnx([x], [y])
        # print('The model is:\n{}'.format(model))

        # backend
        sg_ir = sonnx.prepare(model, device=gpu_dev)
        for idx, tens in sg_ir.tensor_map.items():
            tens.stores_grad = False
            sg_ir.tensor_map[idx] = tens
        # forward
        x1 = sg_ir.run([x], last_layers=-1)[0]
        x2 = autograd.Conv2d(1, 1, 2)(x1)
        y_o = autograd.Flatten()(x2)[0]
        # backward
        y_ot = tensor.Tensor(shape=(2, 1), device=gpu_dev)
        y_ot.gaussian(0.0, 1.0)
        loss = autograd.MeanSquareError()(y_o, y_ot)[0]
        sgd = opt.SGD(lr=0.01)
        for p, gp in autograd.backward(loss):
            sgd.update(p, gp)
        sgd.step()

if __name__ == '__main__':
    unittest.main()
