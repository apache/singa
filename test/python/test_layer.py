#
# Copyright 2015 The Apache Software Foundation
# 
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 

import sys
import os
import unittest
import numpy as np

#sys.path.append(os.path.join(os.path.dirname(__file__), '../../build/python'))

from singa import layer
from singa import device
from singa import tensor
from singa.proto import model_pb2


def _tuple_to_string(t):
    lt = [str(x) for x in t]
    return '(' + ', '.join(lt) + ')'


class TestPythonLayer(unittest.TestCase):

    def check_shape(self, actual, expect):
        self.assertEqual(actual, expect, 'shape mismatch, actual shape is %s'
                         ' exepcted is %s' % (_tuple_to_string(actual),
                                              _tuple_to_string(expect))
                         )

    def setUp(self):
        layer.engine='singacpp'
        self.w = {'init': 'Xavier', 'regularizer': 1e-4}
        self.b = {'init': 'Constant', 'value': 0}
        self.sample_shape = None

    def test_conv2D_shape(self):
        in_sample_shape = (3, 224, 224)
        conv = layer.Conv2D('conv', 64, 3, 1, W_specs=self.w, b_specs=self.b,
                            input_sample_shape=in_sample_shape)
        out_sample_shape = conv.get_output_sample_shape()
        self.check_shape(out_sample_shape, (64, 224, 224))

    def test_conv2D_forward_backward(self):
        in_sample_shape = (1, 3, 3)
        conv = layer.Conv2D('conv', 1, 3, 2, W_specs=self.w, b_specs=self.b,
                            pad=1, input_sample_shape=in_sample_shape)
        # cuda = device.create_cuda_gpu()
        # conv.to_device(cuda)
        params = conv.param_values()

        raw_x = np.arange(9, dtype=np.float32) + 1
        x = tensor.from_numpy(raw_x)
        x.reshape((1, 1, 3, 3))
        w = np.array([1, 1, 0, 0, 0, -1, 0, 1, 0], dtype=np.float32)
        params[0].copy_from_numpy(w)
        params[1].set_value(1.0)

        # x.to_device(cuda)
        y = conv.forward(model_pb2.kTrain, x)
        # y.to_host()
        npy = tensor.to_numpy(y).flatten()

        self.assertAlmostEqual(3.0, npy[0])
        self.assertAlmostEqual(7.0, npy[1])
        self.assertAlmostEqual(-3.0, npy[2])
        self.assertAlmostEqual(12.0, npy[3])

        dy = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32).reshape(y.shape)
        grad = tensor.from_numpy(dy)
        # grad.to_device(cuda)
        (dx, [dw, db]) = conv.backward(model_pb2.kTrain, grad)
        dx.to_host()
        dw.to_host()
        dx = tensor.to_numpy(dx).flatten()
        dw = tensor.to_numpy(dw).flatten()
        dy = dy.flatten()
        self.assertAlmostEquals(dy[0] * w[4], dx[0])
        self.assertAlmostEquals(dy[0] * w[5] + dy[1] * w[3], dx[1])
        self.assertAlmostEquals(dy[1] * w[4], dx[2])
        self.assertAlmostEquals(dy[0] * w[7] + dy[2] * w[1], dx[3])
        self.assertAlmostEquals(
            dy[0] *
            w[8] +
            dy[1] *
            w[6] +
            dy[2] *
            w[2] +
            dy[3] *
            w[0],
            dx[4])
        self.assertAlmostEquals(dy[1] * w[7] + dy[3] * w[1], dx[5])
        self.assertAlmostEquals(dy[2] * w[4], dx[6])
        self.assertAlmostEquals(dy[2] * w[5] + dy[3] * w[3], dx[7])
        self.assertAlmostEquals(dy[3] * w[4], dx[8])

        self.assertAlmostEquals(dy[3] * raw_x[4], dw[0])
        self.assertAlmostEquals(dy[3] * raw_x[5] + dy[2] * raw_x[3], dw[1])
        self.assertAlmostEquals(dy[2] * raw_x[4], dw[2])
        self.assertAlmostEquals(dy[1] * raw_x[1] + dy[3] * raw_x[7], dw[3])
        self.assertAlmostEquals(
            dy[0] *
            raw_x[0] +
            dy[1] *
            raw_x[2] +
            dy[2] *
            raw_x[6] +
            dy[3] *
            raw_x[8],
            dw[4], 5)
        self.assertAlmostEquals(dy[0] * raw_x[1] + dy[2] * raw_x[7], dw[5])
        self.assertAlmostEquals(dy[1] * raw_x[4], dw[6])
        self.assertAlmostEquals(dy[0] * raw_x[3] + dy[1] * raw_x[5], dw[7])
        self.assertAlmostEquals(dy[0] * raw_x[4], dw[8])

    def test_conv1D(self):
        in_sample_shape = (224,)
        conv = layer.Conv1D('conv', 64, 3, 1, W_specs=self.w, b_specs=self.b,
                            pad=1, input_sample_shape=in_sample_shape)
        out_sample_shape = conv.get_output_sample_shape()
        self.check_shape(out_sample_shape, (64, 224,))

    def test_max_pooling2D(self):
        in_sample_shape = (64, 224, 224)
        pooling = layer.MaxPooling2D('pool', 3, 2,
                                     input_sample_shape=in_sample_shape)
        out_sample_shape = pooling.get_output_sample_shape()
        self.check_shape(out_sample_shape, (64, 112, 112))

    def test_max_pooling1D(self):
        in_sample_shape = (224,)
        pooling = layer.MaxPooling1D('pool', 3, 2,
                                     input_sample_shape=in_sample_shape)
        out_sample_shape = pooling.get_output_sample_shape()
        self.check_shape(out_sample_shape, (112,))

    def test_avg_pooling2D(self):
        in_sample_shape = (64, 224, 224)
        pooling = layer.AvgPooling2D('pool', 3, 2,
                                     input_sample_shape=in_sample_shape)
        out_sample_shape = pooling.get_output_sample_shape()
        self.check_shape(out_sample_shape, (64, 112, 112))

    def test_avg_pooling1D(self):
        in_sample_shape = (224,)
        pooling = layer.AvgPooling1D('pool', 3, 2,
                                     input_sample_shape=in_sample_shape)
        out_sample_shape = pooling.get_output_sample_shape()
        self.check_shape(out_sample_shape, (112,))

    def test_batch_normalization(self):
        in_sample_shape = (3, 224, 224)
        bn = layer.BatchNormalization('bn', input_sample_shape=in_sample_shape)
        out_sample_shape = bn.get_output_sample_shape()
        self.check_shape(out_sample_shape, in_sample_shape)

    def test_lrn(self):
        in_sample_shape = (3, 224, 224)
        lrn = layer.LRN('lrn', input_sample_shape=in_sample_shape)
        out_sample_shape = lrn.get_output_sample_shape()
        self.check_shape(out_sample_shape, in_sample_shape)

    def test_dense(self):
        dense = layer.Dense('ip', 32, input_sample_shape=(64,))
        out_sample_shape = dense.get_output_sample_shape()
        self.check_shape(out_sample_shape, (32,))

    def test_dropout(self):
        input_sample_shape = (64, 1, 12)
        dropout = layer.Dropout('drop', input_sample_shape=input_sample_shape)
        out_sample_shape = dropout.get_output_sample_shape()
        self.check_shape(out_sample_shape, input_sample_shape)

    def test_activation(self):
        input_sample_shape = (64, 1, 12)
        act = layer.Activation('act', input_sample_shape=input_sample_shape)
        out_sample_shape = act.get_output_sample_shape()
        self.check_shape(out_sample_shape, input_sample_shape)

    def test_softmax(self):
        input_sample_shape = (12,)
        softmax = layer.Softmax('soft', input_sample_shape=input_sample_shape)
        out_sample_shape = softmax.get_output_sample_shape()
        self.check_shape(out_sample_shape, input_sample_shape)

    def test_flatten(self):
        input_sample_shape = (64, 1, 12)
        flatten = layer.Flatten('flat', input_sample_shape=input_sample_shape)
        out_sample_shape = flatten.get_output_sample_shape()
        self.check_shape(out_sample_shape, (64 * 1 * 12, ))

        flatten = layer.Flatten('flat', axis=2,
                                input_sample_shape=input_sample_shape)
        out_sample_shape = flatten.get_output_sample_shape()
        self.check_shape(out_sample_shape, (12,))


if __name__ == '__main__':
    unittest.main()
