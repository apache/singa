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
from __future__ import division
from builtins import zip

import unittest
import math
import numpy as np

from singa import net
from singa import layer
from singa import tensor
from singa import loss

layer.engine = 'singacpp'
# net.verbose = True


class TestFeedForwardNet(unittest.TestCase):

    def test_single_input_output(self):
        ffn = net.FeedForwardNet(loss.SoftmaxCrossEntropy())
        ffn.add(layer.Activation('relu1', input_sample_shape=(2,)))
        ffn.add(layer.Activation('relu2'))
        x = np.array([[-1, 1], [1, 1], [-1, -2]], dtype=np.float32)
        x = tensor.from_numpy(x)
        y = tensor.Tensor((3,))
        y.set_value(0)
        out, _ = ffn.evaluate(x, y)
        self.assertAlmostEqual(out * 3,
                               - math.log(1.0 / (1+math.exp(1))) -
                               math.log(0.5) - math.log(0.5),
                               5)

    def test_mult_inputs(self):
        ffn = net.FeedForwardNet(loss.SoftmaxCrossEntropy())
        s1 = ffn.add(layer.Activation('relu1', input_sample_shape=(2,)), [])
        s2 = ffn.add(layer.Activation('relu2', input_sample_shape=(2,)), [])
        ffn.add(layer.Merge('merge', input_sample_shape=(2,)), [s1, s2])
        x1 = tensor.Tensor((2, 2))
        x1.set_value(1.1)
        x2 = tensor.Tensor((2, 2))
        x2.set_value(0.9)
        out = ffn.forward(False, {'relu1': x1, 'relu2': x2})
        out = tensor.to_numpy(out)
        self.assertAlmostEqual(np.average(out), 2)

    def test_mult_outputs(self):
        ffn = net.FeedForwardNet(loss.SoftmaxCrossEntropy())
        s1 = ffn.add(layer.Activation('relu1', input_sample_shape=(2,)), [])
        s2 = ffn.add(layer.Activation('relu2', input_sample_shape=(2,)), [])
        ffn.add(layer.Merge('merge', input_sample_shape=(2,)), [s1, s2])
        split = ffn.add(layer.Split('split', 2))
        ffn.add(layer.Dummy('split1'), split)
        ffn.add(layer.Dummy('split2'), split)
        x1 = tensor.Tensor((2, 2))
        x1.set_value(1.1)
        x2 = tensor.Tensor((2, 2))
        x2.set_value(0.9)
        out = ffn.forward(False, {'relu1': x1, 'relu2': x2})
        out = tensor.to_numpy(out['split1'])
        self.assertAlmostEqual(np.average(out), 2)

    def test_save_load(self):
        ffn = net.FeedForwardNet(loss.SoftmaxCrossEntropy())
        ffn.add(layer.Conv2D('conv', 4, 3, input_sample_shape=(3, 12, 12)))
        ffn.add(layer.Flatten('flat'))
        # ffn.add(layer.BatchNorm('bn'))
        ffn.add(layer.Dense('dense', num_output=4))
        for pname, pval in zip(ffn.param_names(), ffn.param_values()):
            pval.set_value(0.1)
        ffn.save('test_snaphost')
        ffn.save('test_pickle', use_pickle=True)

        ffn.load('test_snaphost')
        ffn.load('test_pickle', use_pickle=True)

    def test_train_one_batch(self):
        ffn = net.FeedForwardNet(loss.SoftmaxCrossEntropy())
        ffn.add(layer.Conv2D('conv', 4, 3, input_sample_shape=(3, 12, 12)))
        ffn.add(layer.Flatten('flat'))
        ffn.add(layer.Dense('dense', num_output=4))
        for pname, pval in zip(ffn.param_names(), ffn.param_values()):
            pval.set_value(0.1)
        x = tensor.Tensor((4, 3, 12, 12))
        x.gaussian(0, 0.01)
        y = np.asarray([[1, 0, 0],
                        [0, 0, 1],
                        [0, 0, 1],
                        [0, 1, 0]], dtype=np.int32)
        y = tensor.from_numpy(y)
        o = ffn.forward(True, x)
        ffn.loss.forward(True, o, y)
        g = ffn.loss.backward()
        for pname, pvalue, pgrad, _ in ffn.backward(g):
            self.assertEqual(len(pvalue), len(pgrad))
            for p, g in zip(pvalue, pgrad):
                self.assertEqual(p.size(), g.size())


if __name__ == '__main__':
    unittest.main()
