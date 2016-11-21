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
                - math.log(1.0/(1+math.exp(1))) - math.log(0.5) -math.log(0.5),
                5);

    def test_mult_inputs(self):
        ffn = net.FeedForwardNet(loss.SoftmaxCrossEntropy())
        s1 = ffn.add(layer.Activation('relu1', input_sample_shape=(2,)), [])
        s2 = ffn.add(layer.Activation('relu2', input_sample_shape=(2,)), [])
        ffn.add(layer.Merge('merge', input_sample_shape=(2,)), [s1, s2])
        x1 = tensor.Tensor((2, 2))
        x1.set_value(1.1)
        x2 = tensor.Tensor((2, 2))
        x2.set_value(0.9)
        out = ffn.forward(False, {'relu1':x1, 'relu2':x2})
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
        out = ffn.forward(False, {'relu1':x1, 'relu2':x2})
        out = tensor.to_numpy(out['split1'])
        self.assertAlmostEqual(np.average(out), 2)


if __name__ == '__main__':
    unittest.main()
