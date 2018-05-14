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

import unittest
import numpy as np

from singa import loss
from singa import tensor


class TestLoss(unittest.TestCase):
    def setUp(self):
        self.x_np = np.asarray([[0.9, 0.2, 0.1],
                                [0.1, 0.4, 0.5],
                                [0.2, 0.4, 0.4]],
                               dtype=np.float32)

        self.y_np = np.asarray([[1, 0, 1],
                                [0, 1, 1],
                                [1, 0, 0]],
                               dtype=np.float32)

        self.x = tensor.from_numpy(self.x_np)
        self.y = tensor.from_numpy(self.y_np)

    def test_sigmoid_cross_entropy(self):
        sig = loss.SigmoidCrossEntropy()
        l1 = sig.forward(True, self.x, self.y)
        sig.backward()
        l2 = sig.evaluate(True, self.x, self.y)

        p = 1.0 / (1 + np.exp(-self.x_np))
        l = - (self.y_np * np.log(p) + (1 - self.y_np) * np.log(1 - p))
        self.assertAlmostEqual(l1.l1(), l2)
        self.assertAlmostEqual(l1.l1(), np.average(l))

    def test_squared_error(self):
        sqe = loss.SquaredError()
        l1 = sqe.forward(True, self.x, self.y)
        sqe.backward()
        l2 = sqe.evaluate(True, self.x, self.y)

        l = 0.5 * (self.y_np - self.x_np) ** 2
        self.assertAlmostEqual(l1.l1(), l2)
        self.assertAlmostEqual(l1.l1(), np.average(l))
        
    def test_softmax_cross_entropy(self):
        sce = loss.SoftmaxCrossEntropy()
        l1 = sce.forward(True, self.x, self.y)
        sce.backward()
        l2 = sce.evaluate(True, self.x, self.y)

        self.assertAlmostEqual(l1.l1(), l2)


if __name__ == '__main__':
    unittest.main()
