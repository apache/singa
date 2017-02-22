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

import numpy as np

from singa import metric
from singa import tensor


class TestPrecision(unittest.TestCase):
    def setUp(self):
        x_np = np.asarray([[0.7, 0.2, 0.1],
                           [0.2, 0.4, 0.5],
                           [0.2,0.4,0.4]],
                          dtype=np.float32)

        y_np = np.asarray([[1, 0, 1],
                           [0, 1, 1],
                           [1, 0, 0]],
                           dtype=np.int32)

        self.prcs = metric.Precision(top_k=2)
        self.x = tensor.from_numpy(x_np)
        self.y = tensor.from_numpy(y_np)


    def test_forward(self):
        p = self.prcs.forward(self.x,self.y)
        self.assertAlmostEqual(tensor.to_numpy(p)[0], 0.5)
        self.assertAlmostEqual(tensor.to_numpy(p)[1], 1)
        self.assertAlmostEqual(tensor.to_numpy(p)[2], 0)


    def test_evaluate(self):
        e = self.prcs.evaluate(self.x,self.y)
        self.assertAlmostEqual(e, (0.5 + 1 + 0) / 3)

if __name__ == '__main__':
    unittest.main()
