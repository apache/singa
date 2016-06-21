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
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# =============================================================================
import sys
import os
import math
import unittest
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__),'../../build/python'))

from singa.tensor import *
from singa.device import *

from singa.proto.core_pb2 import *


class TestTensorMethods(unittest.TestCase):

    def setUp(self):
        self.shape = (3, 2)
        self.t = Tensor(self.shape)
        self.s = Tensor(self.shape)

    def test_tensor_fields(self):
        t = self.t
        shape = self.shape
        self.assertTupleEqual(t.shape(), shape)
        self.assertEqual(t.shape(0), shape[0])
        self.assertEqual(t.shape(1), shape[1])
        self.assertEqual(product(shape), 3*2)
        self.assertEqual(t.ndim(), 2)
        self.assertEqual(t.size(), 3*2)
        self.assertEqual(t.memsize(), 3*2*sizeof(kFloat32))
        self.assertFalse(t.is_transpose())

    def test_unary_operators(self):
        t = self.t
        arr = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=np.float32)
        t.copy_from_numpy(arr)
        npary = to_numpy(t)
        self.assertAlmostEqual(npary[0, 0], arr[0, 0])
        self.assertAlmostEqual(npary[0, 1], arr[0, 1])
        self.assertAlmostEqual(npary[2, 1], arr[2, 1])
        t += 1.23
        npary = to_numpy(t)
        self.assertAlmostEqual(npary[0, 0], 1.0+1.23)
        t -= 0.23
        npary = to_numpy(t)
        self.assertAlmostEqual(npary[0, 0], 2.23-0.23)
        t *= 2.5
        npary = to_numpy(t)
        self.assertAlmostEqual(npary[0, 0], (2.23-0.23)*2.5)
        t /= 2
        npary = to_numpy(t)
        self.assertAlmostEqual(npary[0, 0], (2.23-0.23)*2.5/2)

    def test_binary_operators(self):
        t = self.t
        arr = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=np.float32)
        t.copy_from_numpy(arr)
        s = self.s
        arr = np.array([[4.0, 3.0], [3.0, 2.0], [2.0, 1.0]], dtype=np.float32)
        s.copy_from_numpy(arr)
        a = t + s
        self.assertAlmostEqual(to_numpy(a)[0, 0], 1.0+4.0)
        a = t - s
        self.assertAlmostEqual(to_numpy(a)[0, 0], 1.0-4.0)
        a = t * s
        self.assertAlmostEqual(to_numpy(a)[0, 0], 1.0*4.0)
        a = t / s
        self.assertAlmostEqual(to_numpy(a)[0, 0], 1.0/4.0)

    def test_comparison_operators(self):
        t = self.t
        t.set_value(3.45)
        a = t < 3.45
        self.assertEqual(to_numpy(a)[0, 0], 0)
        a = t <= 3.45
        self.assertEqual(to_numpy(a)[0, 0], 1)
        a = t > 3.45
        self.assertEqual(to_numpy(a)[0, 0], 0)
        a = t >= 3.45
        self.assertEqual(to_numpy(a)[0, 0], 1)
        a = lt(t, 3.45)
        self.assertEqual(to_numpy(a)[0, 0], 0)
        a = le(t, 3.45)
        self.assertEqual(to_numpy(a)[0, 0], 1)
        a = gt(t, 3.45)
        self.assertEqual(to_numpy(a)[0, 0], 0)
        a = ge(t, 3.45)
        self.assertEqual(to_numpy(a)[0, 0], 1)

    def test_tensor_manipulation(self):
        t = self.t
        arr = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        t.copy_from_numpy(arr)
        s = Tensor((3, 1))
        arr = np.array([7, 8, 9], dtype=np.float32)
        s.copy_from_numpy(arr)
        t.add_column(s)
        self.assertEqual(to_numpy(t)[0, 0], 1+7)
        self.assertEqual(to_numpy(t)[1, 0], 3+8)
        self.assertEqual(to_numpy(t)[1, 1], 4+8)

        arr = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        t.copy_from_numpy(arr)
        add_column(2, s, 3, t)
        self.assertEqual(to_numpy(t)[0, 0], 3*1+2*7)
        self.assertEqual(to_numpy(t)[1, 0], 3*3+2*8)
        self.assertEqual(to_numpy(t)[1, 1], 3*4+2*8)

    def test_random_operations(self):
        # TODO(chonho)
        pass

    def test_tensor_copy(self):
        t = Tensor((2, 3))
        t.set_value(1.23)
        self.assertAlmostEqual(to_numpy(t)[0, 0], 1.23)
        tc = t.copy()
        tdc = t.deepcopy()
        self.assertAlmostEqual(to_numpy(tc)[0, 0], 1.23)
        self.assertAlmostEqual(to_numpy(tdc)[0, 0], 1.23)
        t += 1.23
        self.assertAlmostEqual(to_numpy(t)[0, 0], 2.46)
        self.assertAlmostEqual(to_numpy(tc)[0, 0], 2.46)
        self.assertAlmostEqual(to_numpy(tdc)[0, 0], 1.23)

    def test_copy_data(self):
        t = self.t
        t.set_value(1.23)
        s = self.s
        s.set_value(5.43)
        self.assertAlmostEqual(to_numpy(t)[0, 0], 1.23)
        copy_data_to_from(t, s, 2)
        self.assertAlmostEqual(to_numpy(t)[0, 0], 5.43, 5)
        self.assertAlmostEqual(to_numpy(t)[0, 1], 5.43, 5)
        self.assertAlmostEqual(to_numpy(t)[1, 0], 1.23, 5)

    def test_global_method(self):
        t = self.t
        t.set_value(12.34)
        a = log(t)
        self.assertAlmostEqual(to_numpy(a)[0, 0], math.log(12.34), 5)

if __name__ == '__main__':
    unittest.main()
