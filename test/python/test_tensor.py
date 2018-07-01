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
from __future__ import division

import math
import unittest
import numpy as np


from singa import tensor
from singa.proto import core_pb2


class TestTensorMethods(unittest.TestCase):

    def setUp(self):
        self.shape = (2, 3)
        self.t = tensor.Tensor(self.shape)
        self.s = tensor.Tensor(self.shape)
        self.t.set_value(0)
        self.s.set_value(0)

    def test_tensor_fields(self):
        t = self.t
        shape = self.shape
        self.assertTupleEqual(t.shape, shape)
        self.assertEqual(t.shape[0], shape[0])
        self.assertEqual(t.shape[1], shape[1])
        self.assertEqual(tensor.product(shape), 2*3)
        self.assertEqual(t.ndim(), 2)
        self.assertEqual(t.size(), 2*3)
        self.assertEqual(t.memsize(), 2*3*tensor.sizeof(core_pb2.kFloat32))
        self.assertFalse(t.is_transpose())

    def test_unary_operators(self):
        t = self.t
        self.assertAlmostEqual(tensor.to_numpy(t)[0, 0], 0.0)
        t += 1.23
        self.assertAlmostEqual(tensor.to_numpy(t)[0, 0], 1.23)
        t -= 0.23
        self.assertAlmostEqual(tensor.to_numpy(t)[0, 0], 1.23-0.23)
        t *= 2.5
        self.assertAlmostEqual(tensor.to_numpy(t)[0, 0], (1.23-0.23)*2.5)
        t /= 2
        self.assertAlmostEqual(tensor.to_numpy(t)[0, 0], (1.23-0.23)*2.5/2)

    def test_binary_operators(self):
        t = self.t
        t += 3.2
        s = self.s
        s += 2.1
        a = t + s
        self.assertAlmostEqual(tensor.to_numpy(a)[0, 0], 3.2+2.1, 5)
        a = t - s
        self.assertAlmostEqual(tensor.to_numpy(a)[0, 0], 3.2-2.1, 5)
        a = t * s
        self.assertAlmostEqual(tensor.to_numpy(a)[0, 0], 3.2*2.1, 5)
        ''' not implemented yet
        a = t / s
        self.assertAlmostEqual(tensor.to_numpy(a)[0,0], 3.2/2.1, 5)
        '''

    def test_comparison_operators(self):
        t = self.t
        t += 3.45
        a = t < 3.45
        self.assertEqual(tensor.to_numpy(a)[0, 0], 0)
        a = t <= 3.45
        self.assertEqual(tensor.to_numpy(a)[0, 0], 1)
        a = t > 3.45
        self.assertEqual(tensor.to_numpy(a)[0, 0], 0)
        a = t >= 3.45
        self.assertEqual(tensor.to_numpy(a)[0, 0], 1)
        a = tensor.lt(t, 3.45)
        self.assertEqual(tensor.to_numpy(a)[0, 0], 0)
        a = tensor.le(t, 3.45)
        self.assertEqual(tensor.to_numpy(a)[0, 0], 1)
        a = tensor.gt(t, 3.45)
        self.assertEqual(tensor.to_numpy(a)[0, 0], 0)
        a = tensor.ge(t, 3.45)
        self.assertEqual(tensor.to_numpy(a)[0, 0], 1)

    def test_tensor_copy(self):
        t = tensor.Tensor((2, 3))
        t += 1.23
        self.assertAlmostEqual(tensor.to_numpy(t)[0, 0], 1.23)
        tc = t.copy()
        tdc = t.deepcopy()
        self.assertAlmostEqual(tensor.to_numpy(tc)[0, 0], 1.23)
        self.assertAlmostEqual(tensor.to_numpy(tdc)[0, 0], 1.23)
        t += 1.23
        self.assertAlmostEqual(tensor.to_numpy(t)[0, 0], 2.46)
        self.assertAlmostEqual(tensor.to_numpy(tc)[0, 0], 2.46)
        self.assertAlmostEqual(tensor.to_numpy(tdc)[0, 0], 1.23)

    def test_copy_data(self):
        t = self.t
        t += 1.23
        s = self.s
        s += 5.43
        self.assertAlmostEqual(tensor.to_numpy(t)[0, 0], 1.23)
        tensor.copy_data_to_from(t, s, 2)
        self.assertAlmostEqual(tensor.to_numpy(t)[0, 0], 5.43, 5)
        self.assertAlmostEqual(tensor.to_numpy(t)[0, 1], 5.43, 5)
        self.assertAlmostEqual(tensor.to_numpy(t)[0, 2], 1.23)

    def test_global_method(self):
        t = self.t
        t += 12.34
        a = tensor.log(t)
        self.assertAlmostEqual(tensor.to_numpy(a)[0, 0], math.log(12.34))

    def test_random(self):
        x = tensor.Tensor((1000,))
        x.gaussian(1, 0.01)
        self.assertAlmostEqual(tensor.average(x), 1, 3)

    def test_radd(self):
        x = tensor.Tensor((3,))
        x.set_value(1)
        y = 1 + x
        self.assertEqual(tensor.average(y), 2.)

    def test_rsub(self):
        x = tensor.Tensor((3,))
        x.set_value(1)
        y = 1 - x
        self.assertEqual(tensor.average(y), 0.)

    def test_rmul(self):
        x = tensor.Tensor((3,))
        x.set_value(1)
        y = 2 * x
        self.assertEqual(tensor.average(y), 2.)

    def test_rdiv(self):
        x = tensor.Tensor((3,))
        x.set_value(1)
        y = 2 / x
        self.assertEqual(tensor.average(y), 2.)

    def test_numpy_convert(self):
        a = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.int)
        t = tensor.from_numpy(a)
        b = tensor.to_numpy(t)
        self.assertEqual(np.sum(a-b), 0)

        a = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        t = tensor.from_numpy(a)
        b = tensor.to_numpy(t)
        self.assertEqual(np.sum(a-b), 0.)

    def test_transpose(self):
        a = np.array([1.1,1.1,1.1,1.1,1.4,1.3,1.1,1.6,1.1,1.1,1.1,1.2])
        a = np.reshape(a,(2,3,2))
        ta = tensor.from_numpy(a)

        A1 = np.transpose(a)
        tA1 = tensor.transpose(ta)
        TA1 = tensor.to_numpy(tA1)
        A2 = np.transpose(a,[0,2,1])
        tA2 = tensor.transpose(ta,[0,2,1])
        TA2 = tensor.to_numpy(tA2)

        self.assertAlmostEqual(np.sum(TA1 - A1), 0.,places=3)
        self.assertAlmostEqual(np.sum(TA2 - A2), 0.,places=3)

    def test_einsum(self):

        a = np.array([1.1,1.1,1.1,1.1,1.4,1.3,1.1,1.6,1.1,1.1,1.1,1.2])
        a = np.reshape(a,(2,3,2))
        ta = tensor.from_numpy(a)

        res1 = np.einsum('kij,kij->kij', a, a)
        tres1 = tensor.einsum('kij,kij->kij', ta, ta)
        Tres1 = tensor.to_numpy(tres1)
        res2 = np.einsum('kij,kih->kjh', a, a)
        tres2 = tensor.einsum('kij,kih->kjh', ta, ta)
        Tres2 = tensor.to_numpy(tres2)
        
        self.assertAlmostEqual(np.sum(Tres1 - res1), 0.,places=3)
        self.assertAlmostEqual(np.sum(Tres2 - res2), 0.,places=3)

    def test_repeat(self):

        a = np.array([1.1,1.1,1.1,1.1,1.4,1.3,1.1,1.6,1.1,1.1,1.1,1.2])
        a = np.reshape(a,(2,3,2))
        ta = tensor.from_numpy(a)

        ta_repeat1 = tensor.repeat(ta,2,axis = None)
        a_repeat1 = np.repeat(a,2,axis = None)
        Ta_repeat1 = tensor.to_numpy(ta_repeat1)
        ta_repeat2 = tensor.repeat(ta, 4, axis = 1)
        a_repeat2 = np.repeat(a, 4, axis = 1)
        Ta_repeat2 = tensor.to_numpy(ta_repeat2)

        self.assertAlmostEqual(np.sum(Ta_repeat1 - a_repeat1), 0., places=3)
        self.assertAlmostEqual(np.sum(Ta_repeat2 - a_repeat2), 0., places=3)

    def test_sum(self):
        a = np.array([1.1,1.1,1.1,1.1,1.4,1.3,1.1,1.6,1.1,1.1,1.1,1.2])
        a = np.reshape(a,(2,3,2))
        ta = tensor.from_numpy(a)

        a_sum0 = np.sum(a)
        ta_sum0 = tensor.sum(ta)
        Ta_sum0 = tensor.to_numpy(ta_sum0)
        a_sum1 = np.sum(a, axis = 1)
        ta_sum1 = tensor.sum(ta, axis = 1)
        Ta_sum1 = tensor.to_numpy(ta_sum1)
        a_sum2 = np.sum(a, axis = 2)
        ta_sum2 = tensor.sum(ta, axis = 2)
        Ta_sum2 = tensor.to_numpy(ta_sum2)

        self.assertAlmostEqual(np.sum(a_sum0 - Ta_sum0), 0., places=3)
        self.assertAlmostEqual(np.sum(a_sum1 - Ta_sum1), 0., places=3)
        self.assertAlmostEqual(np.sum(a_sum2 - Ta_sum2), 0., places=3)

    def test_tensordot(self):
        a = np.array([1.1,1.1,1.1,1.1,1.4,1.3,1.1,1.6,1.1,1.1,1.1,1.2])
        a = np.reshape(a,(2,3,2))

        ta = tensor.from_numpy(a)

        res1 = np.tensordot(a, a, axes = 1)
        tres1 = tensor.tensordot(ta, ta, axes = 1)
        Tres1 = tensor.to_numpy(tres1)
        res2 = np.tensordot(a, a, axes = ([0,1],[2,1]))
        tres2 = tensor.tensordot(ta, ta, axes = ([0,1],[2,1]))
        Tres2 = tensor.to_numpy(tres2)

        self.assertAlmostEqual(np.sum(Tres1 - res1), 0., places=3)
        self.assertAlmostEqual(np.sum(Tres2 - res2), 0., places=3)


if __name__ == '__main__':
    unittest.main()
