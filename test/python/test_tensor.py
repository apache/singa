#!/usr/bin/env python

#/************************************************************
#*
#* Licensed to the Apache Software Foundation (ASF) under one
#* or more contributor license agreements.  See the NOTICE file
#* distributed with this work for additional information
#* regarding copyright ownership.  The ASF licenses this file
#* to you under the Apache License, Version 2.0 (the
#* "License"); you may not use this file except in compliance
#* with the License.  You may obtain a copy of the License at
#*
#*   http://www.apache.org/licenses/LICENSE-2.0
#*
#* Unless required by applicable law or agreed to in writing,
#* software distributed under the License is distributed on an
#* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#* KIND, either express or implied.  See the License for the
#* specific language governing permissions and limitations
#* under the License.
#*
#*************************************************************/

import sys
import os
import math
import unittest
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__),
                             '../../src/python'))
from tensor import *
from device import *

sys.path.append(os.path.join(os.path.dirname(__file__),
                             '../../build/src'))
from core_pb2 import *

class TestTensorMethods(unittest.TestCase):

    def setUp(self):
        self.shape = (2, 3)
        self.t = Tensor(self.shape)
        self.s = Tensor(self.shape)

    def test_tensor_fields(self):
        t = self.t
        shape = self.shape
        self.assertTupleEqual(t.shape(), shape)
        self.assertEqual(t.shape(0), shape[0])
        self.assertEqual(t.shape(1), shape[1])
        self.assertEqual(product(shape), 2*3)
        self.assertEqual(t.ndim(), 2)
        self.assertEqual(t.size(), 2*3)
        self.assertEqual(t.memsize(), 2*3*sizeof(kFloat32))
        self.assertFalse(t.is_transpose())

    def test_unary_operators(self):
        t = self.t
        self.assertAlmostEqual(t.toarray()[0,0], 0.0)
        t += 1.23
        self.assertAlmostEqual(t.toarray()[0,0], 1.23)
        t -= 0.23
        self.assertAlmostEqual(t.toarray()[0,0], 1.23-0.23)
        t *= 2.5
        self.assertAlmostEqual(t.toarray()[0,0], (1.23-0.23)*2.5)
        t /= 2
        self.assertAlmostEqual(t.toarray()[0,0], (1.23-0.23)*2.5/2)

    def test_binary_operators(self):
        t = self.t
        t += 3.2
        s = self.s
        s += 2.1
        a = t + s
        self.assertAlmostEqual(a.toarray()[0,0], 3.2+2.1, 5)
        a = t - s
        self.assertAlmostEqual(a.toarray()[0,0], 3.2-2.1, 5)
        a = t * s
        self.assertAlmostEqual(a.toarray()[0,0], 3.2*2.1, 5)
        ''' not implemented yet
        a = t / s
        self.assertAlmostEqual(a.toarray()[0,0], 3.2/2.1, 5)
        '''

    def test_comparison_operators(self):
        t = self.t
        t += 3.45
        a = t < 3.45
        self.assertEqual(a.toarray()[0,0], 0)
        a = t <= 3.45
        self.assertEqual(a.toarray()[0,0], 1)
        a = t > 3.45
        self.assertEqual(a.toarray()[0,0], 0)
        a = t >= 3.45
        self.assertEqual(a.toarray()[0,0], 1)
        a = lt(t, 3.45)
        self.assertEqual(a.toarray()[0,0], 0)
        a = le(t, 3.45)
        self.assertEqual(a.toarray()[0,0], 1)
        a = gt(t, 3.45)
        self.assertEqual(a.toarray()[0,0], 0)
        a = ge(t, 3.45)
        self.assertEqual(a.toarray()[0,0], 1)


    def test_tensor_copy(self):
        t = Tensor((2,3))
        t += 1.23
        self.assertAlmostEqual(t.toarray()[0,0], 1.23)
        tc = t.copy()
        tdc = t.deepcopy()
        self.assertAlmostEqual(tc.toarray()[0,0], 1.23)
        self.assertAlmostEqual(tdc.toarray()[0,0], 1.23)
        t += 1.23
        self.assertAlmostEqual(t.toarray()[0,0], 2.46)
        self.assertAlmostEqual(tc.toarray()[0,0], 2.46)
        self.assertAlmostEqual(tdc.toarray()[0,0], 1.23)

    def test_copy_data(self):
        t = self.t
        t += 1.23
        s = self.s
        s += 5.43
        self.assertAlmostEqual(t.toarray()[0,0], 1.23)
        copy_data_to_from(t, s, 2)
        self.assertAlmostEqual(t.toarray()[0,0], 5.43, 5)
        self.assertAlmostEqual(t.toarray()[0,1], 5.43, 5)
        self.assertAlmostEqual(t.toarray()[0,2], 1.23)


    def test_global_method(self):
        t = self.t
        t += 12.34
        a = log(t)
        self.assertAlmostEqual(a.toarray()[0,0], math.log(12.34))

if __name__ == '__main__':
    unittest.main()
