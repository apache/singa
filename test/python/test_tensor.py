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
import random
import numpy as np

from singa import tensor
from singa import singa_wrap as singa_api
from singa import autograd

from cuda_helper import gpu_dev, cpu_dev


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
        self.assertEqual(tensor.product(shape), 2 * 3)
        self.assertEqual(t.ndim(), 2)
        self.assertEqual(t.size(), 2 * 3)
        self.assertEqual(t.memsize(), 2 * 3 * tensor.sizeof(tensor.float32))
        self.assertFalse(t.is_transpose())

    def test_unary_operators(self):
        t = self.t
        self.assertAlmostEqual(tensor.to_numpy(t)[0, 0], 0.0)
        t += 1.23
        self.assertAlmostEqual(tensor.to_numpy(t)[0, 0], 1.23)
        t -= 0.23
        self.assertAlmostEqual(tensor.to_numpy(t)[0, 0], 1.23 - 0.23)
        t *= 2.5
        self.assertAlmostEqual(tensor.to_numpy(t)[0, 0], (1.23 - 0.23) * 2.5)
        t /= 2
        self.assertAlmostEqual(
            tensor.to_numpy(t)[0, 0], (1.23 - 0.23) * 2.5 / 2)

    def test_binary_operators(self):
        t = self.t
        t += 3.2
        s = self.s
        s += 2.1
        a = t + s
        self.assertAlmostEqual(tensor.to_numpy(a)[0, 0], 3.2 + 2.1, 5)
        a = t - s
        self.assertAlmostEqual(tensor.to_numpy(a)[0, 0], 3.2 - 2.1, 5)
        a = t * s
        self.assertAlmostEqual(tensor.to_numpy(a)[0, 0], 3.2 * 2.1, 5)
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
        a = t == 3.45
        self.assertEqual(tensor.to_numpy(a)[0, 0], 1)
        a = tensor.lt(t, 3.45)
        self.assertEqual(tensor.to_numpy(a)[0, 0], 0)
        a = tensor.le(t, 3.45)
        self.assertEqual(tensor.to_numpy(a)[0, 0], 1)
        a = tensor.gt(t, 3.45)
        self.assertEqual(tensor.to_numpy(a)[0, 0], 0)
        a = tensor.ge(t, 3.45)
        self.assertEqual(tensor.to_numpy(a)[0, 0], 1)
        a = tensor.eq(t, 3.45)
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

    def matmul_high_dim_helper(self, dev):
        configs = [
            [(1, 12, 7, 64), (1, 12, 64, 7)],
            [(1, 7, 768), (768, 768)],
        ]
        print()
        for config in configs:
            X = np.random.random(config[0]).astype(np.float32)
            x = tensor.from_numpy(X)
            x.to_device(dev)

            W = np.random.random(config[1]).astype(np.float32)
            w = tensor.from_numpy(W)
            w.to_device(dev)

            y_t = np.matmul(X, W)
            y = autograd.matmul(x, w)
            np.testing.assert_array_almost_equal(tensor.to_numpy(y), y_t, 3)

    def test_matmul_high_dim_cpu(self):
        self.matmul_high_dim_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_matmul_high_dim_gpu(self):
        self.matmul_high_dim_helper(gpu_dev)

    def test_tensor_inplace_api(self):
        """ tensor inplace methods alter internal state and also return self
        """
        x = tensor.Tensor((3,))
        y = x.set_value(1)
        self.assertTrue(y is x)

        x = tensor.Tensor((3,))
        y = x.uniform(1, 2)
        self.assertTrue(y is x)

        x = tensor.Tensor((3,))
        y = x.bernoulli(1)
        self.assertTrue(y is x)

        x = tensor.Tensor((3,))
        y = x.gaussian(1, 2)
        self.assertTrue(y is x)

    def test_numpy_convert(self):
        a = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.int)
        t = tensor.from_numpy(a)
        b = tensor.to_numpy(t)
        self.assertEqual(np.sum(a - b), 0)

        a = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        t = tensor.from_numpy(a)
        b = tensor.to_numpy(t)
        self.assertEqual(np.sum(a - b), 0.)

    def test_transpose(self):
        a = np.array(
            [1.1, 1.1, 1.1, 1.1, 1.4, 1.3, 1.1, 1.6, 1.1, 1.1, 1.1, 1.2])
        a = np.reshape(a, (2, 3, 2))
        ta = tensor.from_numpy(a)

        A1 = np.transpose(a)
        tA1 = tensor.transpose(ta)
        TA1 = tensor.to_numpy(tA1)
        A2 = np.transpose(a, [0, 2, 1])
        tA2 = tensor.transpose(ta, [0, 2, 1])
        TA2 = tensor.to_numpy(tA2)

        np.testing.assert_array_almost_equal(TA1, A1)
        np.testing.assert_array_almost_equal(TA2, A2)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_gpu_6d_transpose(self, dev=gpu_dev):
        s0 = (2, 3, 4, 5, 6, 7)
        axes1 = [5, 4, 3, 2, 1, 0]
        s1 = (2, 7, 6, 5, 4, 3)
        s2 = (2, 4, 3, 5, 7, 6)
        a = np.random.random(s1)

        ta = tensor.from_numpy(a)
        ta.to_device(dev)

        ta = tensor.reshape(ta, s1)
        ta = tensor.transpose(ta, axes1)
        ta = tensor.reshape(ta, s2)

        a = np.reshape(a, s1)
        a = np.transpose(a, axes1)
        a = np.reshape(a, s2)

        np.testing.assert_array_almost_equal(tensor.to_numpy(ta), a)

    def test_einsum(self):

        a = np.array(
            [1.1, 1.1, 1.1, 1.1, 1.4, 1.3, 1.1, 1.6, 1.1, 1.1, 1.1, 1.2])
        a = np.reshape(a, (2, 3, 2))
        ta = tensor.from_numpy(a)

        res1 = np.einsum('kij,kij->kij', a, a)
        tres1 = tensor.einsum('kij,kij->kij', ta, ta)
        Tres1 = tensor.to_numpy(tres1)
        res2 = np.einsum('kij,kih->kjh', a, a)
        tres2 = tensor.einsum('kij,kih->kjh', ta, ta)
        Tres2 = tensor.to_numpy(tres2)

        self.assertAlmostEqual(np.sum(Tres1 - res1), 0., places=3)
        self.assertAlmostEqual(np.sum(Tres2 - res2), 0., places=3)

    def test_repeat(self):

        a = np.array(
            [1.1, 1.1, 1.1, 1.1, 1.4, 1.3, 1.1, 1.6, 1.1, 1.1, 1.1, 1.2])
        a = np.reshape(a, (2, 3, 2))
        ta = tensor.from_numpy(a)

        ta_repeat1 = tensor.repeat(ta, 2, axis=None)
        a_repeat1 = np.repeat(a, 2, axis=None)
        Ta_repeat1 = tensor.to_numpy(ta_repeat1)
        ta_repeat2 = tensor.repeat(ta, 4, axis=1)
        a_repeat2 = np.repeat(a, 4, axis=1)
        Ta_repeat2 = tensor.to_numpy(ta_repeat2)

        self.assertAlmostEqual(np.sum(Ta_repeat1 - a_repeat1), 0., places=3)
        self.assertAlmostEqual(np.sum(Ta_repeat2 - a_repeat2), 0., places=3)

    def test_sum(self):
        a = np.array(
            [1.1, 1.1, 1.1, 1.1, 1.4, 1.3, 1.1, 1.6, 1.1, 1.1, 1.1, 1.2])
        a = np.reshape(a, (2, 3, 2))
        ta = tensor.from_numpy(a)

        a_sum0 = np.sum(a)
        ta_sum0 = tensor.sum(ta)
        Ta_sum0 = tensor.to_numpy(ta_sum0)
        a_sum1 = np.sum(a, axis=1)
        ta_sum1 = tensor.sum(ta, axis=1)
        Ta_sum1 = tensor.to_numpy(ta_sum1)
        a_sum2 = np.sum(a, axis=2)
        ta_sum2 = tensor.sum(ta, axis=2)
        Ta_sum2 = tensor.to_numpy(ta_sum2)

        self.assertAlmostEqual(np.sum(a_sum0 - Ta_sum0), 0., places=3)
        self.assertAlmostEqual(np.sum(a_sum1 - Ta_sum1), 0., places=3)
        self.assertAlmostEqual(np.sum(a_sum2 - Ta_sum2), 0., places=3)

    def test_tensordot(self):
        a = np.array(
            [1.1, 1.1, 1.1, 1.1, 1.4, 1.3, 1.1, 1.6, 1.1, 1.1, 1.1, 1.2])
        a = np.reshape(a, (2, 3, 2))

        ta = tensor.from_numpy(a)

        res1 = np.tensordot(a, a, axes=1)
        tres1 = tensor.tensordot(ta, ta, axes=1)
        Tres1 = tensor.to_numpy(tres1)
        self.assertAlmostEqual(np.sum(Tres1 - res1), 0., places=3)
        np.testing.assert_array_almost_equal(Tres1, res1)

        res2 = np.tensordot(a, a, axes=([0, 1], [2, 1]))
        tres2 = tensor.tensordot(ta, ta, axes=([0, 1], [2, 1]))
        np.testing.assert_array_almost_equal(tensor.to_numpy(tres2), res2)

    def test_reshape(self):
        a = np.array([[[1.1, 1.1, 1.4], [1.1, 1.1, 1.1]],
                      [[1.1, 1.1, 1.3], [1.6, 1.1, 1.2]]])
        ta = tensor.from_numpy(a)
        tb = tensor.reshape(ta, [2, 6])
        self.assertAlmostEqual(tb.shape[0], 2., places=3)
        self.assertAlmostEqual(tb.shape[1], 6., places=3)
        np.testing.assert_array_almost_equal(tensor.to_numpy(tb),
                                             a.reshape((2, 6)))

    def test_transpose_then_reshape(self):
        a = np.array([[[1.1, 1.1], [1.1, 1.1], [1.4, 1.3]],
                      [[1.1, 1.6], [1.1, 1.1], [1.1, 1.2]]])
        TRANSPOSE_AXES = (2, 0, 1)
        RESHAPE_DIMS = (2, 6)

        ta = tensor.from_numpy(a)
        ta = ta.transpose(TRANSPOSE_AXES)
        ta = ta.reshape(RESHAPE_DIMS)

        np.testing.assert_array_almost_equal(
            tensor.to_numpy(ta),
            np.reshape(a.transpose(TRANSPOSE_AXES), RESHAPE_DIMS))

    def _concatenate_helper(self, dev):
        np1 = np.random.random([5, 6, 7, 8]).astype(np.float32)
        np2 = np.random.random([5, 6, 7, 1]).astype(np.float32)
        np3 = np.concatenate((np1, np2), axis=3)

        t1 = tensor.Tensor(device=dev, data=np1)
        t2 = tensor.Tensor(device=dev, data=np2)

        t3 = tensor.concatenate((t1, t2), 3)

        np.testing.assert_array_almost_equal(tensor.to_numpy(t3), np3)

    def test_concatenate_cpu(self):
        self._concatenate_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_concatenate_gpu(self):
        self._concatenate_helper(gpu_dev)

    def _subscription_helper(self, dev):
        np1 = np.random.random((5, 5, 5, 5)).astype(np.float32)
        sg_tensor = tensor.Tensor(device=dev, data=np1)
        sg_tensor_ret = sg_tensor[1:3, :, 1:, :-1]
        np.testing.assert_array_almost_equal((tensor.to_numpy(sg_tensor_ret)),
                                             np1[1:3, :, 1:, :-1])

    def test_subscription_cpu(self):
        self._subscription_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_subscription_gpu(self):
        self._subscription_helper(gpu_dev)

    def _ceil_helper(self, dev):

        np1 = np.random.random([5, 6, 7, 8]).astype(np.float32)
        np1 = np1 * 10
        np2 = np.ceil(np1)

        t1 = tensor.Tensor(device=dev, data=np1)

        t2 = tensor.ceil(t1)

        np.testing.assert_array_almost_equal(tensor.to_numpy(t2), np2)

    def test_ceil_cpu(self):
        self._ceil_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_ceil_gpu(self):
        self._ceil_helper(gpu_dev)

    def _astype_helper(self, dev):
        shape1 = [2, 3]
        shape2 = [3, 2]

        np_flt = np.random.random(shape1).astype(np.float32)
        np_flt = np_flt * 10 - 5

        np_int = np_flt.astype(np.int32)
        np_flt2 = np_int.astype(np.float32)

        t2 = tensor.Tensor(device=dev, data=np_flt)
        t2 = t2.as_type('int')
        np.testing.assert_array_almost_equal(tensor.to_numpy(t2), np_int)

        t1 = t2.reshape(shape2)
        np.testing.assert_array_almost_equal(tensor.to_numpy(t1),
                                             np_int.reshape(shape2))

        t1 = t1.as_type('float')
        np.testing.assert_array_almost_equal(tensor.to_numpy(t1),
                                             np_flt2.reshape(shape2))

    def test_astype_cpu(self):
        self._astype_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_astype_gpu(self):
        self._astype_helper(gpu_dev)

    def _3d_matmul_helper(self, dev):
        np_x1 = np.random.randn(2, 3, 4).astype(np.float32)
        np_x2 = np.random.randn(2, 4, 3).astype(np.float32)
        x1 = tensor.from_numpy(np_x1)
        x1.to_device(dev)
        x2 = tensor.from_numpy(np_x2)
        x2.to_device(dev)
        y = autograd.matmul(x1, x2)
        np_y = np.matmul(np_x1, np_x2)
        np.testing.assert_array_almost_equal(tensor.to_numpy(y), np_y)

        np_x1 = np.random.randn(2, 3, 4).astype(np.float32)
        np_x2 = np.random.randn(2, 4, 5).astype(np.float32)
        x1 = tensor.from_numpy(np_x1)
        x1.to_device(dev)
        x2 = tensor.from_numpy(np_x2)
        x2.to_device(dev)
        y = autograd.matmul(x1, x2)
        np_y = np.matmul(np_x1, np_x2)
        np.testing.assert_array_almost_equal(tensor.to_numpy(y), np_y)

    def test_3d_matmul_cpu(self):
        self._3d_matmul_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_3d_matmul_gpu(self):
        self._3d_matmul_helper(gpu_dev)

    def _4d_matmul_helper(self, dev):
        np_x1 = np.random.randn(2, 12, 256, 64).astype(np.float32)
        np_x2 = np.random.randn(2, 12, 64, 256).astype(np.float32)
        x1 = tensor.from_numpy(np_x1)
        x1.to_device(dev)
        x2 = tensor.from_numpy(np_x2)
        x2.to_device(dev)
        y = autograd.matmul(x1, x2)
        np_y = np.matmul(np_x1, np_x2)
        np.testing.assert_array_almost_equal(tensor.to_numpy(y), np_y)

        np_x1 = np.random.randn(2, 12, 256, 64).astype(np.float32)
        np_x2 = np.random.randn(2, 12, 64, 1024).astype(np.float32)
        x1 = tensor.from_numpy(np_x1)
        x1.to_device(dev)
        x2 = tensor.from_numpy(np_x2)
        x2.to_device(dev)
        y = autograd.matmul(x1, x2)
        np_y = np.matmul(np_x1, np_x2)
        np.testing.assert_array_almost_equal(tensor.to_numpy(y), np_y)

    def test_4d_matmul_cpu(self):
        self._4d_matmul_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_4d_matmul_gpu(self):
        self._4d_matmul_helper(gpu_dev)

    def _matmul_transpose_helper(self, dev):

        X = np.random.random((1, 256, 12, 64)).astype(np.float32)
        x = tensor.from_numpy(X)
        x.to_device(dev)

        W = np.random.random((1, 256, 12, 64)).astype(np.float32)
        w = tensor.from_numpy(W)
        w.to_device(dev)

        X = np.transpose(X, (0, 2, 1, 3))
        W = np.transpose(W, (0, 2, 1, 3))
        W = np.transpose(W, (0, 1, 3, 2))
        Y = np.matmul(X, W)

        x = autograd.transpose(x, (0, 2, 1, 3))
        w = autograd.transpose(w, (0, 2, 1, 3))
        w = autograd.transpose(w, (0, 1, 3, 2))
        y = autograd.matmul(x, w)

        np.testing.assert_array_almost_equal(tensor.to_numpy(x), X)
        np.testing.assert_array_almost_equal(tensor.to_numpy(w), W)
        np.testing.assert_array_almost_equal(tensor.to_numpy(y), Y)

    def test_matmul_transpose_cpu(self):
        self._matmul_transpose_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_matmul_transpose_gpu(self):
        self._matmul_transpose_helper(gpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_gaussian_gpu(self, dev=gpu_dev):
        x = tensor.Tensor((3, 5, 3, 5), device=dev)
        x.gaussian(0, 1)
        x = tensor.Tensor((4, 5, 3, 2), device=dev)
        x.gaussian(0, 1)

    def _kfloat32_int(self, dev=gpu_dev):
        np.random.seed(0)
        x_val = np.random.random((2, 3)).astype(np.float32) * 10
        x = tensor.from_numpy(x_val)
        x.to_device(dev)
        scalar = np.random.random((1,))[0] * 100
        y = x + scalar
        self.assertEqual(y.dtype, tensor.float32)
        np.testing.assert_array_almost_equal(tensor.to_numpy(y), x_val + scalar)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_kfloat32_int_gpu(self):
        self._kfloat32_int(gpu_dev)

    def test_kfloat32_int_cpu(self):
        self._kfloat32_int(cpu_dev)

    def _kint_float(self, dev=gpu_dev):
        np.random.seed(0)
        x_val = np.random.randint(0, 10, (2, 3))
        x = tensor.from_numpy(x_val)
        x.to_device(dev)
        scalar = random.random() * 100
        y = x + scalar
        self.assertEqual(y.dtype, tensor.float32)
        np.testing.assert_array_almost_equal(tensor.to_numpy(y), x_val + scalar,
                                             5)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_kint_float_gpu(self):
        self._kint_float(gpu_dev)

    def test_kint_float_cpu(self):
        self._kint_float(cpu_dev)

    def _kint_kint(self, dev=gpu_dev):
        a_np = np.array([[[17, 4, 9, 22, 18], [-9, 9, -1, -1, 4],
                          [1, 14, 7, 1, 4], [3, 14, -2, 3, -8]],
                         [[-25, 6, 8, -7, 22], [-14, 0, -1, 15, 14],
                          [1, 3, -8, -19, -3], [1, 12, 12, -3, -3]],
                         [[-10, -14, -17, 19, -5], [-4, -12, 7, -16, -2],
                          [-8, 3, -5, -11, 0], [4, 0, 3, -6, -3]]],
                        dtype=np.int32)
        b_np = np.array([[[-6, -3, -8, -17, 1], [-4, -16, 4, -9, 0],
                          [7, 1, 11, -12, 4], [-6, -8, -5, -3, 0]],
                         [[-11, 9, 4, -15, 14], [18, 11, -1, -10, 10],
                          [-4, 12, 2, 9, 3], [7, 0, 17, 1, 4]],
                         [[18, -13, -12, 9, -11], [19, -4, -7, 19, 14],
                          [18, 9, -8, 19, -2], [8, 9, -1, 6, 9]]],
                        dtype=np.int32)
        ta = tensor.from_numpy(a_np)
        tb = tensor.from_numpy(b_np)
        ta.to_device(dev)
        tb.to_device(dev)
        y = ta - tb
        np.testing.assert_array_almost_equal(tensor.to_numpy(y), a_np - b_np)

    def test_kint_kint_cpu(self, dev=cpu_dev):
        self._kint_kint(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_kint_kint_gpu(self, dev=gpu_dev):
        self._kint_kint(gpu_dev)

    def _kint_kint_bc(self, dev=gpu_dev):
        a_np = np.array([[[17, 4, 9, 22, 18], [-9, 9, -1, -1, 4],
                          [1, 14, 7, 1, 4], [3, 14, -2, 3, -8]],
                         [[-25, 6, 8, -7, 22], [-14, 0, -1, 15, 14],
                          [1, 3, -8, -19, -3], [1, 12, 12, -3, -3]],
                         [[-10, -14, -17, 19, -5], [-4, -12, 7, -16, -2],
                          [-8, 3, -5, -11, 0], [4, 0, 3, -6, -3]]],
                        dtype=np.int32)
        b_np = np.array([[-6, -3, -8, -17, 1], [-4, -16, 4, -9, 0],
                         [7, 1, 11, -12, 4], [-6, -8, -5, -3, 0]],
                        dtype=np.int32)
        ta = tensor.from_numpy(a_np)
        tb = tensor.from_numpy(b_np)
        ta.to_device(dev)
        tb.to_device(dev)
        y = ta - tb
        np.testing.assert_array_almost_equal(tensor.to_numpy(y), a_np - b_np)

    def test_kint_kint_bc_cpu(self, dev=cpu_dev):
        self._kint_kint_bc(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_kint_kint_bc_gpu(self, dev=gpu_dev):
        self._kint_kint_bc(gpu_dev)


if __name__ == '__main__':
    unittest.main()
