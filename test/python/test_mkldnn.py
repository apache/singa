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
from singa import singa_wrap


class TestPythonOperation(unittest.TestCase):

    def test_conv2d(self):
        print("TEST CONV2D FORWARD")
        x_shape = [2, 1, 3, 3]
        x = singa_wrap.Tensor(x_shape)
        x.CopyFloatDataFromHostPtr(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        W_shape = [1, 1, 3, 3]
        W = singa_wrap.Tensor(W_shape)
        W.CopyFloatDataFromHostPtr([1, 1, 0, 0, 0, -1, 0, 1, 0])

        b_shape = [1]
        b = singa_wrap.Tensor(b_shape)
        b.CopyFloatDataFromHostPtr([1])

        dy_shape = [2, 1, 2, 2]
        dy = singa_wrap.Tensor(dy_shape)
        dy.CopyFloatDataFromHostPtr([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4])

        handle = singa_wrap.ConvHandle(x, (3, 3), (2, 2), (1, 1), 1, 1, True, 1)
        y = singa_wrap.CpuConvForward(x, W, b, handle)

        self.assertListEqual([2, 1, 2, 2], list(y.shape()))

        _y = y.GetFloatValue(int(y.Size()))
        self.assertAlmostEqual(3.0, _y[0])
        self.assertAlmostEqual(7.0, _y[1])
        self.assertAlmostEqual(-3.0, _y[2])
        self.assertAlmostEqual(12.0, _y[3])
        self.assertAlmostEqual(3.0, _y[4])
        self.assertAlmostEqual(7.0, _y[5])
        self.assertAlmostEqual(-3.0, _y[6])
        self.assertAlmostEqual(12.0, _y[7])

        print("TEST CONV2D DATA BACKWARD")

        dx = singa_wrap.CpuConvBackwardx(dy, W, x, handle)
        self.assertListEqual([2, 1, 3, 3], list(dx.shape()))

        _dx = dx.GetFloatValue(int(dx.Size()))
        self.assertAlmostEqual(0.0, _dx[0])
        self.assertAlmostEqual(-0.1, _dx[1])
        self.assertAlmostEqual(0.0, _dx[2])
        self.assertAlmostEqual(0.4, _dx[3])
        self.assertAlmostEqual(0.4, _dx[4])
        self.assertAlmostEqual(0.6, _dx[5])
        self.assertAlmostEqual(0.0, _dx[6])
        self.assertAlmostEqual(-0.3, _dx[7])

        print("TEST CONV2D WEIGHT BACKWARD")
        dW = singa_wrap.CpuConvBackwardW(dy, x, W, handle)
        self.assertListEqual([1, 1, 3, 3], list(dW.shape()))

        _dW = dW.GetFloatValue(int(dW.Size()))
        self.assertAlmostEqual(4.0, _dW[0], places=5)
        self.assertAlmostEqual(7.2, _dW[1], places=5)
        self.assertAlmostEqual(3.0, _dW[2], places=5)
        self.assertAlmostEqual(7.2, _dW[3], places=5)
        self.assertAlmostEqual(12.8, _dW[4], places=5)
        self.assertAlmostEqual(5.2, _dW[5], places=5)
        self.assertAlmostEqual(2.0, _dW[6], places=5)
        self.assertAlmostEqual(3.2, _dW[7], places=5)
        self.assertAlmostEqual(1.0, _dW[8], places=5)

        print("TEST CONV2D DATA BACKWARD")
        db = singa_wrap.CpuConvBackwardb(dy, b, handle)
        self.assertEqual(1, dW.shape()[0])

        _db = db.GetFloatValue(int(db.Size()))
        print(_db)
        self.assertAlmostEqual(2.0, _db[0], places=5)

    def test_pooling(self):
        x_shape = [2, 1, 3, 3]
        x = singa_wrap.Tensor(x_shape)
        x.CopyFloatDataFromHostPtr(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        y_shape = [2, 1, 2, 2]
        dy = singa_wrap.Tensor(y_shape)
        dy.CopyFloatDataFromHostPtr([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4])

        k_dim = [2, 2]
        s_dim = [1, 1]
        p_dim = [0, 0]

        # max pooling
        handle = singa_wrap.PoolingHandle(x, k_dim, s_dim, p_dim, True)
        y = singa_wrap.CpuPoolingForward(handle, x)
        self.assertListEqual([2, 1, 2, 2], list(y.shape()))
        dx = singa_wrap.CpuPoolingBackward(handle, dy, x, y)
        self.assertListEqual([2, 1, 3, 3], list(dx.shape()))

        # avg pooling
        handle = singa_wrap.PoolingHandle(x, k_dim, s_dim, p_dim, False)
        y = singa_wrap.CpuPoolingForward(handle, x)
        self.assertListEqual([2, 1, 2, 2], list(y.shape()))
        dx = singa_wrap.CpuPoolingBackward(handle, dy, x, y)
        self.assertListEqual([2, 1, 3, 3], list(dx.shape()))

    def test_batch_norm(self):
        x_shape = [2, 2]
        x = singa_wrap.Tensor(x_shape)
        x.CopyFloatDataFromHostPtr([1, 2, 3, 4])

        dy_shape = [2, 2]
        dy = singa_wrap.Tensor(dy_shape)
        dy.CopyFloatDataFromHostPtr([4, 3, 2, 1])

        scale_shape = [2]
        scale = singa_wrap.Tensor(scale_shape)
        scale.CopyFloatDataFromHostPtr([1, 1])

        bias_shape = [2]
        bias = singa_wrap.Tensor(bias_shape)
        bias.CopyFloatDataFromHostPtr([0, 0])

        mean_shape = [2]
        mean = singa_wrap.Tensor(mean_shape)
        mean.CopyFloatDataFromHostPtr([1, 2])
        var = singa_wrap.Tensor(mean_shape)
        var.CopyFloatDataFromHostPtr([1, 2])

        handle = singa_wrap.BatchNormHandle(0.9, x)

        # 2D Forward Inference
        y = singa_wrap.CpuBatchNormForwardInference(handle, x, scale, bias,
                                                    mean, var)
        self.assertListEqual([2, 2], list(y.shape()))

        # 2D Forward Training
        (y, mean_updated, var_updated) = singa_wrap.CpuBatchNormForwardTraining(
            handle, x, scale, bias, mean, var)
        self.assertListEqual([2, 2], list(y.shape()))
        self.assertListEqual([2], list(mean_updated.shape()))
        self.assertListEqual([2], list(var_updated.shape()))

        # 2D Backward dx
        (dx, dscale,
         dbias) = singa_wrap.CpuBatchNormBackwardx(handle, y, dy, x, scale,
                                                   bias, mean_updated,
                                                   var_updated)
        self.assertListEqual([2, 2], list(dx.shape()))
        self.assertListEqual([2], list(dscale.shape()))
        self.assertListEqual([2], list(dbias.shape()))

        # 4D Forward Inference

        x2_shape = [1, 2, 4, 4]
        x2 = singa_wrap.Tensor(x2_shape)
        x2.CopyFloatDataFromHostPtr([
            0.0736655, 0.0459045, 0.0779517, 0.0771059, 0.0586862, 0.0561263,
            0.0708457, 0.0977273, 0.0405025, -0.170897, 0.0208982, 0.136865,
            -0.0367905, -0.0618205, -0.0103908, -0.0522777, -0.122161,
            -0.025427, -0.0718576, -0.185941, 0.0166533, 0.178679, -0.0576606,
            -0.137817, 0.150676, 0.153442, -0.0929899, -0.148675, -0.112459,
            -0.106284, -0.103074, -0.0668811
        ])

        handle = singa_wrap.BatchNormHandle(0.9, x)
        y2 = singa_wrap.CpuBatchNormForwardInference(handle, x2, scale, bias,
                                                     mean, var)
        self.assertListEqual([1, 2, 4, 4], list(y2.shape()))


if __name__ == '__main__':
    unittest.main()
