#
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
#

from __future__ import division

import unittest
import numpy as np
import math

from singa import singa_wrap as singa_api
from singa import tensor
from singa import device
from cuda_helper import gpu_dev, cpu_dev

# helper fn
def _np_to_pyTensor(_np):
    return tensor.Tensor(device=cpu_dev, data=_np)


def _cTensor_to_pyTensor(cTensor):
    new_t = tensor.Tensor()
    new_t.data = cTensor
    new_t.shape = tuple(new_t.data.shape())
    new_t.device = new_t.data.device()
    new_t.dtype = new_t.data.data_type()
    return new_t



class TestAPI(unittest.TestCase):
    def test_softmax_api(self):
        def _run_test(org_shape, axis, aft_shape):
            x_0 = np.random.random(org_shape).astype(np.float32)
            x_0 = x_0 + 1000
            x0 = tensor.Tensor(device=gpu_dev, data=x_0)

            # test with axis
            y0 = tensor._call_singa_func(singa_api.SoftMax, x0.data, axis)

            # test with numpy
            x_0 = x_0.reshape(aft_shape)
            x_0 = x_0 - np.max(x_0)
            y1 = np.divide(np.exp(x_0),
                           np.sum(np.exp(x_0),
                                  axis=1).reshape(x_0.shape[0],
                                                  1))  # 2d softmax
            y1 = y1.reshape(org_shape)

            np.testing.assert_array_almost_equal(tensor.to_numpy(y0), y1)

        _run_test([2, 2], 1, [2, 2])
        _run_test([2, 2], 0, [1, 4])
        _run_test([2, 2], -1, [2, 2])
        _run_test([2, 2], -2, [1, 4])

        _run_test([2, 2, 2], 2, [4, 2])
        _run_test([2, 2, 2], 1, [2, 4])
        _run_test([2, 2, 2], 0, [1, 8])
        _run_test([2, 2, 2], -1, [4, 2])
        _run_test([2, 2, 2], -2, [2, 4])
        _run_test([2, 2, 2], -3, [1, 8])

        _run_test([2, 2, 2, 2], 3, [8, 2])
        _run_test([2, 2, 2, 2], 2, [4, 4])
        _run_test([2, 2, 2, 2], 1, [2, 8])
        _run_test([2, 2, 2, 2], 0, [1, 16])
        _run_test([2, 2, 2, 2], -1, [8, 2])
        _run_test([2, 2, 2, 2], -2, [4, 4])
        _run_test([2, 2, 2, 2], -3, [2, 8])
        _run_test([2, 2, 2, 2], -4, [1, 16])

    def test_tensor_add_mul_api(self):
        def _run_test(singa_op, np_op, s1, s2):
            x_0 = np.random.random(s1).astype(np.float32)
            y_0 = np.random.random(s2).astype(np.float32)
            x0 = tensor.Tensor(device=gpu_dev, data=x_0)
            y0 = tensor.Tensor(device=gpu_dev, data=y_0)

            z0 = tensor._call_singa_func(singa_op, x0.data, y0.data)
            z0.to_host()
            #z0 = x0*y0

            #print(s1,s2,tensor.to_numpy(z0).shape)
            np.testing.assert_array_almost_equal(tensor.to_numpy(z0),
                                                 np_op(x_0, y_0))
            return

        _run_test(singa_api.__add__, np.add, [6], [1])
        _run_test(singa_api.__add__, np.add, [3, 2], [1])
        _run_test(singa_api.__add__, np.add, [3, 1, 2], [3, 1, 1])
        _run_test(singa_api.__add__, np.add, [2, 3, 4, 5], [5])
        _run_test(singa_api.__add__, np.add, [2, 3, 4, 5], [1, 1, 1])
        _run_test(singa_api.__add__, np.add, [2, 3, 4, 5], [1, 1, 1, 1])
        _run_test(singa_api.__add__, np.add, [3, 1, 2, 1], [3, 1, 2])
        _run_test(singa_api.__add__, np.add, [4, 5],
                  [2, 3, 4, 5])  # 45+2345=2345
        _run_test(singa_api.__add__, np.add, [2, 3, 4, 5],
                  [4, 5])  # 45+2345=2345
        _run_test(singa_api.__add__, np.add, [1, 4, 5],
                  [2, 3, 1, 1])  # 145+2311=2345
        _run_test(singa_api.__add__, np.add, [3, 4, 5],
                  [2, 1, 1, 1])  # 345+2111=2345

        _run_test(singa_api.__mul__, np.multiply, [2, 3], [2, 3])
        _run_test(singa_api.__mul__, np.multiply, [6], [1])
        _run_test(singa_api.__mul__, np.multiply, [3, 2], [1])
        _run_test(singa_api.__mul__, np.multiply, [3, 1, 2], [3, 1, 1])
        _run_test(singa_api.__mul__, np.multiply, [2, 3, 4, 5], [5])
        _run_test(singa_api.__mul__, np.multiply, [2, 3, 4, 5], [1, 1, 1])
        _run_test(singa_api.__mul__, np.multiply, [2, 3, 4, 5], [1, 1, 1, 1])
        _run_test(singa_api.__mul__, np.multiply, [3, 1, 2, 1], [3, 1, 2])
        _run_test(singa_api.__mul__, np.multiply, [4, 5],
                  [2, 3, 4, 5])  # 45+2345=2345
        _run_test(singa_api.__mul__, np.multiply, [2, 3, 4, 5],
                  [4, 5])  # 45+2345=2345
        _run_test(singa_api.__mul__, np.multiply, [1, 4, 5],
                  [2, 3, 1, 1])  # 145+2311=2345
        _run_test(singa_api.__mul__, np.multiply, [3, 4, 5],
                  [2, 1, 1, 1])  # 345+2111=2345

    def test_transpose_and_mul(self):
        s1 = [3, 2, 1, 1]
        s2 = [3, 2, 1, 1]
        x_0 = np.random.random(s1).astype(np.float32)
        y_0 = np.random.random(s2).astype(np.float32)

        x0 = tensor.Tensor(device=gpu_dev, data=x_0)
        y0 = tensor.Tensor(device=gpu_dev, data=y_0)
        x1 = x0.transpose([3, 2, 1, 0])

        z0 = x1 * y0
        np.testing.assert_array_almost_equal(tensor.to_numpy(z0),
                                             x_0.transpose() * y_0)

    def test_transpose_and_mul_2(self):
        def _test(s1, s2, axis1, axis2, s3):
            x_0 = np.random.random(s1).astype(np.float32)
            y_0 = np.random.random(s2).astype(np.float32)

            x0 = tensor.Tensor(device=gpu_dev, data=x_0)
            y0 = tensor.Tensor(device=gpu_dev, data=y_0)

            x1 = x0.transpose(axis1)
            y1 = y0.transpose(axis2)

            z0 = x1 * y1
            np.testing.assert_array_almost_equal(
                tensor.to_numpy(z0),
                x_0.transpose(axis1) * y_0.transpose(axis2))
            np.testing.assert_array_almost_equal(z0.shape, s3)

        s1 = [1, 5, 1, 3]
        s2 = [3, 1, 1, 4]
        axis1 = [3, 2, 1, 0]  # 3121
        axis2 = [1, 0, 2, 3]  # 1314
        s3 = [3, 3, 5, 4]
        _test(s1, s2, axis1, axis2, s3)

        s1 = [1, 5, 1]
        s2 = [1, 3, 2]
        axis1 = [2, 1, 0]  # 151
        axis2 = [1, 0, 2]  # 312
        s3 = [3, 5, 2]
        _test(s1, s2, axis1, axis2, s3)

        s1 = [5, 1]
        s2 = [1, 3]
        axis1 = [1, 0]  # 15
        axis2 = [1, 0]  # 31
        s3 = [3, 5]
        _test(s1, s2, axis1, axis2, s3)


    def test_batchnorm_training(self):
        def _np_bn_training(x, scale, bias, rm, rv, momentum=0.1, e=1e-5):
            channel = x.shape[1]
            np.testing.assert_array_almost_equal(scale.shape, (1, channel, 1, 1))
            np.testing.assert_array_almost_equal(bias.shape, (1, channel, 1, 1))
            np.testing.assert_array_almost_equal(rm.shape, (1, channel, 1, 1))
            np.testing.assert_array_almost_equal(rv.shape, (1, channel, 1, 1))

            batch_m = x.mean(axis=(0, 2, 3), keepdims=True)
            batch_v = x.var(axis=(0, 2, 3), keepdims=True)

            x_norm = (x - batch_m) / np.sqrt(batch_v + e)
            y_norm = x_norm * scale + bias

            # https://arxiv.org/pdf/1502.03167.pdf
            s = list(x.shape)
            s[1] = 1
            batch_v_unbiased = np.prod(s) * batch_v / (np.prod(s) - 1)

            rm = momentum * batch_m + (1 - momentum) * rm
            rv = momentum * batch_v_unbiased + (1 - momentum) * rv

            # https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnBatchNormalizationForwardTraining
            # this value is useful for bwd computation
            resultSaveInvVariance = 1 / np.sqrt(batch_v)
            return y_norm, rm, rv, batch_m, resultSaveInvVariance

        def _run_training(x_0, s_0, b_0, rm_0, rv_0, m_0=0.1):
            # np api
            (y_1, rm_1, rv_1, bm_1, bv_1) = _np_bn_training(x_0,
                                                            s_0,
                                                            b_0,
                                                            rm_0,
                                                            rv_0,
                                                            momentum=m_0)

            # singa api
            hndl = singa_api.BatchNormHandle(m_0, _np_to_pyTensor(x_0).data)
            (y_2_c, rm_2_c, rv_2_c, bm_2_c,
             bv_2_c) = singa_api.CpuBatchNormForwardTraining(
                 hndl,
                 _np_to_pyTensor(x_0).data,
                 _np_to_pyTensor(s_0).data,
                 _np_to_pyTensor(b_0).data,
                 _np_to_pyTensor(rm_0).data,
                 _np_to_pyTensor(rv_0).data)

            np.testing.assert_array_almost_equal(
                y_1, tensor.to_numpy(_cTensor_to_pyTensor(y_2_c)))
            #np.testing.assert_array_almost_equal(
            #    bm_1, tensor.to_numpy(_cTensor_to_pyTensor(bm_2_c)))
            np.testing.assert_array_almost_equal(
                rm_1, tensor.to_numpy(_cTensor_to_pyTensor(rm_2_c)))
            #print(bv_1)
            #print(tensor.to_numpy(_cTensor_to_pyTensor(bv_2_c)))
            #np.testing.assert_array_almost_equal(
            #    bv_1, tensor.to_numpy(_cTensor_to_pyTensor(bv_2_c)), decimal=3)
            np.testing.assert_array_almost_equal(
                rv_1, tensor.to_numpy(_cTensor_to_pyTensor(rv_2_c)), decimal=4)
            return

        x_0 = np.array(
            [1, 1, 1, 1, 2, 2, 2, 2, 10, 10, 10, 10, 20, 20, 20, 20],
            dtype=np.float32).reshape((2, 2, 2, 2))
        s_0 = np.array([1, 10], dtype=np.float32).reshape((1, 2, 1, 1))
        b_0 = np.array([1, 10], dtype=np.float32).reshape((1, 2, 1, 1))
        rm_0 = np.array([1, 10], dtype=np.float32).reshape((1, 2, 1, 1))
        rv_0 = np.array([1, 10], dtype=np.float32).reshape((1, 2, 1, 1))
        _run_training(x_0, s_0, b_0, rm_0, rv_0, m_0=1.0)
        _run_training(x_0, s_0, b_0, rm_0, rv_0, m_0=0.0)
        _run_training(x_0, s_0, b_0, rm_0, rv_0, m_0=0.2)

        c = 10
        x_0 = np.random.random((10, c, 20, 20)).astype(np.float32)
        s_0 = np.random.random((1, c, 1, 1)).astype(np.float32)
        b_0 = np.random.random((1, c, 1, 1)).astype(np.float32)
        rm_0 = np.random.random((1, c, 1, 1)).astype(np.float32)
        rv_0 = np.random.random((1, c, 1, 1)).astype(np.float32)
        _run_training(x_0, s_0, b_0, rm_0, rv_0, m_0=0.2)

    def test_batchnorm_testing(self):
        def _np_bn_testing(x, scale, bias, rm, rv, momentum=0.1, e=1e-5):
            channel = x.shape[1]
            np.testing.assert_array_almost_equal(scale.shape, (1, channel, 1, 1))
            np.testing.assert_array_almost_equal(bias.shape, (1, channel, 1, 1))
            np.testing.assert_array_almost_equal(rm.shape, (1, channel, 1, 1))
            np.testing.assert_array_almost_equal(rv.shape, (1, channel, 1, 1))
            return scale * (x - rm) / np.sqrt(rv + e) + bias

        def _run_testing(x_0, s_0, b_0, rm_0, rv_0, m_0=0.1):
            # np api
            y_1 = _np_bn_testing(x_0, s_0, b_0, rm_0, rv_0, momentum=m_0)

            # singa api
            hndl = singa_api.BatchNormHandle(m_0, _np_to_pyTensor(x_0).data)
            y_2_c = singa_api.CpuBatchNormForwardInference(
                hndl,
                _np_to_pyTensor(x_0).data,
                _np_to_pyTensor(s_0).data,
                _np_to_pyTensor(b_0).data,
                _np_to_pyTensor(rm_0).data,
                _np_to_pyTensor(rv_0).data)
            #print(y_1)
            #print(tensor.to_numpy(_cTensor_to_pyTensor(y_2_c)))

            np.testing.assert_array_almost_equal(
                y_1, tensor.to_numpy(_cTensor_to_pyTensor(y_2_c)), decimal=5)
            return

        x_0 = np.array(
            [1, 1, 1, 1, 2, 2, 2, 2, 10, 10, 10, 10, 20, 20, 20, 20],
            dtype=np.float32).reshape((2, 2, 2, 2))
        s_0 = np.array([1, 10], dtype=np.float32).reshape((1, 2, 1, 1))
        b_0 = np.array([1, 10], dtype=np.float32).reshape((1, 2, 1, 1))
        rm_0 = np.array([1, 10], dtype=np.float32).reshape((1, 2, 1, 1))
        rv_0 = np.array([1, 10], dtype=np.float32).reshape((1, 2, 1, 1))
        _run_testing(x_0, s_0, b_0, rm_0, rv_0, m_0=1.0)
        c = 10
        x_0 = np.random.random((10, c, 20, 20)).astype(np.float32)
        s_0 = np.random.random((1, c, 1, 1)).astype(np.float32)
        b_0 = np.random.random((1, c, 1, 1)).astype(np.float32)
        rm_0 = np.random.random((1, c, 1, 1)).astype(np.float32)
        rv_0 = np.random.random((1, c, 1, 1)).astype(np.float32)
        _run_testing(x_0, s_0, b_0, rm_0, rv_0, m_0=1.0)

    def test_batchnorm_backward(self):
        N=1; C=3; H=2; W=2

        data_shape = [N,C,H,W]
        param_shape = [1,C,1,1]
        data = [1,2,3,4,5,6,7,8,9,10,11,12]

        x_0 = np.array( data, dtype=np.float32).reshape(data_shape)
        y_0 = np.array( data, dtype=np.float32).reshape(data_shape)
        dy_0 = np.array( data, dtype=np.float32).reshape(data_shape)
        scale_0 = np.array([1]*C, dtype=np.float32).reshape(param_shape)
        bias_0 = np.array([0]*C, dtype=np.float32).reshape(param_shape)

        mean_0 = x_0.mean(axis=(0, 2, 3), keepdims=True)
        var_0 = x_0.var(axis=(0, 2, 3), keepdims=True)

        hndl = singa_api.BatchNormHandle(0.1, _np_to_pyTensor(x_0).data)
        (dx_2_c, _, _) = singa_api.CpuBatchNormBackwardx(
            hndl,
            _np_to_pyTensor(y_0).data,
            _np_to_pyTensor(dy_0).data,
            _np_to_pyTensor(x_0).data,
            _np_to_pyTensor(scale_0).data,
            _np_to_pyTensor(bias_0).data,
            _np_to_pyTensor(mean_0).data,
            _np_to_pyTensor(var_0).data,
        )

        dx_truth = np.array([[[[-1.0769e-05, -3.5985e-06],
          [ 3.5985e-06,  1.0769e-05]],
         [[-1.0769e-05, -3.5985e-06],
          [ 3.5985e-06,  1.0769e-05]],
         [[-1.0769e-05, -3.5985e-06],
          [ 3.5985e-06,  1.0769e-05]]]])
        np.testing.assert_array_almost_equal(tensor.to_numpy(_cTensor_to_pyTensor(dx_2_c)), dx_truth)

        return


    def test_softmax_api_dnnl_backend(self):
        def _run_test(org_shape, axis, aft_shape):
            x_0 = np.random.random(org_shape).astype(np.float32)
            x_0 = x_0 + 1000
            x0 = tensor.Tensor(device=cpu_dev, data=x_0)

            # test with axis
            y0 = tensor._call_singa_func(singa_api.SoftMax, x0.data, axis)

            # test with numpy
            x_0 = x_0.reshape(aft_shape)
            x_0 = x_0 - np.max(x_0)
            y1 = np.divide(np.exp(x_0),
                           np.sum(np.exp(x_0),
                                  axis=1).reshape(x_0.shape[0],
                                                  1))  # 2d softmax
            y1 = y1.reshape(org_shape)

            np.testing.assert_array_almost_equal(tensor.to_numpy(y0), y1)

        _run_test([2, 2], 1, [2, 2])
        _run_test([2, 2], 0, [1, 4])
        _run_test([2, 2], -1, [2, 2])
        _run_test([2, 2], -2, [1, 4])

        _run_test([2, 2, 2], 2, [4, 2])
        _run_test([2, 2, 2], 1, [2, 4])
        _run_test([2, 2, 2], 0, [1, 8])
        _run_test([2, 2, 2], -1, [4, 2])
        _run_test([2, 2, 2], -2, [2, 4])
        _run_test([2, 2, 2], -3, [1, 8])

        _run_test([2, 2, 2, 2], 3, [8, 2])
        _run_test([2, 2, 2, 2], 2, [4, 4])
        _run_test([2, 2, 2, 2], 1, [2, 8])
        _run_test([2, 2, 2, 2], 0, [1, 16])
        _run_test([2, 2, 2, 2], -1, [8, 2])
        _run_test([2, 2, 2, 2], -2, [4, 4])
        _run_test([2, 2, 2, 2], -3, [2, 8])
        _run_test([2, 2, 2, 2], -4, [1, 16])

    def test_dnnl_pooling_max(self):
        N=1; C=3; H=2; W=2

        data_shape = [N,C,H,W]
        param_shape = [1,C,1,1]
        data = [1,2,3,4,5,6,7,8,9,10,11,12]

        x0 = np.array( data, dtype=np.float32).reshape(data_shape)
        x0_ct = _np_to_pyTensor(x0).data

        dy0 = np.array([1,2,3], dtype=np.float32).reshape([1,3,1,1])
        dy0_ct = _np_to_pyTensor(dy0).data

        hndl = singa_api.PoolingHandle(x0_ct, [2, 2], [1,1], [0,0], True)

        y0_ct=singa_api.CpuPoolingForward(hndl, x0_ct);
        y1 = np.array([[[[ 4.]], [[ 8.]], [[12.]]]])
        np.testing.assert_array_almost_equal(tensor.to_numpy(_cTensor_to_pyTensor(y0_ct)), y1)

        dx0_ct = singa_api.CpuPoolingBackward(hndl, dy0_ct, x0_ct, y0_ct);
        dx1 = np.array( [[[[0., 0.], [0., 1.]], [[0., 0.], [0., 2.]], [[0., 0.], [0., 3.]]]] )
        np.testing.assert_array_almost_equal(tensor.to_numpy(_cTensor_to_pyTensor(dx0_ct)), dx1)

    def test_dnnl_pooling_avg(self):
        N=1; C=3; H=2; W=2

        data_shape = [N,C,H,W]
        param_shape = [1,C,1,1]
        data = [1,2,3,4,5,6,7,8,9,10,11,12]

        x0 = np.array( data, dtype=np.float32).reshape(data_shape)
        x0_ct = _np_to_pyTensor(x0).data

        dy0 = np.array([1,2,3], dtype=np.float32).reshape([1,3,1,1])
        dy0_ct = _np_to_pyTensor(dy0).data

        hndl = singa_api.PoolingHandle(x0_ct, [2, 2], [1,1], [0,0], False)

        y0_ct=singa_api.CpuPoolingForward(hndl, x0_ct);

        y1 = np.array([[[[ 2.5000]], [[ 6.5000]], [[10.5000]]]])
        np.testing.assert_array_almost_equal(tensor.to_numpy(_cTensor_to_pyTensor(y0_ct)), y1)
        dx0_ct = singa_api.CpuPoolingBackward(hndl, dy0_ct, x0_ct, y0_ct);
        dx1=np.array([[[[0.2500, 0.2500], [0.2500, 0.2500]], [[0.5000, 0.5000], [0.5000, 0.5000]], [[0.7500, 0.7500], [0.7500, 0.7500]]]])
        np.testing.assert_array_almost_equal(tensor.to_numpy(_cTensor_to_pyTensor(dx0_ct)), dx1)


if __name__ == '__main__':
    unittest.main()
