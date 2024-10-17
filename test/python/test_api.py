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
import math
import numpy as np

from singa import singa_wrap as singa_api
from singa import tensor
from cuda_helper import gpu_dev, cpu_dev


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
    resultSaveInvVariance = 1 / np.sqrt(batch_v)
    return y_norm, rm, rv, batch_m, resultSaveInvVariance


def _np_bn_testing(x, scale, bias, rm, rv, momentum=0.1, e=1e-5):
    channel = x.shape[1]
    np.testing.assert_array_almost_equal(scale.shape, (1, channel, 1, 1))
    np.testing.assert_array_almost_equal(bias.shape, (1, channel, 1, 1))
    np.testing.assert_array_almost_equal(rm.shape, (1, channel, 1, 1))
    np.testing.assert_array_almost_equal(rv.shape, (1, channel, 1, 1))
    return scale * (x - rm) / np.sqrt(rv + e) + bias


def _cTensor_to_pyTensor(cTensor):
    new_t = tensor.Tensor()
    new_t.data = cTensor
    new_t.shape = tuple(new_t.data.shape())
    new_t.device = new_t.data.device()
    new_t.dtype = new_t.data.data_type()
    return new_t


def _ctensor_eq_ndarray(t1, np1):
    d = t1.device()
    t1.ToHost()
    if t1.data_type() == singa_api.kInt:
        np.testing.assert_array_almost_equal(t1.GetIntValue(t1.Size()),
                                             np1.flatten())
    elif t1.data_type() == singa_api.kFloat32:
        np.testing.assert_array_almost_equal(t1.GetFloatValue(t1.Size()),
                                             np1.flatten())

    if np1.dtype == np.float32:
        np.testing.assert_equal(t1.data_type(), singa_api.kFloat32)
    elif np1.dtype == np.int32:
        np.testing.assert_equal(t1.data_type(), singa_api.kInt)

    np.testing.assert_array_almost_equal(t1.shape(), np1.shape)
    t1.ToDevice(d)


def print_t(t1):
    d = t1.device()
    t1.ToHost()
    if t1.data_type() == singa_api.kInt:
        print(t1.GetIntValue(t1.Size()))
    elif t1.data_type() == singa_api.kFloat32:
        print(t1.GetFloatValue(t1.Size()))
    t1.ToDevice(d)


class TestAPI(unittest.TestCase):

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_batchnorm_training_gpu(self):
        dev = gpu_dev

        def _run_training(x_0, s_0, b_0, rm_0, rv_0, m_0=0.1):
            # np api
            (y_1, rm_1, rv_1, bm_1, bv_1) = _np_bn_training(x_0,
                                                            s_0,
                                                            b_0,
                                                            rm_0,
                                                            rv_0,
                                                            momentum=m_0)

            # singa api
            rm_t = tensor.Tensor(device=dev, data=rm_0)
            rv_t = tensor.Tensor(device=dev, data=rv_0)
            hndl = singa_api.CudnnBatchNormHandle(
                m_0,
                tensor.Tensor(device=dev, data=x_0).data)
            (y_2_c, bm_2_c, bv_2_c) = singa_api.GpuBatchNormForwardTraining(
                hndl,
                tensor.Tensor(device=dev, data=x_0).data,
                tensor.Tensor(device=dev, data=s_0).data,
                tensor.Tensor(device=dev, data=b_0).data, rm_t.data, rv_t.data)

            np.testing.assert_array_almost_equal(
                y_1, tensor.to_numpy(_cTensor_to_pyTensor(y_2_c)), decimal=4)
            np.testing.assert_array_almost_equal(
                bm_1, tensor.to_numpy(_cTensor_to_pyTensor(bm_2_c)))
            np.testing.assert_array_almost_equal(rm_1, tensor.to_numpy(rm_t))
            #print(bv_1)
            #print(tensor.to_numpy(_cTensor_to_pyTensor(bv_2_c)))
            np.testing.assert_array_almost_equal(
                bv_1, tensor.to_numpy(_cTensor_to_pyTensor(bv_2_c)), decimal=3)
            np.testing.assert_array_almost_equal(rv_1,
                                                 tensor.to_numpy(rv_t),
                                                 decimal=4)
            return

        x_0 = np.array([1, 1, 1, 1, 2, 2, 2, 2, 10, 10, 10, 10, 20, 20, 20, 20],
                       dtype=np.float32).reshape((2, 2, 2, 2))
        s_0 = np.array([1, 10], dtype=np.float32).reshape((1, 2, 1, 1))
        b_0 = np.array([1, 10], dtype=np.float32).reshape((1, 2, 1, 1))
        rm_0 = np.array([1, 10], dtype=np.float32).reshape((1, 2, 1, 1))
        rv_0 = np.array([1, 10], dtype=np.float32).reshape((1, 2, 1, 1))
        _run_training(x_0, s_0, b_0, rm_0, rv_0, m_0=0.0)
        _run_training(x_0, s_0, b_0, rm_0, rv_0, m_0=1.0)
        _run_training(x_0, s_0, b_0, rm_0, rv_0, m_0=0.2)

        c = 10
        x_0 = np.random.random((10, c, 20, 20)).astype(np.float32)
        s_0 = np.random.random((1, c, 1, 1)).astype(np.float32)
        b_0 = np.random.random((1, c, 1, 1)).astype(np.float32)
        rm_0 = np.random.random((1, c, 1, 1)).astype(np.float32)
        rv_0 = np.random.random((1, c, 1, 1)).astype(np.float32)
        _run_training(x_0, s_0, b_0, rm_0, rv_0, m_0=0.2)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_batchnorm_testing_gpu(self):
        dev = gpu_dev

        def _run_testing(x_0, s_0, b_0, rm_0, rv_0, m_0=0.1):
            # np api
            y_1 = _np_bn_testing(x_0, s_0, b_0, rm_0, rv_0, momentum=m_0)

            # singa api
            hndl = singa_api.CudnnBatchNormHandle(
                m_0,
                tensor.Tensor(device=dev, data=x_0).data)
            y_2_c = singa_api.GpuBatchNormForwardInference(
                hndl,
                tensor.Tensor(device=dev, data=x_0).data,
                tensor.Tensor(device=dev, data=s_0).data,
                tensor.Tensor(device=dev, data=b_0).data,
                tensor.Tensor(device=dev, data=rm_0).data,
                tensor.Tensor(device=dev, data=rv_0).data)
            #print(y_1)
            #print(tensor.to_numpy(_cTensor_to_pyTensor(y_2_c)))

            np.testing.assert_array_almost_equal(
                y_1, tensor.to_numpy(_cTensor_to_pyTensor(y_2_c)), decimal=3)
            return

        x_0 = np.array([1, 1, 1, 1, 2, 2, 2, 2, 10, 10, 10, 10, 20, 20, 20, 20],
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

    def _softmax_api_helper(self, dev):

        def _run_test(dev, org_shape, axis, aft_shape):
            x_0 = np.random.random(org_shape).astype(np.float32)
            x_0 = x_0 + 1000
            x0 = tensor.Tensor(device=dev, data=x_0)

            # test with axis
            y0 = tensor._call_singa_func(singa_api.SoftMax, x0.data, axis)

            # test with numpy
            x_0 = x_0.reshape(aft_shape)
            x_0 = x_0 - np.max(x_0)
            y1 = np.divide(np.exp(x_0),
                           np.sum(np.exp(x_0), axis=1).reshape(x_0.shape[0],
                                                               1))  # 2d softmax
            y1 = y1.reshape(org_shape)

            np.testing.assert_array_almost_equal(tensor.to_numpy(y0), y1)

        _run_test(dev, [2, 2], 1, [2, 2])
        _run_test(dev, [2, 2], 0, [1, 4])
        _run_test(dev, [2, 2], -1, [2, 2])
        _run_test(dev, [2, 2], -2, [1, 4])
        _run_test(dev, [2, 2, 2], 2, [4, 2])
        _run_test(dev, [2, 2, 2], 1, [2, 4])
        _run_test(dev, [2, 2, 2], 0, [1, 8])
        _run_test(dev, [2, 2, 2], -1, [4, 2])
        _run_test(dev, [2, 2, 2], -2, [2, 4])
        _run_test(dev, [2, 2, 2], -3, [1, 8])
        _run_test(dev, [2, 2, 2, 2], 3, [8, 2])
        _run_test(dev, [2, 2, 2, 2], 2, [4, 4])
        _run_test(dev, [2, 2, 2, 2], 1, [2, 8])
        _run_test(dev, [2, 2, 2, 2], 0, [1, 16])
        _run_test(dev, [2, 2, 2, 2], -1, [8, 2])
        _run_test(dev, [2, 2, 2, 2], -2, [4, 4])
        _run_test(dev, [2, 2, 2, 2], -3, [2, 8])
        _run_test(dev, [2, 2, 2, 2], -4, [1, 16])

    def test_softmax_api_cpu(self):
        self._softmax_api_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_softmax_api_gpu(self):
        self._softmax_api_helper(gpu_dev)

    def _tensor_arithmetic_op_broadcast_helper(self, dev):

        def _run_test(dev, singa_op, np_op, s1, s2):
            x_0 = np.random.random(s1).astype(np.float32)
            y_0 = np.random.random(s2).astype(np.float32)
            x0 = tensor.Tensor(device=dev, data=x_0)
            y0 = tensor.Tensor(device=dev, data=y_0)

            z0 = tensor._call_singa_func(singa_op, x0.data, y0.data)
            z0.to_host()
            np.testing.assert_array_almost_equal(tensor.to_numpy(z0),
                                                 np_op(x_0, y_0))
            return

        for s_op, n_op in zip([
                singa_api.Pow,
                singa_api.__add__,
                singa_api.__div__,
                singa_api.__sub__,
                singa_api.__mul__,
        ], [np.power, np.add, np.divide, np.subtract, np.multiply]):
            _run_test(dev, s_op, n_op, [6], [1])
            _run_test(dev, s_op, n_op, [2, 3], [2, 3])
            _run_test(dev, s_op, n_op, [3, 2], [1])
            _run_test(dev, s_op, n_op, [3, 1, 2], [3, 1, 1])
            _run_test(dev, s_op, n_op, [2, 3, 4, 5], [5])
            _run_test(dev, s_op, n_op, [2, 3, 4, 5], [1, 1, 1])
            _run_test(dev, s_op, n_op, [2, 3, 4, 5], [1, 1, 1, 1])
            _run_test(dev, s_op, n_op, [2, 3, 4, 5], [4, 5])  # 45+2345=2345
            _run_test(dev, s_op, n_op, [3, 1, 2, 1], [3, 1, 2])
            _run_test(dev, s_op, n_op, [4, 5], [2, 3, 4, 5])  # 45+2345=2345
            _run_test(dev, s_op, n_op, [1, 4, 5], [2, 3, 1, 1])  # 145+2311=2345
            _run_test(dev, s_op, n_op, [3, 4, 5], [2, 1, 1, 1])  # 345+2111=2345

    def test_tensor_arithmetic_op_broadcast_cpu(self):
        self._tensor_arithmetic_op_broadcast_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_tensor_arithmetic_op_broadcast_gpu(self):
        self._tensor_arithmetic_op_broadcast_helper(gpu_dev)

    def _transpose_and_arithmetic_op_broadcast_helper(self, dev):

        def _test(s1, s2, axis1, axis2, s3, s_op, n_op, dev):
            x_0 = np.random.random(s1).astype(np.float32)
            y_0 = np.random.random(s2).astype(np.float32)

            x0 = tensor.Tensor(device=dev, data=x_0)
            y0 = tensor.Tensor(device=dev, data=y_0)

            x1 = x0.transpose(axis1)
            y1 = y0.transpose(axis2)

            z0 = tensor._call_singa_func(s_op, x1.data, y1.data)
            z0.to_host()

            np.testing.assert_array_almost_equal(
                tensor.to_numpy(z0),
                n_op(x_0.transpose(axis1), y_0.transpose(axis2)))
            np.testing.assert_array_almost_equal(z0.shape, s3)
            return

        for s_op, n_op in zip([
                singa_api.Pow,
                singa_api.__add__,
                singa_api.__div__,
                singa_api.__sub__,
                singa_api.__mul__,
        ], [np.power, np.add, np.divide, np.subtract, np.multiply]):
            s1 = [1, 5, 1, 3]
            s2 = [3, 1, 1, 4]
            axis1 = [3, 2, 1, 0]  # 3121
            axis2 = [1, 0, 2, 3]  # 1314
            s3 = [3, 3, 5, 4]
            _test(s1, s2, axis1, axis2, s3, s_op, n_op, dev)

            s1 = [1, 5, 1]
            s2 = [1, 3, 2]
            axis1 = [2, 1, 0]  # 151
            axis2 = [1, 0, 2]  # 312
            s3 = [3, 5, 2]
            _test(s1, s2, axis1, axis2, s3, s_op, n_op, dev)

            s1 = [5, 1]
            s2 = [1, 3]
            axis1 = [1, 0]  # 15
            axis2 = [1, 0]  # 31
            s3 = [3, 5]
            _test(s1, s2, axis1, axis2, s3, s_op, n_op, dev)

    def test_transpose_and_arithmetic_op_broadcast_cpu(self):
        self._transpose_and_arithmetic_op_broadcast_helper(cpu_dev)

    def _erf(self, dev=cpu_dev):
        np1 = np.random.random((2, 3)).astype(np.float32)

        x1 = tensor.from_numpy(np1)
        x1.to_device(dev)
        y1 = tensor.from_raw_tensor(singa_api.Erf(x1.data))

        # from scipy.special import erf
        # np.testing.assert_array_almost_equal(erf(np1), tensor.to_numpy(y1))

    def test_erf_cpu(self):
        self._erf(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_transpose_and_arithmetic_op_broadcast_gpu(self):
        self._transpose_and_arithmetic_op_broadcast_helper(gpu_dev)

    def test_batchnorm_training_dnnl(self):
        dev = cpu_dev

        def _np_bn_training(x, scale, bias, rm, rv, momentum=0.1, e=1e-5):
            channel = x.shape[1]
            np.testing.assert_array_almost_equal(scale.shape,
                                                 (1, channel, 1, 1))
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
            hndl = singa_api.BatchNormHandle(
                m_0,
                tensor.Tensor(device=dev, data=x_0).data)
            (y_2_c, bm_2_c, bv_2_c) = singa_api.CpuBatchNormForwardTraining(
                hndl,
                tensor.Tensor(device=dev, data=x_0).data,
                tensor.Tensor(device=dev, data=s_0).data,
                tensor.Tensor(device=dev, data=b_0).data,
                tensor.Tensor(device=dev, data=rm_0).data,
                tensor.Tensor(device=dev, data=rv_0).data)

            np.testing.assert_array_almost_equal(
                y_1, tensor.to_numpy(_cTensor_to_pyTensor(y_2_c)), decimal=5)
            np.testing.assert_array_almost_equal(
                bm_1, tensor.to_numpy(_cTensor_to_pyTensor(bm_2_c)), decimal=5)
            #print(bv_1)
            #print(tensor.to_numpy(_cTensor_to_pyTensor(bv_2_c)))
            #np.testing.assert_array_almost_equal(
            #    bv_1, tensor.to_numpy(_cTensor_to_pyTensor(bv_2_c)), decimal=3)
            return

        x_0 = np.array([1, 1, 1, 1, 2, 2, 2, 2, 10, 10, 10, 10, 20, 20, 20, 20],
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

    def test_batchnorm_testing_dnnl(self):
        dev = cpu_dev

        def _np_bn_testing(x, scale, bias, rm, rv, momentum=0.1, e=1e-5):
            channel = x.shape[1]
            np.testing.assert_array_almost_equal(scale.shape,
                                                 (1, channel, 1, 1))
            np.testing.assert_array_almost_equal(bias.shape, (1, channel, 1, 1))
            np.testing.assert_array_almost_equal(rm.shape, (1, channel, 1, 1))
            np.testing.assert_array_almost_equal(rv.shape, (1, channel, 1, 1))
            return scale * (x - rm) / np.sqrt(rv + e) + bias

        def _run_testing(x_0, s_0, b_0, rm_0, rv_0, m_0=0.1):
            # np api
            y_1 = _np_bn_testing(x_0, s_0, b_0, rm_0, rv_0, momentum=m_0)

            # singa api
            hndl = singa_api.BatchNormHandle(
                m_0,
                tensor.Tensor(device=dev, data=x_0).data)
            y_2_c = singa_api.CpuBatchNormForwardInference(
                hndl,
                tensor.Tensor(device=dev, data=x_0).data,
                tensor.Tensor(device=dev, data=s_0).data,
                tensor.Tensor(device=dev, data=b_0).data,
                tensor.Tensor(device=dev, data=rm_0).data,
                tensor.Tensor(device=dev, data=rv_0).data)
            #print(y_1)
            #print(tensor.to_numpy(_cTensor_to_pyTensor(y_2_c)))

            np.testing.assert_array_almost_equal(
                y_1, tensor.to_numpy(_cTensor_to_pyTensor(y_2_c)), decimal=5)
            return

        x_0 = np.array([1, 1, 1, 1, 2, 2, 2, 2, 10, 10, 10, 10, 20, 20, 20, 20],
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

    def test_batchnorm_backward_dnnl(self):
        dev = cpu_dev
        N = 1
        C = 3
        H = 2
        W = 2

        data_shape = [N, C, H, W]
        param_shape = [1, C, 1, 1]
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

        x_0 = np.array(data, dtype=np.float32).reshape(data_shape)
        y_0 = np.array(data, dtype=np.float32).reshape(data_shape)
        dy_0 = np.array(data, dtype=np.float32).reshape(data_shape)
        scale_0 = np.array([1] * C, dtype=np.float32).reshape(param_shape)
        bias_0 = np.array([0] * C, dtype=np.float32).reshape(param_shape)

        mean_0 = x_0.mean(axis=(0, 2, 3), keepdims=True)
        var_0 = x_0.var(axis=(0, 2, 3), keepdims=True)

        hndl = singa_api.BatchNormHandle(
            0.1,
            tensor.Tensor(device=dev, data=x_0).data)
        (dx_2_c, _, _) = singa_api.CpuBatchNormBackwardx(
            hndl,
            tensor.Tensor(device=dev, data=y_0).data,
            tensor.Tensor(device=dev, data=dy_0).data,
            tensor.Tensor(device=dev, data=x_0).data,
            tensor.Tensor(device=dev, data=scale_0).data,
            tensor.Tensor(device=dev, data=bias_0).data,
            tensor.Tensor(device=dev, data=mean_0).data,
            tensor.Tensor(device=dev, data=var_0).data,
        )

        dx_truth = np.array([[[[-1.0769e-05, -3.5985e-06],
                               [3.5985e-06, 1.0769e-05]],
                              [[-1.0769e-05, -3.5985e-06],
                               [3.5985e-06, 1.0769e-05]],
                              [[-1.0769e-05, -3.5985e-06],
                               [3.5985e-06, 1.0769e-05]]]])
        np.testing.assert_array_almost_equal(
            tensor.to_numpy(_cTensor_to_pyTensor(dx_2_c)), dx_truth)

        return

    def test_softmax_api_dnnl_backend(self):
        dev = cpu_dev

        def _run_test(org_shape, axis, aft_shape):
            x_0 = np.random.random(org_shape).astype(np.float32)
            x_0 = x_0 + 1000
            x0 = tensor.Tensor(device=dev, data=x_0)

            # test with axis
            y0 = tensor._call_singa_func(singa_api.SoftMax, x0.data, axis)

            # test with numpy
            x_0 = x_0.reshape(aft_shape)
            x_0 = x_0 - np.max(x_0)
            y1 = np.divide(np.exp(x_0),
                           np.sum(np.exp(x_0), axis=1).reshape(x_0.shape[0],
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
        dev = cpu_dev
        N = 1
        C = 3
        H = 2
        W = 2

        data_shape = [N, C, H, W]
        param_shape = [1, C, 1, 1]
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

        x0 = np.array(data, dtype=np.float32).reshape(data_shape)
        x0_ct = tensor.Tensor(device=dev, data=x0).data

        dy0 = np.array([1, 2, 3], dtype=np.float32).reshape([1, 3, 1, 1])
        dy0_ct = tensor.Tensor(device=dev, data=dy0).data

        hndl = singa_api.PoolingHandle(x0_ct, [2, 2], [1, 1], [0, 0], True)

        y0_ct = singa_api.CpuPoolingForward(hndl, x0_ct)
        y1 = np.array([[[[4.]], [[8.]], [[12.]]]])
        np.testing.assert_array_almost_equal(
            tensor.to_numpy(_cTensor_to_pyTensor(y0_ct)), y1)

        dx0_ct = singa_api.CpuPoolingBackward(hndl, dy0_ct, x0_ct, y0_ct)
        dx1 = np.array([[[[0., 0.], [0., 1.]], [[0., 0.], [0., 2.]],
                         [[0., 0.], [0., 3.]]]])
        np.testing.assert_array_almost_equal(
            tensor.to_numpy(_cTensor_to_pyTensor(dx0_ct)), dx1)

    def test_dnnl_pooling_avg(self):
        dev = cpu_dev
        N = 1
        C = 3
        H = 2
        W = 2

        data_shape = [N, C, H, W]
        param_shape = [1, C, 1, 1]
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

        x0 = np.array(data, dtype=np.float32).reshape(data_shape)
        x0_ct = tensor.Tensor(device=dev, data=x0).data

        dy0 = np.array([1, 2, 3], dtype=np.float32).reshape([1, 3, 1, 1])
        dy0_ct = tensor.Tensor(device=dev, data=dy0).data

        hndl = singa_api.PoolingHandle(x0_ct, [2, 2], [1, 1], [0, 0], False)

        y0_ct = singa_api.CpuPoolingForward(hndl, x0_ct)

        y1 = np.array([[[[2.5000]], [[6.5000]], [[10.5000]]]])
        np.testing.assert_array_almost_equal(
            tensor.to_numpy(_cTensor_to_pyTensor(y0_ct)), y1)
        dx0_ct = singa_api.CpuPoolingBackward(hndl, dy0_ct, x0_ct, y0_ct)
        dx1 = np.array([[[[0.2500, 0.2500], [0.2500, 0.2500]],
                         [[0.5000, 0.5000], [0.5000, 0.5000]],
                         [[0.7500, 0.7500], [0.7500, 0.7500]]]])
        np.testing.assert_array_almost_equal(
            tensor.to_numpy(_cTensor_to_pyTensor(dx0_ct)), dx1)

    def _concat_helper(self, dev):
        np1 = np.random.random([5, 6, 7, 8]).astype(np.float32)
        np2 = np.random.random([5, 6, 7, 1]).astype(np.float32)
        np3 = np.concatenate((np1, np2), axis=3)

        t1 = tensor.Tensor(device=dev, data=np1)
        t2 = tensor.Tensor(device=dev, data=np2)

        ctensors = singa_api.VecTensor()
        ctensors.append(t1.data)
        ctensors.append(t2.data)

        t3_ct = singa_api.ConcatOn(ctensors, 3)

        np.testing.assert_array_almost_equal(
            tensor.to_numpy(_cTensor_to_pyTensor(t3_ct)), np3)

    def test_concat_cpu(self):
        self._concat_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_concat_gpu(self):
        self._concat_helper(gpu_dev)

    def _ceil_helper(self, dev):

        np1 = np.random.random([5, 6, 7, 8]).astype(np.float32)

        np1 = np.random.random([5, 6, 7, 8]).astype(np.float32)
        np1 = np1 * 10
        np2 = np.ceil(np1)

        t1 = tensor.Tensor(device=dev, data=np1)

        t2_ct = singa_api.Ceil(t1.data)

        np.testing.assert_array_almost_equal(
            tensor.to_numpy(_cTensor_to_pyTensor(t2_ct)), np2)

    def test_ceil_cpu(self):
        self._ceil_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_ceil_gpu(self):
        self._ceil_helper(gpu_dev)

    def _floor_helper(self, dev):

        np1 = np.random.random([5, 6, 7, 8]).astype(np.float32)

        np1 = np.random.random([5, 6, 7, 8]).astype(np.float32)
        np1 = np1 * 10
        np2 = np.floor(np1)

        t1 = tensor.Tensor(device=dev, data=np1)

        t2_ct = singa_api.Floor(t1.data)

        np.testing.assert_array_almost_equal(
            tensor.to_numpy(_cTensor_to_pyTensor(t2_ct)), np2)

    def test_floor_cpu(self):
        self._floor_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_floor_gpu(self):
        self._floor_helper(gpu_dev)

    def _as_type_helper(self, dev):

        np1 = np.random.random([3]).astype(np.float32)
        np1 = np1 * 10 - 5
        np2 = np1.astype(np.int32)
        np3 = np2.astype(np.float32)

        t1 = tensor.Tensor(device=dev, data=np1)

        t1 = tensor.Tensor(device=dev, data=np1)

        t1_ct = t1.data

        self.assertEqual(t1_ct.data_type(), singa_api.kFloat32)

        t1_ct = t1_ct.AsType(singa_api.kInt)

        self.assertEqual(t1_ct.data_type(), singa_api.kInt)

        np.testing.assert_array_almost_equal(
            tensor.to_numpy(_cTensor_to_pyTensor(t1_ct)), np2)

        t1_ct = t1_ct.AsType(singa_api.kFloat32)

        self.assertEqual(t1_ct.data_type(), singa_api.kFloat32)

        np.testing.assert_array_almost_equal(
            tensor.to_numpy(_cTensor_to_pyTensor(t1_ct)), np3)

    def test_as_type_cpu(self):
        self._as_type_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_as_type_gpu(self):
        self._as_type_helper(gpu_dev)

    def _as_type2_helper(self, dev):
        shape1 = [1, 2, 3, 4]
        shape2 = [4, 3, 2, 1]
        np_int = np.random.randint(0, 10, shape1).astype(np.int32)
        np_flt = np_int.astype(np.float32)

        t1 = singa_api.Tensor(shape1, dev, singa_api.kInt)
        t1.CopyIntDataFromHostPtr(np_int.flatten())
        _ctensor_eq_ndarray(t1, np_int)

        t1 = singa_api.Reshape(t1, shape2)
        t2 = t1.AsType(singa_api.kFloat32)
        _ctensor_eq_ndarray(t2, np_flt.reshape(shape2))

        t3 = t2.AsType(singa_api.kInt)
        _ctensor_eq_ndarray(t3, np_int.reshape(shape2))

        t1 = singa_api.Reshape(t1, shape1)
        t4 = t1.AsType(singa_api.kFloat32)
        _ctensor_eq_ndarray(t4, np_flt.reshape(shape1))

    def test_as_type2_cpu(self):
        self._as_type2_helper(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_as_type2_gpu(self):
        self._as_type2_helper(gpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_rnn_relu(self):
        self._rnn_helper(0)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_rnn_tanh(self):
        self._rnn_helper(1)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_rnn_lstm(self):
        self._rnn_helper(2)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_rnn_gru(self):
        self._rnn_helper(3)

    def _rnn_helper(self, mode):
        dev = gpu_dev

        hidden_size = 7
        seq_length = 5
        batch_size = 6
        feature_size = 3
        directions = 2
        num_layers = 2

        x = tensor.Tensor(shape=(seq_length, batch_size, feature_size),
                          device=dev).gaussian(0, 1)
        hx = tensor.Tensor(shape=(num_layers * directions, batch_size,
                                  hidden_size),
                           device=dev).gaussian(0, 1)
        cx = tensor.Tensor(shape=(num_layers * directions, batch_size,
                                  hidden_size),
                           device=dev).gaussian(0, 1)

        rnn_handle = singa_api.CudnnRNNHandle(x.data,
                                              hidden_size,
                                              mode,
                                              num_layers=num_layers,
                                              dropout=0.1,
                                              bidirectional=1)

        w = tensor.Tensor(shape=(rnn_handle.weights_size,),
                          device=dev).gaussian(0, 1)
        # print("weights size is ", rnn_handle.weights_size)

        (y, hy, cy) = singa_api.GpuRNNForwardTraining(x.data, hx.data, cx.data,
                                                      w.data, rnn_handle)
        self.assertEqual(y.shape(),
                         (seq_length, batch_size, directions * hidden_size))
        self.assertEqual(hy.shape(), hx.shape)
        self.assertEqual(cy.shape(), cx.shape)

        (y2, hy2,
         cy2) = singa_api.GpuRNNForwardInference(x.data, hx.data, cx.data,
                                                 w.data, rnn_handle)
        self.assertEqual(y2.shape(),
                         (seq_length, batch_size, directions * hidden_size))
        self.assertEqual(hy2.shape(), hx.shape)
        self.assertEqual(cy2.shape(), cx.shape)

        dy = tensor.Tensor(shape=(seq_length, batch_size,
                                  directions * hidden_size),
                           device=dev).gaussian(0, 1)
        dhy = tensor.Tensor(shape=(num_layers * directions, batch_size,
                                   hidden_size),
                            device=dev).gaussian(0, 1)
        dcy = tensor.Tensor(shape=(num_layers * directions, batch_size,
                                   hidden_size),
                            device=dev).gaussian(0, 1)

        (dx, dhx, dcx) = singa_api.GpuRNNBackwardx(y, dy.data, dhy.data,
                                                   dcy.data, w.data, hx.data,
                                                   cx.data, rnn_handle)
        self.assertEqual(dx.shape(), (seq_length, batch_size, feature_size))
        self.assertEqual(dhx.shape(), hx.shape)
        self.assertEqual(dcx.shape(), cx.shape)

        dW = singa_api.GpuRNNBackwardW(x.data, hx.data, y, rnn_handle)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_rnn_with_seq_lengths(self):
        dev = gpu_dev

        # params
        hidden_size = 7
        seq_length = 5
        batch_size = 6
        feature_size = 3
        directions = 2
        num_layers = 2

        # shapes
        x_s = (seq_length, batch_size, feature_size)
        y_s = (seq_length, batch_size, hidden_size)
        states_s = (num_layers * directions, batch_size, hidden_size)

        # tensors
        x = tensor.Tensor(x_s, dev).gaussian(0, 1)
        y = tensor.Tensor(y_s, dev).gaussian(0, 1)
        dy = tensor.Tensor(y_s, dev).gaussian(0, 1)
        dhy = tensor.Tensor(states_s, dev).gaussian(0, 1)
        dcy = tensor.Tensor(states_s, dev).gaussian(0, 1)
        hx = tensor.Tensor(states_s, dev).gaussian(0, 1)
        cx = tensor.Tensor(states_s, dev).gaussian(0, 1)

        # handle
        rnn_handle = singa_api.CudnnRNNHandle(x.data, hidden_size, 2)
        w = tensor.Tensor((rnn_handle.weights_size,), dev).gaussian(0, 1)

        # seq lengths
        seq_lengths = tensor.from_numpy(np.array([seq_length] * batch_size))

        # operations
        (dx, dhx, dcx) = singa_api.GpuRNNBackwardxEx(y.data, dy.data, dhy.data,
                                                     dcy.data, w.data, hx.data,
                                                     cx.data, seq_lengths.data,
                                                     rnn_handle)

    def test_round_cpu(self):
        self._round(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_round_gpu(self):
        self._round(gpu_dev)

    def _round(self, dev=gpu_dev):
        x = tensor.Tensor(shape=(3, 4, 5), device=dev).gaussian(0, 1)
        y = tensor._call_singa_func(singa_api.Round, x.data)
        np.testing.assert_array_almost_equal(np.round(tensor.to_numpy(x)),
                                             tensor.to_numpy(y))

    def test_round_even_cpu(self):
        self._round_even(cpu_dev)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_round_even_gpu(self):
        self._round_even(gpu_dev)

    def _round_even(self, dev=gpu_dev):
        q = np.array([
            0.1, 0.5, 0.9, 1.2, 1.5, 1.8, 2.3, 2.5, 2.7, -1.1, -1.5, -1.9, -2.2,
            -2.5, -2.8
        ]).astype(np.float32)
        ans = np.array(
            [0., 0., 1., 1., 2., 2., 2., 2., 3., -1., -2., -2., -2., -2.,
             -3.]).astype(np.float32)

        x = tensor.Tensor(shape=q.shape, device=dev)
        x.copy_from_numpy(q)
        y = tensor._call_singa_func(singa_api.RoundE, x.data)
        np.testing.assert_array_almost_equal(ans, tensor.to_numpy(y))


if __name__ == '__main__':
    unittest.main()
