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
from cuda_helper import gpu_dev, cpu_dev


class TestAPI(unittest.TestCase):
    def test_batchnorm_api(self):
        def run_batchnorm_inference(x0, scale, bias, mean, var):
            hndl = singa_api.CudnnBatchNormHandle(0, x0.data)
            return tensor._call_singa_func(
                singa_api.GpuBatchNormForwardInference, hndl, x0.data,
                scale.data, bias.data, mean.data, var.data)

        def _batchnorm_inference_value_check(x,
                                             s,
                                             bias,
                                             mean,
                                             var,
                                             epsilon=1e-5):
            dims_x = len(x.shape)
            dim_ones = (1, ) * (dims_x - 2)
            s = s.reshape(-1, *dim_ones)
            bias = bias.reshape(-1, *dim_ones)
            mean = mean.reshape(-1, *dim_ones)
            var = var.reshape(-1, *dim_ones)
            return s * (x - mean) / np.sqrt(var + epsilon) + bias

        # prepare data
        x_0 = np.random.random([2, 2, 2, 2]).astype(np.float32)
        param_0 = np.random.random([2]).astype(np.float32)

        # prepare singa py tensor
        x0 = tensor.Tensor(device=gpu_dev, data=x_0)
        scale = tensor.Tensor(device=gpu_dev, data=param_0)
        bias = tensor.Tensor(device=gpu_dev, data=param_0)
        mean = tensor.Tensor(device=gpu_dev, data=param_0)
        var = tensor.Tensor(device=gpu_dev, data=param_0)

        # singa api
        y0 = run_batchnorm_inference(x0, scale, bias, mean, var)
        # numpy api
        y_1 = _batchnorm_inference_value_check(x_0, param_0, param_0, param_0,
                                               param_0)

        np.testing.assert_array_almost_equal(tensor.to_numpy(y0),
                                             y_1,
                                             decimal=5)


if __name__ == '__main__':
    unittest.main()
