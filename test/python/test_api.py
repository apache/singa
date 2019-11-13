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
            y1 = np.divide(np.exp(x_0), np.sum(np.exp(x_0),axis=1).reshape(x_0.shape[0],1) ) # 2d softmax
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

            z0 = tensor._call_singa_func(singa_op, x0.data,y0.data)
            z0.to_host()
            #z0 = x0*y0

            #print(s1,s2,tensor.to_numpy(z0).shape)
            np.testing.assert_array_almost_equal(tensor.to_numpy(z0), np_op(x_0,y_0) )
            return

        _run_test(singa_api.__add__,np.add,[6],[1])
        _run_test(singa_api.__add__,np.add,[3,2],[1])
        _run_test(singa_api.__add__,np.add,[3,1,2],[3,1,1])
        _run_test(singa_api.__add__,np.add,[2,3,4,5],[5])
        _run_test(singa_api.__add__,np.add,[2,3,4,5],[1,1,1])
        _run_test(singa_api.__add__,np.add,[2,3,4,5],[1,1,1,1])
        _run_test(singa_api.__add__,np.add,[3,1,2,1],[3,1,2])
        _run_test(singa_api.__add__,np.add,[4,5],[2,3,4,5]) # 45+2345=2345
        _run_test(singa_api.__add__,np.add,[2,3,4,5],[4,5]) # 45+2345=2345
        _run_test(singa_api.__add__,np.add,[1,4,5],[2,3,1,1]) # 145+2311=2345
        _run_test(singa_api.__add__,np.add,[3,4,5],[2,1,1,1]) # 345+2111=2345

        _run_test(singa_api.__mul__,np.multiply,[2,3],[2,3])
        _run_test(singa_api.__mul__,np.multiply,[6],[1])
        _run_test(singa_api.__mul__,np.multiply,[3,2],[1])
        _run_test(singa_api.__mul__,np.multiply,[3,1,2],[3,1,1])
        _run_test(singa_api.__mul__,np.multiply,[2,3,4,5],[5])
        _run_test(singa_api.__mul__,np.multiply,[2,3,4,5],[1,1,1])
        _run_test(singa_api.__mul__,np.multiply,[2,3,4,5],[1,1,1,1])
        _run_test(singa_api.__mul__,np.multiply,[3,1,2,1],[3,1,2])
        _run_test(singa_api.__mul__,np.multiply,[4,5],[2,3,4,5]) # 45+2345=2345
        _run_test(singa_api.__mul__,np.multiply,[2,3,4,5],[4,5]) # 45+2345=2345
        _run_test(singa_api.__mul__,np.multiply,[1,4,5],[2,3,1,1]) # 145+2311=2345
        _run_test(singa_api.__mul__,np.multiply,[3,4,5],[2,1,1,1]) # 345+2111=2345

    def test_transpose_and_mul(self):
        s1 = [3,2,1,1]
        s2 = [3,2,1,1]
        x_0 = np.random.random(s1).astype(np.float32)
        y_0 = np.random.random(s2).astype(np.float32)

        x0 = tensor.Tensor(device=gpu_dev, data=x_0)
        y0 = tensor.Tensor(device=gpu_dev, data=y_0)
        x1 = x0.transpose([3,2,1,0])

        #print(x1.shape)
        #print(y0.shape)

        z0 = x1 * y0
        np.testing.assert_array_almost_equal(tensor.to_numpy(z0), x_0.transpose()*y_0 )

    def test_transpose_and_mul_2(self):
        s1 = [1,5,1,3]
        s2 = [3,1,1,4]
        axis0=[3,2,1,0] # 3121
        axis1=[1,0,2,3] # 1314
        x_0 = np.random.random(s1).astype(np.float32)
        y_0 = np.random.random(s2).astype(np.float32)

        x0 = tensor.Tensor(device=gpu_dev, data=x_0)
        y0 = tensor.Tensor(device=gpu_dev, data=y_0)

        x1 = x0.transpose(axis0)
        y1 = y0.transpose(axis1)
        #print(x1.shape)
        #print(y1.shape)

        z0 = x1 * y1
        np.testing.assert_array_almost_equal(tensor.to_numpy(z0), x_0.transpose(axis0)*y_0.transpose(axis1) )
        np.testing.assert_array_almost_equal(z0.shape, [3,3,5,4])




if __name__ == '__main__':
    unittest.main()
