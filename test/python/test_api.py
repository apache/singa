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
    # this tests cpp softmax api directly from SWIG wrapper
    def test_softmax_api(self):
        def run_test(org_shape, axis, aft_shape):
            x_0 = np.random.random(org_shape).astype(np.float32)
            x0 = tensor.Tensor(device=gpu_dev, data=x_0)

            # test with axis
            y0 = tensor._call_singa_func(singa_api.SoftMax, x0.data, axis)

            # test with manual reshape
            y1 = tensor.softmax(x0.reshape(aft_shape))
            y1 = y1.reshape(org_shape)

            np.testing.assert_array_almost_equal(tensor.to_numpy(y0),
                                                 tensor.to_numpy(y1),
                                                 decimal=5)

        run_test([2, 2], 1, [2, 2])
        run_test([2, 2], 0, [1, 4])
        run_test([2, 2], -1, [2, 2])
        run_test([2, 2], -2, [1, 4])

        run_test([2, 2, 2], 2, [4, 2])
        run_test([2, 2, 2], 1, [2, 4])
        run_test([2, 2, 2], 0, [1, 8])
        run_test([2, 2, 2], -1, [4, 2])
        run_test([2, 2, 2], -2, [2, 4])
        run_test([2, 2, 2], -3, [1, 8])

        run_test([2, 2, 2, 2], 3, [8, 2])
        run_test([2, 2, 2, 2], 2, [4, 4])
        run_test([2, 2, 2, 2], 1, [2, 8])
        run_test([2, 2, 2, 2], 0, [1, 16])
        run_test([2, 2, 2, 2], -1, [8, 2])
        run_test([2, 2, 2, 2], -2, [4, 4])
        run_test([2, 2, 2, 2], -3, [2, 8])
        run_test([2, 2, 2, 2], -4, [1, 16])


if __name__ == '__main__':
    unittest.main()
