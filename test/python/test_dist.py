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
import numpy as np
from singa import tensor
from singa import opt
from singa import device
from singa import singa_wrap

if (singa_wrap.USE_DIST):
    sgd = opt.SGD(lr=0.1)
    sgd = opt.DistOpt(sgd)
    dev = device.create_cuda_gpu_on(sgd.local_rank)
    param = tensor.Tensor((10, 10), dev, tensor.float32)
    grad = tensor.Tensor((10, 10), dev, tensor.float32)
    expected = np.ones((10, 10), dtype=np.float32) * (10 - 0.1)


@unittest.skipIf(not singa_wrap.USE_DIST, 'DIST is not enabled')
class TestDistOptimizer(unittest.TestCase):

    def test_dist_opt_fp32(self):
        # Test the C++ all reduce operation in fp32

        param.set_value(10)
        grad.set_value(1)

        sgd.all_reduce(grad.data)
        sgd.wait()
        sgd.update(param, grad)

        np.testing.assert_array_almost_equal(tensor.to_numpy(param),
                                             expected,
                                             decimal=5)

    def test_dist_opt_fp32_fused(self):
        # Test the C++ all reduce operation in fp32

        param.set_value(10)
        grad.set_value(1)

        sgd.fused_all_reduce([grad.data], send=False)
        sgd.fused_all_reduce([grad.data])
        sgd.wait()
        sgd.update(param, grad)

        np.testing.assert_array_almost_equal(tensor.to_numpy(param),
                                             expected,
                                             decimal=5)

    def test_dist_opt_fp16(self):
        # Test the C++ all reduce operation in fp16

        param.set_value(10)
        grad.set_value(1)

        sgd.all_reduce_half(grad.data)
        sgd.wait()
        sgd.update(param, grad)

        np.testing.assert_array_almost_equal(tensor.to_numpy(param),
                                             expected,
                                             decimal=5)

    def test_dist_opt_fp16_fused(self):
        # Test the C++ all reduce operation in fp16

        param.set_value(10)
        grad.set_value(1)

        sgd.fused_all_reduce_half([grad.data], send=False)
        sgd.fused_all_reduce_half([grad.data])
        sgd.wait()
        sgd.update(param, grad)

        np.testing.assert_array_almost_equal(tensor.to_numpy(param),
                                             expected,
                                             decimal=5)

    def test_dist_opt_spars_value(self):
        # Test the C++ value based sparsification operation for all reduce

        param.set_value(10)
        grad.set_value(1)

        sgd.sparsification(grad.data, accumulation=None, spars=0.05, topK=False)
        sgd.wait()
        sgd.update(param, grad)

        np.testing.assert_array_almost_equal(tensor.to_numpy(param),
                                             expected,
                                             decimal=5)

    def test_dist_opt_spars_topk(self):
        # Test the C++ TopK based sparsification operation for all reduce

        param.set_value(10)
        grad.set_value(1)

        sgd.sparsification(grad.data, accumulation=None, spars=1, topK=True)
        sgd.wait()
        sgd.update(param, grad)

        np.testing.assert_array_almost_equal(tensor.to_numpy(param),
                                             expected,
                                             decimal=5)


if __name__ == '__main__':
    unittest.main()
