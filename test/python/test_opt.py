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
from singa import singa_wrap as singa
from singa import opt

from cuda_helper import gpu_dev, cpu_dev


class TestOpt(unittest.TestCase):
    def _const_decay_scheduler_helper(self, dev):
        c1 = opt.Constant(0.2)
        step = tensor.Tensor((1,), device=dev).set_value(0)
        lr_val = c1(step)
        np.testing.assert_array_almost_equal(tensor.to_numpy(c1(step)) , [0.2])
        step+=1
        np.testing.assert_array_almost_equal(tensor.to_numpy(c1(step)) , [0.2])

    @unittest.skipIf(not singa.USE_CUDA, 'CUDA is not enabled')
    def test_const_decay_scheduler_gpu(self):
        self._const_decay_scheduler_helper(gpu_dev)

    def test_const_decay_scheduler_cpu(self):
        self._const_decay_scheduler_helper(cpu_dev)

    def _optimizer_helper(self, dev):
        o1 = opt.Optimizer(0.1)

        # test step
        o1.step()
        o1.step()

        # test get states
        s1 = o1.get_states()
        self.assertAlmostEqual(s1['step_counter'], 2)

        # test set states
        s2 = {'step_counter': 5}
        o1.set_states(s2)
        np.testing.assert_array_almost_equal( tensor.to_numpy(o1.step_counter), [5])

    @unittest.skipIf(not singa.USE_CUDA, 'CUDA is not enabled')
    def test_optimizer_gpu(self):
        self._optimizer_helper(gpu_dev)

    def test_optimizer_cpu(self):
        self._optimizer_helper(cpu_dev)

    def _sgd_apply_helper(self, dev):
        sgd1 = opt.SGD(lr=0.01, momentum=0.9)

        w_shape=(2,3)
        w = tensor.Tensor(w_shape, device=dev).set_value(0.1)
        g = tensor.Tensor(w_shape, device=dev).set_value(0.01)
        sgd1.apply(w, g)
        sgd1.step()
        sgd1.apply(w, g)
        sgd1.step()

        _ = sgd1.get_states()
        sgd1.set_states(_)

    @unittest.skipIf(not singa.USE_CUDA, 'CUDA is not enabled')
    def test_sgd_apply_gpu(self):
        self._sgd_apply_helper(gpu_dev)

    def test_sgd_apply_cpu(self):
        self._sgd_apply_helper(cpu_dev)


if __name__ == '__main__':
    unittest.main()
