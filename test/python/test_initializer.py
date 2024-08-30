#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from singa import initializer
from singa import tensor
from singa import singa_wrap

from cuda_helper import gpu_dev, cpu_dev

import unittest
import numpy as np


class TestInitializer(unittest.TestCase):

    def setUp(self):
        self.t1 = tensor.Tensor((40, 90))
        self.t2 = tensor.Tensor((30, 50, 8))
        self.t3 = tensor.Tensor((30, 50, 4, 8))

    def compute_fan(self, shape):
        if len(shape) == 2:
            fan_in = shape[0]
            fan_out = shape[1]
        elif len(shape) in {3, 4, 5}:
            fan_in = shape[1] * np.prod(shape[2:])
            fan_out = shape[0] * np.prod(shape[2:])
        else:
            fan_in = fan_out = np.sqrt(np.prod(shape))

        return fan_in, fan_out

    def he_uniform(self, dev):

        def init(shape):
            fan_in, _ = self.compute_fan(shape)
            limit = np.sqrt(6 / fan_in)
            return limit

        self.t1.to_device(dev)
        initializer.he_uniform(self.t1)
        np_t1 = tensor.to_numpy(self.t1)
        limit = init(self.t1.shape)
        self.assertAlmostEqual(np_t1.max(), limit, delta=limit / 10)
        self.assertAlmostEqual(np_t1.min(), -limit, delta=limit / 10)
        self.assertAlmostEqual(np_t1.mean(), 0, delta=limit / 10)

        self.t2.to_device(dev)
        initializer.he_uniform(self.t2)
        np_t2 = tensor.to_numpy(self.t2)
        limit = init(self.t2.shape)
        self.assertAlmostEqual(np_t2.max(), limit, delta=limit / 10)
        self.assertAlmostEqual(np_t2.min(), -limit, delta=limit / 10)
        self.assertAlmostEqual(np_t2.mean(), 0, delta=limit / 10)

        self.t3.to_device(dev)
        initializer.he_uniform(self.t3)
        np_t3 = tensor.to_numpy(self.t3)
        limit = init(self.t3.shape)
        self.assertAlmostEqual(np_t3.max(), limit, delta=limit / 10)
        self.assertAlmostEqual(np_t3.min(), -limit, delta=limit / 10)
        self.assertAlmostEqual(np_t3.mean(), 0, delta=limit / 10)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_he_uniform_gpu(self):
        self.he_uniform(gpu_dev)

    def test_he_uniform_cpu(self):
        self.he_uniform(cpu_dev)

    def he_normal(self, dev):

        def init(shape):
            fan_in, _ = self.compute_fan(shape)
            stddev = np.sqrt(2 / fan_in)
            return stddev

        self.t1.to_device(dev)
        initializer.he_normal(self.t1)
        np_t1 = tensor.to_numpy(self.t1)
        stddev = init(self.t1.shape)
        self.assertAlmostEqual(np_t1.mean(), 0, delta=stddev / 10)
        self.assertAlmostEqual(np_t1.std(), stddev, delta=stddev / 10)

        self.t2.to_device(dev)
        initializer.he_normal(self.t2)
        np_t2 = tensor.to_numpy(self.t2)
        stddev = init(self.t2.shape)
        self.assertAlmostEqual(np_t2.mean(), 0, delta=stddev / 10)
        self.assertAlmostEqual(np_t2.std(), stddev, delta=stddev / 10)

        self.t3.to_device(dev)
        initializer.he_normal(self.t3)
        np_t3 = tensor.to_numpy(self.t3)
        stddev = init(self.t3.shape)
        self.assertAlmostEqual(np_t3.mean(), 0, delta=stddev / 10)
        self.assertAlmostEqual(np_t3.std(), stddev, delta=stddev / 10)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_he_normal_gpu(self):
        self.he_uniform(gpu_dev)

    def test_he_normal_cpu(self):
        self.he_uniform(cpu_dev)


if __name__ == '__main__':
    unittest.main()
