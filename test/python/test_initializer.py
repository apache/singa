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
        self.t1 = tensor.Tensor((4, 9))
        self.t2 = tensor.Tensor((3, 5, 8))
        self.t3 = tensor.Tensor((3, 5, 4, 8))

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
            return np.random.uniform(-limit, limit, shape)

        self.t1.to_device(dev)
        initializer.he_uniform(self.t1)
        np.testing.assert_almost_equal(tensor.to_numpy(self.t1),
                                       init(self.t1.shape))

        self.t2.to_device(dev)
        initializer.he_uniform(self.t2)
        np.testing.assert_almost_equal(tensor.to_numpy(self.t2),
                                       init(self.t2.shape))

        self.t3.to_device(dev)
        initializer.he_uniform(self.t3)
        np.testing.assert_almost_equal(tensor.to_numpy(self.t3),
                                       init(self.t3.shape))

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_he_uniform_gpu(self):
        self.he_uniform(gpu_dev)

    def test_he_uniform_cpu(self):
        self.he_uniform(cpu_dev)

    def he_normal(self, dev):

        def init(shape):
            fan_in, _ = self.compute_fan(shape)
            stddev = np.sqrt(2 / fan_in)
            return np.random.normal(0, stddev, shape)

        self.t1.to_device(dev)
        initializer.he_normal(self.t1)
        np.testing.assert_almost_equal(tensor.to_numpy(self.t1),
                                       init(self.t1.shape))

        self.t2.to_device(dev)
        initializer.he_normal(self.t2)
        np.testing.assert_almost_equal(tensor.to_numpy(self.t2),
                                       init(self.t2.shape))

        self.t3.to_device(dev)
        initializer.he_normal(self.t3)
        np.testing.assert_almost_equal(tensor.to_numpy(self.t3),
                                       init(self.t3.shape))

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_he_normal_gpu(self):
        self.he_uniform(gpu_dev)

    def test_he_normal_cpu(self):
        self.he_uniform(cpu_dev)