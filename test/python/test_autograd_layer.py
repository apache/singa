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
import logging
import sys

from singa import tensor
from singa import singa_wrap as singa_api
from singa import autograd
from singa.proto import core_pb2

from cuda_helper import gpu_dev, cpu_dev


class TestLayers(unittest.TestCase):
    def test_linear_force_in_features(self):
        x = tensor.Tensor((10,2)).gaussian(1, 2)
        W = tensor.Tensor((2,3)).gaussian(1, 2)
        b = tensor.Tensor((3,)).gaussian(1, 2)
        li1 = autograd.Linear(3,in_features=2)
        li1.set_params(W=W,b=b)
        (W,b) = li1.get_params()
        y=li1(x)

    def test_linear_auto_in_features_by_set_params(self):
        x = tensor.Tensor((10,2)).gaussian(1, 2)
        W = tensor.Tensor((2,3)).gaussian(1, 2)
        b = tensor.Tensor((3,)).gaussian(1, 2)
        li1 = autograd.Linear(3)
        # print(li1.__dict__)
        li1.set_params(W=W,b=b)
        (W,b) = li1.get_params()
        y=li1(x)

    def test_linear_auto_in_features_by_forward(self):
        x = tensor.Tensor((10,2)).gaussian(1, 2)
        W = tensor.Tensor((2,3)).gaussian(1, 2)
        b = tensor.Tensor((3,)).gaussian(1, 2)
        li1 = autograd.Linear(3)
        y=li1(x)
        (W,b) = li1.get_params()

# logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
if __name__ == '__main__':
    unittest.main()
