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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from singa import tensor
from singa import singa_wrap as singa
from singa import autograd
from singa import sonnx
from singa import opt

import os

import unittest
import onnx.backend.test

# This is a pytest magic variable to load extra plugins
pytest_plugins = 'onnx.backend.test.report',

backend_test = onnx.backend.test.BackendTest(sonnx.SingaBackend, __name__)

_include_nodes_patterns = {
    # rename some patterns
    'ReduceSum': r'(test_reduce_sum)',
    'ReduceMean': r'(test_reduce_mean)',
    'BatchNormalization': r'(test_batchnorm)',
    'ScatterElements': r'(test_scatter_elements)',
    'Conv': r'(test_basic_conv_|test_conv_with_|test_Conv2d)',
    'MaxPool': r'(test_maxpool_2d)',
    'AveragePool': r'(test_averagepool_2d)',
}

_exclude_nodes_patterns = [
    # not support data type
    r'(uint)',  # does not support uint
    r'(scalar)',  # does not support scalar
    r'(STRING)',  # does not support string
    # not support some features
    r'(test_split_zero_size_splits|test_slice_start_out_of_bounds)',  # not support empty tensor
    r'(test_batchnorm_epsilon)',  # does not support epsilon
    r'(dilations)',  # does not support dilations
    r'(test_maxpool_2d_ceil|test_averagepool_2d_ceil)',  # does not ceil for max or avg pool
    r'(count_include_pad)',  # pool not support count_include_pad
    # interrupt some include patterns
    r'(test_matmulinteger)',  # interrupt matmulinteger
    r'(test_less_equal)',  # interrupt les
    r'(test_greater_equal)',  # interrupt greater
    r'(test_negative_log)',  # interrupt negative
    r'(test_softmax_cross_entropy)',  # interrupt softmax
    r'(test_reduce_sum_square)',  # interrupt reduce sum squre
    r'(test_log_softmax)',  # interrupt log softmax
    r'(test_maxunpool)',  # interrupt max unpool
    r'(test_gather_elements)',  # interrupt gather elements
    r'(test_logsoftmax)',  # interrupt log softmax
    r'(test_gathernd)',  # interrupt gather nd
    r'(test_maxpool_with_argmax)',  # interrupt maxpool_with_argmax
    # todo, some special error
    r'test_transpose',  # the test cases are wrong
    r'test_conv_with_strides_and_asymmetric_padding',  # the test cases are wrong
    r'(test_gemm_default_single_elem_vector_bias_cuda)',  # status == CURAND_STATUS_SUCCESS
    r'(test_equal_bcast_cuda|test_equal_cuda)',  # Unknown combination of data type kInt and language kCuda
    r'(test_maxpool_1d|test_averagepool_1d|test_maxpool_3d|test_averagepool_3d)',  # Check failed: idx < shape_.size() (3 vs. 3)
    r'test_depthtospace.*cuda',  # cuda cannot support transpose with more than 4 dims
]

_include_real_patterns = []  # todo

_include_simple_patterns = []  # todo

_include_pytorch_converted_patterns = []  # todo

_include_pytorch_operator_patterns = []  # todo

# add supported operators into include patterns
for name in sonnx.SingaBackend._rename_operators.keys():
    if name not in _include_nodes_patterns:
        backend_test.include(r'(test_{})'.format(name.lower()))
    else:
        # todo, need to fix the conv2d
        if name == 'Conv':
            continue
        backend_test.include(_include_nodes_patterns[name])

# exclude the unsupported operators
for pattern in _exclude_nodes_patterns:
    backend_test.exclude(pattern)

# exclude the cuda cases
if not singa.USE_CUDA:
    backend_test.exclude(r'(cuda)')

OnnxBackendNodeModelTest = backend_test.enable_report(
).test_cases['OnnxBackendNodeModelTest']


# disable and enable training before and after test cases
def setUp(self):
    # print("\nIn method", self._testMethodName)
    autograd.training = False


def tearDown(self):
    autograd.training = True


OnnxBackendNodeModelTest.setUp = setUp
OnnxBackendNodeModelTest.tearDown = tearDown

# import all test cases at global scope to make them visible to python.unittest
# print(backend_test.enable_report().test_cases)
test_cases = {'OnnxBackendNodeModelTest': OnnxBackendNodeModelTest}

globals().update(test_cases)

if __name__ == '__main__':
    unittest.main()
