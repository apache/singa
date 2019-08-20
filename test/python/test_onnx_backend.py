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
from builtins import str

from singa import tensor
from singa import singa_wrap as singa
from singa import autograd
from singa import sonnx
from singa import opt

import onnx
from onnx import (defs, checker, helper, numpy_helper, mapping,
                  ModelProto, GraphProto, NodeProto, AttributeProto, TensorProto, OperatorSetIdProto)
from onnx.helper import make_tensor, make_tensor_value_info, make_node, make_graph

from cuda_helper import gpu_dev, cpu_dev

import numpy as np

autograd.training = True

def expect(onnx_node, inputs, outputs, name, opset_version=_default_opset_version):
    outputs_dict = sonnx.run_node(onnx_node, inputs, opset_version)
    for out1, out2 in zip(outputs, outputs_dict.values()):
        np.testing.assert_array_almost_equal(tensor.to_numpy(out1), out2, decimal=5)

class TestPythonOnnxBackend(unittest.TestCase):
    """
    This class aims to test the backend functionality of sonnx,
    The most of the code is borrowed from onnx.
    """
    _default_opset_version = 10

    def test_conv2d(self):
        x = np.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 5, 5) input tensor
                        [5., 6., 7., 8., 9.],
                        [10., 11., 12., 13., 14.],
                        [15., 16., 17., 18., 19.],
                        [20., 21., 22., 23., 24.]]]]).astype(np.float32)

        W = np.array([[[[1., 1., 1.],  # (1, 1, 3, 3) tensor for convolution weights
                        [1., 1., 1.],
                        [1., 1., 1.]]]]).astype(np.float32)

        # Convolution with padding
        node_with_padding = onnx.helper.make_node(
            'Conv',
            inputs=['x', 'W'],
            outputs=['y'],
            kernel_shape=[3, 3],
            # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
            pads=[1, 1, 1, 1],
        )

        y_with_padding = np.array([[[[12., 21., 27., 33., 24.],  # (1, 1, 5, 5) output tensor
                                     [33., 54., 63., 72., 51.],
                                     [63., 99., 108., 117., 81.],
                                     [93., 144., 153., 162., 111.],
                                     [72., 111., 117., 123., 84.]]]]).astype(np.float32)

        expect(node_with_padding, inputs=[x, W], outputs=[y_with_padding],
               name='test_basic_conv_with_padding')

        # Convolution without padding
        node_without_padding = onnx.helper.make_node(
            'Conv',
            inputs=['x', 'W'],
            outputs=['y'],
            kernel_shape=[3, 3],
            # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
            pads=[0, 0, 0, 0],
        )
        y_without_padding = np.array([[[[54., 63., 72.],  # (1, 1, 3, 3) output tensor
                                        [99., 108., 117.],
                                        [144., 153., 162.]]]]).astype(np.float32)
        expect(node_without_padding, inputs=[x, W], outputs=[y_without_padding],
               name='test_basic_conv_without_padding')

if __name__ == '__main__':
    unittest.main()
