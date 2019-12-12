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
import itertools

autograd.training = True

_default_opset_version = 10


def expect(node, inputs, outputs, name, opset_version=_default_opset_version):
    onnx_node = sonnx.OnnxNode(node)
    input_tensors = {}
    input_labels = [x for x in onnx_node.inputs if x != ""]
    # prepare input tensors
    for key, val in zip(input_labels, inputs):
        # very important! must be float
        if not isinstance(val, np.ndarray) or len(val.shape) == 0:
            val = np.array([val])
        x = tensor.from_numpy(val.astype(np.float32))
        x.to_device(gpu_dev)
        input_tensors[key] = x
    outputs_dict = sonnx.run_node(onnx_node, input_tensors, opset_version)
    for out1, out2 in zip(outputs, outputs_dict.values()):
        np.testing.assert_array_almost_equal(
            out1, tensor.to_numpy(out2), decimal=5)


class TestPythonOnnxBackend(unittest.TestCase):
    """
    This class aims to test the backend functionality of sonnx,
    The most of the code is borrowed from onnx.
    """

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

    def test_conv2d_with_strides(self):  # type: () -> None

        x = np.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 7, 5) input tensor
                        [5., 6., 7., 8., 9.],
                        [10., 11., 12., 13., 14.],
                        [15., 16., 17., 18., 19.],
                        [20., 21., 22., 23., 24.],
                        [25., 26., 27., 28., 29.],
                        [30., 31., 32., 33., 34.]]]]).astype(np.float32)
        W = np.array([[[[1., 1., 1.],  # (1, 1, 3, 3) tensor for convolution weights
                        [1., 1., 1.],
                        [1., 1., 1.]]]]).astype(np.float32)

        # Convolution with strides=2 and padding
        node_with_padding = onnx.helper.make_node(
            'Conv',
            inputs=['x', 'W'],
            outputs=['y'],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            # Default values for other attributes: dilations=[1, 1], groups=1
            strides=[2, 2],
        )
        y_with_padding = np.array([[[[12., 27., 24.],  # (1, 1, 4, 3) output tensor
                                     [63., 108., 81.],
                                     [123., 198., 141.],
                                     [112., 177., 124.]]]]).astype(np.float32)
        expect(node_with_padding, inputs=[x, W], outputs=[y_with_padding],
               name='test_conv_with_strides_padding')

        # Convolution with strides=2 and no padding
        node_without_padding = onnx.helper.make_node(
            'Conv',
            inputs=['x', 'W'],
            outputs=['y'],
            kernel_shape=[3, 3],
            pads=[0, 0, 0, 0],
            # Default values for other attributes: dilations=[1, 1], groups=1
            strides=[2, 2],
        )
        y_without_padding = np.array([[[[54., 72.],  # (1, 1, 3, 2) output tensor
                                        [144., 162.],
                                        [234., 252.]]]]).astype(np.float32)
        expect(node_without_padding, inputs=[x, W], outputs=[y_without_padding],
               name='test_conv_with_strides_no_padding')

        # Convolution with strides=2 and padding only along one dimension (the H dimension in NxCxHxW tensor)
        node_with_asymmetric_padding = onnx.helper.make_node(
            'Conv',
            inputs=['x', 'W'],
            outputs=['y'],
            kernel_shape=[3, 3],
            pads=[1, 0, 1, 0],
            # Default values for other attributes: dilations=[1, 1], groups=1
            strides=[2, 2],
        )
        y_with_asymmetric_padding = np.array([[[[21., 33.],  # (1, 1, 4, 2) output tensor
                                                [99., 117.],
                                                [189., 207.],
                                                [171., 183.]]]]).astype(np.float32)
        expect(node_with_asymmetric_padding, inputs=[x, W], outputs=[y_with_asymmetric_padding],
               name='test_conv_with_strides_and_asymmetric_padding')

    def test_averagepool_2d_precomputed_pads(self):  # type: () -> None
        """
        input_shape: [1, 1, 5, 5]
        output_shape: [1, 1, 5, 5]
        pad_shape: [4, 4] -> [2, 2, 2, 2] by axis
        """
        node = onnx.helper.make_node(
            'AveragePool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[5, 5],
            pads=[2, 2, 2, 2]

        )
        x = np.array([[[
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ]]]).astype(np.float32)
        y = np.array([[[[7, 7.5, 8, 8.5, 9],
                        [9.5, 10, 10.5, 11, 11.5],
                        [12, 12.5, 13, 13.5, 14],
                        [14.5, 15, 15.5, 16, 16.5],
                        [17, 17.5, 18, 18.5, 19]]]]).astype(np.float32)

        expect(node, inputs=[x], outputs=[y],
               name='test_averagepool_2d_precomputed_pads')

    def test_averagepool_2d_precomputed_strides(self):  # type: () -> None
        """
        input_shape: [1, 1, 5, 5]
        output_shape: [1, 1, 2, 2]
        """
        node = onnx.helper.make_node(
            'AveragePool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[2, 2],
            strides=[2, 2]
        )
        x = np.array([[[
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ]]]).astype(np.float32)
        y = np.array([[[[4, 6],
                        [14, 16]]]]).astype(np.float32)

        expect(node, inputs=[x], outputs=[y],
               name='test_averagepool_2d_precomputed_strides')

    def test_averagepool_2d_precomputed_same_upper(self):  # type: () -> None
        """
        input_shape: [1, 1, 5, 5]
        output_shape: [1, 1, 3, 3]
        pad_shape: [2, 2] -> [1, 1, 1, 1] by axis
        """
        node = onnx.helper.make_node(
            'AveragePool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[3, 3],
            strides=[2, 2],
            auto_pad='SAME_UPPER'
        )
        x = np.array([[[
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ]]]).astype(np.float32)
        y = np.array([[[[4, 5.5, 7],
                        [11.5, 13, 14.5],
                        [19, 20.5, 22]]]]).astype(np.float32)

        expect(node, inputs=[x], outputs=[y],
               name='test_averagepool_2d_precomputed_same_upper')

    def test_averagepool_2d_default(self):  # type: () -> None
        """
        input_shape: [1, 3, 32, 32]
        output_shape: [1, 3, 31, 31]
        """
        node = onnx.helper.make_node(
            'AveragePool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[2, 2],
        )
        x = np.random.randn(1, 3, 32, 32).astype(np.float32)
        x_shape = np.shape(x)
        kernel_shape = (2, 2)
        strides = (1, 1)
        out_shape = get_output_shape(
            'VALID', x_shape[2:], kernel_shape, strides)
        padded = x
        y = pool(padded, x_shape, kernel_shape,
                 strides, out_shape, (0, 0), 'AVG')

        expect(node, inputs=[x], outputs=[y],
               name='test_averagepool_2d_default')

    def test_averagepool_2d_pads(self):  # type: () -> None
        """
        input_shape: [1, 3, 28, 28]
        output_shape: [1, 3, 30, 30]
        pad_shape: [4, 4] -> [2, 2, 2, 2] by axis
        """
        node = onnx.helper.make_node(
            'AveragePool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[3, 3],
            pads=[2, 2, 2, 2]
        )
        x = np.random.randn(1, 3, 28, 28).astype(np.float32)
        x_shape = np.shape(x)
        kernel_shape = (3, 3)
        strides = (1, 1)
        pad_bottom = 2
        pad_top = 2
        pad_right = 2
        pad_left = 2
        pad_shape = [pad_top + pad_bottom, pad_left + pad_right]
        out_shape = get_output_shape('VALID', np.add(
            x_shape[2:], pad_shape), kernel_shape, strides)
        padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                        constant_values=np.nan)
        y = pool(padded, x_shape, kernel_shape,
                 strides, out_shape, pad_shape, 'AVG')

        expect(node, inputs=[x], outputs=[y], name='test_averagepool_2d_pads')

    def test_averagepool_2d_strides(self):  # type: () -> None
        """
        input_shape: [1, 3, 32, 32]
        output_shape: [1, 3, 10, 10]
        """
        node = onnx.helper.make_node(
            'AveragePool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[5, 5],
            strides=[3, 3]
        )
        x = np.random.randn(1, 3, 32, 32).astype(np.float32)
        x_shape = np.shape(x)
        kernel_shape = (5, 5)
        strides = (3, 3)
        out_shape = get_output_shape(
            'VALID', x_shape[2:], kernel_shape, strides)
        padded = x
        y = pool(padded, x_shape, kernel_shape,
                 strides, out_shape, (0, 0), 'AVG')

        expect(node, inputs=[x], outputs=[y],
               name='test_averagepool_2d_strides')

    def test_maxpool_2d_precomputed_pads(self):  # type: () -> None
        """
        input_shape: [1, 1, 5, 5]
        output_shape: [1, 1, 5, 5]
        pad_shape: [4, 4] -> [2, 2, 2, 2] by axis
        """
        node = onnx.helper.make_node(
            'MaxPool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[5, 5],
            pads=[2, 2, 2, 2]

        )
        x = np.array([[[
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ]]]).astype(np.float32)
        y = np.array([[[
            [13, 14, 15, 15, 15],
            [18, 19, 20, 20, 20],
            [23, 24, 25, 25, 25],
            [23, 24, 25, 25, 25],
            [23, 24, 25, 25, 25]]]]).astype(np.float32)

        expect(node, inputs=[x], outputs=[y],
               name='test_maxpool_2d_precomputed_pads')

    def test_maxpool_with_argmax_2d_precomputed_pads(self):  # type: () -> None
        """
        input_shape: [1, 1, 5, 5]
        output_shape: [1, 1, 5, 5]
        pad_shape: [4, 4] -> [2, 2, 2, 2] by axis
        """
        node = onnx.helper.make_node(
            'MaxPool',
            inputs=['x'],
            outputs=['y', 'z'],
            kernel_shape=[5, 5],
            pads=[2, 2, 2, 2]
        )
        x = np.array([[[
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ]]]).astype(np.float32)
        y = np.array([[[
            [13, 14, 15, 15, 15],
            [18, 19, 20, 20, 20],
            [23, 24, 25, 25, 25],
            [23, 24, 25, 25, 25],
            [23, 24, 25, 25, 25]]]]).astype(np.float32)
        z = np.array([[[
            [12, 13, 14, 14, 14],
            [17, 18, 19, 19, 19],
            [22, 23, 24, 24, 24],
            [22, 23, 24, 24, 24],
            [22, 23, 24, 24, 24]]]]).astype(np.int64)

        expect(node, inputs=[x], outputs=[y, z],
               name='test_maxpool_with_argmax_2d_precomputed_pads')

    def test_maxpool_2d_precomputed_strides(self):  # type: () -> None
        """
        input_shape: [1, 1, 5, 5]
        output_shape: [1, 1, 2, 2]
        """
        node = onnx.helper.make_node(
            'MaxPool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[2, 2],
            strides=[2, 2]
        )
        x = np.array([[[
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ]]]).astype(np.float32)
        y = np.array([[[[7, 9],
                        [17, 19]]]]).astype(np.float32)

        expect(node, inputs=[x], outputs=[y],
               name='test_maxpool_2d_precomputed_strides')

    # type: () -> None
    def test_maxpool_with_argmax_2d_precomputed_strides(self):
        """
        input_shape: [1, 1, 5, 5]
        output_shape: [1, 1, 2, 2]
        """
        node = onnx.helper.make_node(
            'MaxPool',
            inputs=['x'],
            outputs=['y', 'z'],
            kernel_shape=[2, 2],
            strides=[2, 2],
            storage_order=1
        )
        x = np.array([[[
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ]]]).astype(np.float32)
        y = np.array([[[[7, 9],
                        [17, 19]]]]).astype(np.float32)
        z = np.array([[[[6, 16],
                        [8, 18]]]]).astype(np.int64)

        expect(node, inputs=[x], outputs=[
               y, z], name='test_maxpool_with_argmax_2d_precomputed_strides')

    def test_maxpool_2d_precomputed_same_upper(self):  # type: () -> None
        """
        input_shape: [1, 1, 5, 5]
        output_shape: [1, 1, 3, 3]
        pad_shape: [2, 2] -> [1, 1, 1, 1] by axis
        """
        node = onnx.helper.make_node(
            'MaxPool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[3, 3],
            strides=[2, 2],
            auto_pad='SAME_UPPER'
        )
        x = np.array([[[
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ]]]).astype(np.float32)
        y = np.array([[[[7, 9, 10],
                        [17, 19, 20],
                        [22, 24, 25]]]]).astype(np.float32)

        expect(node, inputs=[x], outputs=[y],
               name='test_maxpool_2d_precomputed_same_upper')

    def test_maxpool_2d_default(self):  # type: () -> None
        """
        input_shape: [1, 3, 32, 32]
        output_shape: [1, 3, 31, 31]
        """
        node = onnx.helper.make_node(
            'MaxPool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[2, 2],
        )
        x = np.random.randn(1, 3, 32, 32).astype(np.float32)
        x_shape = np.shape(x)
        kernel_shape = (2, 2)
        strides = (1, 1)
        out_shape = get_output_shape(
            'VALID', x_shape[2:], kernel_shape, strides)
        padded = x
        y = pool(padded, x_shape, kernel_shape,
                 strides, out_shape, (0, 0), 'MAX')

        expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_default')

    def test_maxpool_2d_pads(self):  # type: () -> None
        """
        input_shape: [1, 3, 28, 28]
        output_shape: [1, 3, 30, 30]
        pad_shape: [4, 4] -> [2, 2, 2, 2] by axis
        """
        node = onnx.helper.make_node(
            'MaxPool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[3, 3],
            pads=[2, 2, 2, 2]
        )
        x = np.random.randn(1, 3, 28, 28).astype(np.float32)
        x_shape = np.shape(x)
        kernel_shape = (3, 3)
        strides = (1, 1)
        pad_bottom = pad_top = pad_right = pad_left = 2
        pad_shape = [pad_top + pad_bottom, pad_left + pad_right]
        out_shape = get_output_shape('VALID', np.add(
            x_shape[2:], pad_shape), kernel_shape, strides)
        padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                        constant_values=np.nan)
        y = pool(padded, x_shape, kernel_shape,
                 strides, out_shape, pad_shape, 'MAX')

        expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_pads')

    def test_maxpool_2d_strides(self):  # type: () -> None
        """
        input_shape: [1, 3, 32, 32]
        output_shape: [1, 3, 10, 10]
        """
        node = onnx.helper.make_node(
            'MaxPool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[5, 5],
            strides=[3, 3]
        )
        x = np.random.randn(1, 3, 32, 32).astype(np.float32)
        x_shape = np.shape(x)
        kernel_shape = (5, 5)
        strides = (3, 3)
        out_shape = get_output_shape(
            'VALID', x_shape[2:], kernel_shape, strides)
        padded = x
        y = pool(padded, x_shape, kernel_shape,
                 strides, out_shape, (0, 0), 'MAX')

        expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_strides')

    def test_reshape(self):  # type: () -> None

        def reshape_reference_implementation(data, shape):  # type: (np.ndarray, np.ndarray) -> np.ndarray
            # replace zeros with corresponding dim size
            # we need to do this because np.reshape doesn't support 0
            new_shape = np.copy(shape)
            zeros_index = np.where(shape == 0)
            new_shape[zeros_index] = np.array(data.shape)[zeros_index]
            reshaped = np.reshape(data, new_shape)
            return reshaped

        original_shape = [2, 3, 4]
        test_cases = {
            'reordered_all_dims': np.array([4, 2, 3], dtype=np.int64),
            'reordered_last_dims': np.array([2, 4, 3], dtype=np.int64),
            'reduced_dims': np.array([2, 12], dtype=np.int64),
            'extended_dims': np.array([2, 3, 2, 2], dtype=np.int64),
            'one_dim': np.array([24], dtype=np.int64),
            'negative_dim': np.array([2, -1, 2], dtype=np.int64),
            'negative_extended_dims': np.array([-1, 2, 3, 4], dtype=np.int64),
            'zero_dim': np.array([2, 0, 4, 1], dtype=np.int64),
            'zero_and_negative_dim': np.array([2, 0, 1, -1], dtype=np.int64),
        }
        data = np.random.random_sample(original_shape).astype(np.float32)

        for test_name, shape in test_cases.items():
            node = onnx.helper.make_node(
                'Reshape',
                inputs=['data', 'shape'],
                outputs=['reshaped'],
            )

            reshaped = reshape_reference_implementation(data, shape)

            expect(node, inputs=[data, shape], outputs=[reshaped],
                   name='test_reshape_' + test_name)

    def test_concat(self):  # type: () -> None
        test_cases = {
            # '1d': ([1, 2], not support 1d
            #    [3, 4]),
            '2d': ([[1, 2], [3, 4]],
                   [[5, 6], [7, 8]]),
            '3d': ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                   [[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
        }  # type: Dict[Text, Sequence[Any]]

        for test_case, values_ in test_cases.items():
            values = [np.asarray(v, dtype=np.float32) for v in values_]
            for i in range(len(values[0].shape)):
                in_args = ['value' + str(k) for k in range(len(values))]
                node = onnx.helper.make_node(
                    'Concat',
                    inputs=[s for s in in_args],
                    outputs=['output'],
                    axis=i
                )
                output = np.concatenate(values, i)
                expect(node, inputs=[v for v in values], outputs=[output],
                       name='test_concat_' + test_case + '_axis_' + str(i))

            for i in range(-len(values[0].shape), 0):
                in_args = ['value' + str(k) for k in range(len(values))]
                node = onnx.helper.make_node(
                    'Concat',
                    inputs=[s for s in in_args],
                    outputs=['output'],
                    axis=i
                )
                output = np.concatenate(values, i)
                expect(node, inputs=[v for v in values], outputs=[output],
                       name='test_concat_' + test_case + '_axis_negative_' + str(abs(i)))

    def test_flatten(self):  # type: () -> None
        shape = (2, 3, 4, 5)
        a = np.random.random_sample(shape).astype(np.float32)

        for i in range(len(shape)):
            node = onnx.helper.make_node(
                'Flatten',
                inputs=['a'],
                outputs=['b'],
                axis=i,
            )

            new_shape = (
                1, -1) if i == 0 else (np.prod(shape[0:i]).astype(int), -1)
            b = np.reshape(a, new_shape)
            expect(node, inputs=[a], outputs=[b],
                   name='test_flatten_axis' + str(i))

    def test_flatten_with_default_axis(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Flatten',
            inputs=['a'],
            outputs=['b'],  # Default value for axis: axis=1
        )

        shape = (5, 4, 3, 2)
        a = np.random.random_sample(shape).astype(np.float32)
        new_shape = (5, 24)
        b = np.reshape(a, new_shape)
        expect(node, inputs=[a], outputs=[b],
               name='test_flatten_default_axis')

    def test_flatten_negative_axis(self):  # type: () -> None
        shape = (2, 3, 4, 5)
        a = np.random.random_sample(shape).astype(np.float32)

        for i in range(-len(shape), 0):
            node = onnx.helper.make_node(
                'Flatten',
                inputs=['a'],
                outputs=['b'],
                axis=i,
            )

            new_shape = (np.prod(shape[0:i]).astype(int), -1)
            b = np.reshape(a, new_shape)
            expect(node, inputs=[a], outputs=[b],
                   name='test_flatten_negative_axis' + str(abs(i)))

    def test_add(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Add',
            inputs=['x', 'y'],
            outputs=['sum'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(3, 4, 5).astype(np.float32)
        expect(node, inputs=[x, y], outputs=[x + y],
               name='test_add')

    def test_add_broadcast(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Add',
            inputs=['x', 'y'],
            outputs=['sum'],
        )

        # todo, we don't support 3d here
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(5).astype(np.float32)
        expect(node, inputs=[x, y], outputs=[x + y],
               name='test_add_bcast')

    def test_sum(self):  # type: () -> None
        data_0 = np.array([3, 0, 2]).astype(np.float32)
        data_1 = np.array([1, 3, 4]).astype(np.float32)
        data_2 = np.array([2, 6, 6]).astype(np.float32)
        result = np.array([6, 9, 12]).astype(np.float32)
        node = onnx.helper.make_node(
            'Sum',
            inputs=['data_0', 'data_1', 'data_2'],
            outputs=['result'],
        )
        expect(node, inputs=[data_0, data_1, data_2], outputs=[result],
               name='test_sum_example')

        node = onnx.helper.make_node(
            'Sum',
            inputs=['data_0'],
            outputs=['result'],
        )
        expect(node, inputs=[data_0], outputs=[data_0],
               name='test_sum_one_input')

        result = np.add(data_0, data_1)
        node = onnx.helper.make_node(
            'Sum',
            inputs=['data_0', 'data_1'],
            outputs=['result'],
        )
        expect(node, inputs=[data_0, data_1], outputs=[result],
               name='test_sum_two_inputs')

    def test_relu(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Relu',
            inputs=['x'],
            outputs=['y'],
        )
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.clip(x, 0, np.inf)

        expect(node, inputs=[x], outputs=[y],
               name='test_relu')

    def test_sigmoid(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Sigmoid',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        # expected output [0.26894143, 0.5, 0.7310586]
        y = 1.0 / (1.0 + np.exp(np.negative(x)))
        expect(node, inputs=[x], outputs=[y],
               name='test_sigmoid_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = 1.0 / (1.0 + np.exp(np.negative(x)))
        expect(node, inputs=[x], outputs=[y],
               name='test_sigmoid')

    def test_matmul(self):  # type: () -> None
        node = onnx.helper.make_node(
            'MatMul',
            inputs=['a', 'b'],
            outputs=['c'],
        )

        # 2d
        a = np.random.randn(3, 4).astype(np.float32)
        b = np.random.randn(4, 3).astype(np.float32)
        c = np.matmul(a, b)
        expect(node, inputs=[a, b], outputs=[c],
               name='test_matmul_2d')

        # todo, # 3d not support 3d
        # a = np.random.randn(2, 3, 4).astype(np.float32)
        # b = np.random.randn(2, 4, 3).astype(np.float32)
        # c = np.matmul(a, b)
        # expect(node, inputs=[a, b], outputs=[c],
        #        name='test_matmul_3d')

        # todo, # 4d not support 4d
        # a = np.random.randn(1, 2, 3, 4).astype(np.float32)
        # b = np.random.randn(1, 2, 4, 3).astype(np.float32)
        # c = np.matmul(a, b)
        # expect(node, inputs=[a, b], outputs=[c],
        #        name='test_matmul_4d')

    def test_cos(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Cos',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        y = np.cos(x)
        expect(node, inputs=[x], outputs=[y],
               name='test_cos_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.cos(x)
        expect(node, inputs=[x], outputs=[y],
               name='test_cos')

    def test_cosh(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Cosh',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        y = np.cosh(x)  # expected output [1.54308069,  1.,  1.54308069]
        expect(node, inputs=[x], outputs=[y],
               name='test_cosh_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.cosh(x)
        expect(node, inputs=[x], outputs=[y],
               name='test_cosh')

    def test_Sin(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Sin',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        y = np.sin(x)
        expect(node, inputs=[x], outputs=[y],
               name='test_sin_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.sin(x)
        expect(node, inputs=[x], outputs=[y],
               name='test_sin')

    def test_Sinh(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Sinh',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        y = np.sinh(x)  # expected output [-1.17520118,  0.,  1.17520118]
        expect(node, inputs=[x], outputs=[y],
               name='test_sinh_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.sinh(x)
        expect(node, inputs=[x], outputs=[y],
               name='test_sinh')

    def test_Tan(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Tan',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        y = np.tan(x)
        expect(node, inputs=[x], outputs=[y],
               name='test_tan_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.tan(x)
        expect(node, inputs=[x], outputs=[y],
               name='test_tan')

    def test_Tanh(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Tanh',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        y = np.tanh(x)  # expected output [-0.76159418, 0., 0.76159418]
        expect(node, inputs=[x], outputs=[y],
               name='test_tanh_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.tanh(x)
        expect(node, inputs=[x], outputs=[y],
               name='test_tanh')

    def test_Acos(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Acos',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        y = np.arccos(x)
        expect(node, inputs=[x], outputs=[y],
               name='test_acos_example')

        x = np.random.rand(3, 4, 5).astype(np.float32)
        y = np.arccos(x)
        expect(node, inputs=[x], outputs=[y],
               name='test_acos')

    def test_Acosh(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Acosh',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([10, np.e, 1]).astype(np.float32)
        y = np.arccosh(x)  # expected output [2.99322295,  1.65745449,  0.]
        expect(node, inputs=[x], outputs=[y],
               name='test_acosh_example')

        x = np.random.uniform(1.0, 10.0, (3, 4, 5)).astype(np.float32)
        y = np.arccosh(x)
        expect(node, inputs=[x], outputs=[y],
               name='test_acosh')

    def test_Asin(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Asin',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        y = np.arcsin(x)
        expect(node, inputs=[x], outputs=[y],
               name='test_asin_example')

        x = np.random.rand(3, 4, 5).astype(np.float32)
        y = np.arcsin(x)
        expect(node, inputs=[x], outputs=[y],
               name='test_asin')

    def test_Asinh(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Asinh',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        y = np.arcsinh(x)  # expected output [-0.88137358,  0.,  0.88137358]
        expect(node, inputs=[x], outputs=[y],
               name='test_asinh_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.arcsinh(x)
        expect(node, inputs=[x], outputs=[y],
               name='test_asinh')

    def test_Atan(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Atan',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        y = np.arctan(x)
        expect(node, inputs=[x], outputs=[y],
               name='test_atan_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.arctan(x)
        expect(node, inputs=[x], outputs=[y],
               name='test_atan')

    def test_Atanh(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Atanh',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        y = np.arctanh(x)  # expected output [-0.54930615,  0.,  0.54930615]
        expect(node, inputs=[x], outputs=[y],
               name='test_atanh_example')

        x = np.random.uniform(0.0, 1.0, (3, 4, 5)).astype(np.float32)
        y = np.arctanh(x)
        expect(node, inputs=[x], outputs=[y],
               name='test_atanh')

    def test_selu(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Selu',
            inputs=['x'],
            outputs=['y'],
            alpha=2.0,
            gamma=3.0
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        # expected output [-3.79272318, 0., 3.]
        y = np.clip(x, 0, np.inf) * 3.0 + \
            (np.exp(np.clip(x, -np.inf, 0)) - 1) * 2.0 * 3.0
        expect(node, inputs=[x], outputs=[y],
               name='test_selu_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.clip(x, 0, np.inf) * 3.0 + \
            (np.exp(np.clip(x, -np.inf, 0)) - 1) * 2.0 * 3.0
        expect(node, inputs=[x], outputs=[y],
               name='test_selu')

    def test_selu_default(self):  # type: () -> None
        default_alpha = 1.67326319217681884765625
        default_gamma = 1.05070102214813232421875
        node = onnx.helper.make_node(
            'Selu',
            inputs=['x'],
            outputs=['y'],
        )
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.clip(x, 0, np.inf) * default_gamma + \
            (np.exp(np.clip(x, -np.inf, 0)) - 1) * default_alpha * default_gamma
        expect(node, inputs=[x], outputs=[y],
               name='test_selu_default')

    def test_elu(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Elu',
            inputs=['x'],
            outputs=['y'],
            alpha=2.0
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        # expected output [-1.2642411, 0., 1.]
        y = np.clip(x, 0, np.inf) + (np.exp(np.clip(x, -np.inf, 0)) - 1) * 2.0
        expect(node, inputs=[x], outputs=[y],
               name='test_elu_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.clip(x, 0, np.inf) + (np.exp(np.clip(x, -np.inf, 0)) - 1) * 2.0
        expect(node, inputs=[x], outputs=[y],
               name='test_elu')

    def test_elu_default(self):  # type: () -> None
        default_alpha = 1.0
        node = onnx.helper.make_node(
            'Elu',
            inputs=['x'],
            outputs=['y'],
        )
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.clip(x, 0, np.inf) + \
            (np.exp(np.clip(x, -np.inf, 0)) - 1) * default_alpha
        expect(node, inputs=[x], outputs=[y],
               name='test_elu_default')

    def test_equal(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Equal',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        x = (np.random.randn(3, 4, 5) * 10).astype(np.int32)
        y = (np.random.randn(3, 4, 5) * 10).astype(np.int32)
        z = np.equal(x, y)

        expect(node, inputs=[x, y], outputs=[z],
               name='test_equal')

    def test_equal_broadcast(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Equal',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        x = (np.random.randn(3, 4, 5) * 10).astype(np.int32)
        y = (np.random.randn(5) * 10).astype(np.int32)
        z = np.equal(x, y).astype(np.int32)  # need to convert to int type
        expect(node, inputs=[x, y], outputs=[z],
               name='test_equal_bcast')

    def test_less(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Less',
            inputs=['x', 'y'],
            outputs=['less'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(3, 4, 5).astype(np.float32)
        z = np.less(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_less')

    def test_less_broadcast(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Less',
            inputs=['x', 'y'],
            outputs=['less'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(5).astype(np.float32)
        z = np.less(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_less_bcast')

    def test_sign(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Sign',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array(range(-5, 6)).astype(np.float32)
        y = np.sign(x)
        expect(node, inputs=[x], outputs=[y],
               name='test_sign')

    def test_sub(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Sub',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        x = np.array([1, 2, 3]).astype(np.float32)
        y = np.array([3, 2, 1]).astype(np.float32)
        z = x - y  # expected output [-2., 0., 2.]
        expect(node, inputs=[x, y], outputs=[z],
               name='test_sub_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(3, 4, 5).astype(np.float32)
        z = x - y
        expect(node, inputs=[x, y], outputs=[z],
               name='test_sub')

    def test_sub_broadcast(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Sub',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(5).astype(np.float32)
        z = x - y
        expect(node, inputs=[x, y], outputs=[z],
               name='test_sub_bcast')

    def test_sqrt(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Sqrt',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([1, 4, 9]).astype(np.float32)
        y = np.sqrt(x)  # expected output [1., 2., 3.]
        expect(node, inputs=[x], outputs=[y],
               name='test_sqrt_example')

        x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
        y = np.sqrt(x)
        expect(node, inputs=[x], outputs=[y],
               name='test_sqrt')

    def test_log(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Log',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([1, 10]).astype(np.float32)
        y = np.log(x)  # expected output [0., 2.30258512]
        expect(node, inputs=[x], outputs=[y],
               name='test_log_example')

        x = np.exp(np.random.randn(3, 4, 5).astype(np.float32))
        y = np.log(x)
        expect(node, inputs=[x], outputs=[y],
               name='test_log')

    def test_greater(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Greater',
            inputs=['x', 'y'],
            outputs=['greater'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(3, 4, 5).astype(np.float32)
        z = np.greater(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_greater')

    def test_greater_broadcast(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Greater',
            inputs=['x', 'y'],
            outputs=['greater'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(5).astype(np.float32)
        z = np.greater(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_greater_bcast')

    def test_hardsigmoid(self):  # type: () -> None
        node = onnx.helper.make_node(
            'HardSigmoid',
            inputs=['x'],
            outputs=['y'],
            alpha=0.5,
            beta=0.6
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        y = np.clip(x * 0.5 + 0.6, 0, 1)  # expected output [0.1, 0.6, 1.]
        expect(node, inputs=[x], outputs=[y],
               name='test_hardsigmoid_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.clip(x * 0.5 + 0.6, 0, 1)
        expect(node, inputs=[x], outputs=[y],
               name='test_hardsigmoid')

    def test_hardsigmoid_default(self):  # type: () -> None
        default_alpha = 0.2
        default_beta = 0.5
        node = onnx.helper.make_node(
            'HardSigmoid',
            inputs=['x'],
            outputs=['y'],
        )
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.clip(x * default_alpha + default_beta, 0, 1)
        expect(node, inputs=[x], outputs=[y],
               name='test_hardsigmoid_default')

    def test_identity(self):
        node = onnx.helper.make_node(
            'Identity',
            inputs=['x'],
            outputs=['y'],
        )

        data = np.array([[[
            [1, 2],
            [3, 4],
        ]]], dtype=np.float32)

        expect(node, inputs=[data], outputs=[data],
               name='test_identity')

    def test_softplus(self):
        node = onnx.helper.make_node(
            'Softplus',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        # expected output [0.31326166, 0.69314718, 1.31326163]
        y = np.log(np.exp(x) + 1)
        expect(node, inputs=[x], outputs=[y],
               name='test_softplus_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.log(np.exp(x) + 1)
        expect(node, inputs=[x], outputs=[y],
               name='test_softplus')

    def test_softsign(self):
        node = onnx.helper.make_node(
            'Softsign',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        y = np.array([-0.5, 0, 0.5]).astype(np.float32)
        expect(node, inputs=[x], outputs=[y],
               name='test_softsign_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = x / (1 + np.abs(x))
        expect(node, inputs=[x], outputs=[y],
               name='test_softsign')

    def test_mean(self):
        data_0 = np.array([3, 0, 2]).astype(np.float32)
        data_1 = np.array([1, 3, 4]).astype(np.float32)
        data_2 = np.array([2, 6, 6]).astype(np.float32)
        result = np.array([2, 3, 4]).astype(np.float32)
        node = onnx.helper.make_node(
            'Mean',
            inputs=['data_0', 'data_1', 'data_2'],
            outputs=['result'],
        )
        expect(node, inputs=[data_0, data_1, data_2], outputs=[result],
               name='test_mean_example')

        node = onnx.helper.make_node(
            'Mean',
            inputs=['data_0'],
            outputs=['result'],
        )
        expect(node, inputs=[data_0], outputs=[data_0],
               name='test_mean_one_input')

        result = np.divide(np.add(data_0, data_1), 2.)
        node = onnx.helper.make_node(
            'Mean',
            inputs=['data_0', 'data_1'],
            outputs=['result'],
        )
        expect(node, inputs=[data_0, data_1], outputs=[result],
               name='test_mean_two_inputs')

    def test_transpose_default(self):  # type: () -> None
        shape = (2, 3, 4)
        data = np.random.random_sample(shape).astype(np.float32)

        node = onnx.helper.make_node(
            'Transpose',
            inputs=['data'],
            outputs=['transposed']
        )

        transposed = np.transpose(data)
        expect(node, inputs=[data], outputs=[transposed],
               name='test_transpose_default')

    def test_transpose_all_permutations(self):  # type: () -> None
        shape = (2, 3, 4)
        data = np.random.random_sample(shape).astype(np.float32)
        permutations = list(itertools.permutations(np.arange(len(shape))))

        for i in range(len(permutations)):
            node = onnx.helper.make_node(
                'Transpose',
                inputs=['data'],
                outputs=['transposed'],
                perm=permutations[i]
            )
            transposed = np.transpose(data, permutations[i])
            expect(node, inputs=[data], outputs=[transposed],
                   name='test_transpose_all_permutations_' + str(i))

    def test_max(self):
        data_0 = np.array([3, 2, 1]).astype(np.float32)
        data_1 = np.array([1, 4, 4]).astype(np.float32)
        data_2 = np.array([2, 5, 3]).astype(np.float32)
        result = np.array([3, 5, 4]).astype(np.float32)
        # todo, not support 3 inputs
        node = onnx.helper.make_node(
            'Max',
            inputs=['data_0', 'data_1', 'data_2'],
            outputs=['result'],
        )
        expect(node, inputs=[data_0, data_1, data_2], outputs=[result],
               name='test_max_example')

        # todo, not support 1 inputs
        node = onnx.helper.make_node(
            'Max',
            inputs=['data_0'],
            outputs=['result'],
        )
        expect(node, inputs=[data_0], outputs=[data_0],
               name='test_max_one_input')

        result = np.maximum(data_0, data_1)
        node = onnx.helper.make_node(
            'Max',
            inputs=['data_0', 'data_1'],
            outputs=['result'],
        )
        expect(node, inputs=[data_0, data_1], outputs=[result],
               name='test_max_two_inputs')

    def test_min(self):
        data_0 = np.array([3, 2, 1]).astype(np.float32)
        data_1 = np.array([1, 4, 4]).astype(np.float32)
        data_2 = np.array([2, 5, 0]).astype(np.float32)
        result = np.array([1, 2, 0]).astype(np.float32)
        node = onnx.helper.make_node(
            'Min',
            inputs=['data_0', 'data_1', 'data_2'],
            outputs=['result'],
        )
        expect(node, inputs=[data_0, data_1, data_2], outputs=[result],
               name='test_min_example')

        node = onnx.helper.make_node(
            'Min',
            inputs=['data_0'],
            outputs=['result'],
        )
        expect(node, inputs=[data_0], outputs=[data_0],
               name='test_min_one_input')

        result = np.minimum(data_0, data_1)
        node = onnx.helper.make_node(
            'Min',
            inputs=['data_0', 'data_1'],
            outputs=['result'],
        )
        expect(node, inputs=[data_0, data_1], outputs=[result],
               name='test_min_two_inputs')

    def test_shape(self):
        node = onnx.helper.make_node(
            'Shape',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([
            [1, 2, 3],
            [4, 5, 6],
        ]).astype(np.float32)
        y = np.array([
            2, 3,
        ]).astype(np.int64)

        expect(node, inputs=[x], outputs=[y],
               name='test_shape_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.array(x.shape).astype(np.int64)

        expect(node, inputs=[x], outputs=[y],
               name='test_shape')

    def test_and(self):  # type: () -> None
        node = onnx.helper.make_node(
            'And',
            inputs=['x', 'y'],
            outputs=['and'],
        )

        # 2d
        x = (np.random.randn(3, 4) > 0).astype(np.bool)
        y = (np.random.randn(3, 4) > 0).astype(np.bool)
        z = np.logical_and(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_and2d')

        # 3d
        x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
        y = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
        z = np.logical_and(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_and3d')

        # 4d
        x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
        y = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
        z = np.logical_and(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_and4d')
               
    def test_and_broadcast(self):  # type: () -> None
        node = onnx.helper.make_node(
            'And',
            inputs=['x', 'y'],
            outputs=['and'],
        )

        # 3d vs 1d
        x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
        y = (np.random.randn(5) > 0).astype(np.bool)
        z = np.logical_and(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_and_bcast3v1d')

        # 3d vs 2d
        x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
        y = (np.random.randn(4, 5) > 0).astype(np.bool)
        z = np.logical_and(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_and_bcast3v2d')

        # 4d vs 2d
        x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
        y = (np.random.randn(5, 6) > 0).astype(np.bool)
        z = np.logical_and(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_and_bcast4v2d')

        # 4d vs 3d
        x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
        y = (np.random.randn(4, 5, 6) > 0).astype(np.bool)
        z = np.logical_and(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_and_bcast4v3d')

        # 4d vs 4d
        x = (np.random.randn(1, 4, 1, 6) > 0).astype(np.bool)
        y = (np.random.randn(3, 1, 5, 6) > 0).astype(np.bool)
        z = np.logical_and(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_and_bcast4v4d')

    def test_or(self):
        node = onnx.helper.make_node(
            'Or',
            inputs=['x', 'y'],
            outputs=['or'],
        )

        # 2d
        x = (np.random.randn(3, 4) > 0).astype(np.bool)
        y = (np.random.randn(3, 4) > 0).astype(np.bool)
        z = np.logical_or(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_or2d')

        # 3d
        x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
        y = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
        z = np.logical_or(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_or3d')

        # 4d
        x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
        y = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
        z = np.logical_or(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_or4d')

    def test_or_broadcast(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Or',
            inputs=['x', 'y'],
            outputs=['or'],
        )

        # 3d vs 1d
        x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
        y = (np.random.randn(5) > 0).astype(np.bool)
        z = np.logical_or(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_or_bcast3v1d')

        # 3d vs 2d
        x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
        y = (np.random.randn(4, 5) > 0).astype(np.bool)
        z = np.logical_or(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_or_bcast3v2d')

        # 4d vs 2d
        x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
        y = (np.random.randn(5, 6) > 0).astype(np.bool)
        z = np.logical_or(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_or_bcast4v2d')

        # 4d vs 3d
        x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
        y = (np.random.randn(4, 5, 6) > 0).astype(np.bool)
        z = np.logical_or(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_or_bcast4v3d')

        # 4d vs 4d
        x = (np.random.randn(1, 4, 1, 6) > 0).astype(np.bool)
        y = (np.random.randn(3, 1, 5, 6) > 0).astype(np.bool)
        z = np.logical_or(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_or_bcast4v4d')

    def test_xor(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Xor',
            inputs=['x', 'y'],
            outputs=['xor'],
        )

        # 2d
        x = (np.random.randn(3, 4) > 0).astype(np.bool)
        y = (np.random.randn(3, 4) > 0).astype(np.bool)
        z = np.logical_xor(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_xor2d')

        # 3d
        x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
        y = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
        z = np.logical_xor(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_xor3d')

        # 4d
        x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
        y = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
        z = np.logical_xor(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_xor4d')

    def test_xor_broadcast(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Xor',
            inputs=['x', 'y'],
            outputs=['xor'],
        )

        # 3d vs 1d
        x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
        y = (np.random.randn(5) > 0).astype(np.bool)
        z = np.logical_xor(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_xor_bcast3v1d')

        # 3d vs 2d
        x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
        y = (np.random.randn(4, 5) > 0).astype(np.bool)
        z = np.logical_xor(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_xor_bcast3v2d')

        # 4d vs 2d
        x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
        y = (np.random.randn(5, 6) > 0).astype(np.bool)
        z = np.logical_xor(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_xor_bcast4v2d')

        # 4d vs 3d
        x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
        y = (np.random.randn(4, 5, 6) > 0).astype(np.bool)
        z = np.logical_xor(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_xor_bcast4v3d')

        # 4d vs 4d
        x = (np.random.randn(1, 4, 1, 6) > 0).astype(np.bool)
        y = (np.random.randn(3, 1, 5, 6) > 0).astype(np.bool)
        z = np.logical_xor(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_xor_bcast4v4d')

    def test_not(self):
        node = onnx.helper.make_node(
            'Not',
            inputs=['x'],
            outputs=['not'],
        )

        # 2d
        x = (np.random.randn(3, 4) > 0).astype(np.bool)
        expect(node, inputs=[x], outputs=[np.logical_not(x)],
               name='test_not_2d')

        # 3d
        x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
        expect(node, inputs=[x], outputs=[np.logical_not(x)],
               name='test_not_3d')

        # 4d
        x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
        expect(node, inputs=[x], outputs=[np.logical_not(x)],
               name='test_not_4d')

    def test_neg(self):
        node = onnx.helper.make_node(
            'Neg',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-4, 2]).astype(np.float32)
        y = np.negative(x)  # expected output [4., -2.],
        expect(node, inputs=[x], outputs=[y],
               name='test_neg_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.negative(x)
        expect(node, inputs=[x], outputs=[y],
               name='test_neg')

    def test_reciprocal(self):
        node = onnx.helper.make_node(
            'Reciprocal',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-4, 2]).astype(np.float32)
        y = np.reciprocal(x)  # expected output [-0.25, 0.5],
        expect(node, inputs=[x], outputs=[y],
               name='test_reciprocal_example')

        x = np.random.rand(3, 4, 5).astype(np.float32) + 0.5
        y = np.reciprocal(x)
        expect(node, inputs=[x], outputs=[y],
               name='test_reciprocal')

    def test_batchnorm(self):  # type: () -> None
        # we changed this test cases
        # according to the paper https://arxiv.org/pdf/1502.03167.pdf
        def _batchnorm_test_mode(x, s, bias, mean, var, momentum=0.9, epsilon=1e-5):  # type: ignore
            dims_x = len(x.shape)
            dim_ones = (1,) * (dims_x - 2)
            s = s.reshape(-1, *dim_ones)
            bias = bias.reshape(-1, *dim_ones)
            mean = mean.reshape(-1, *dim_ones)
            var = var.reshape(-1, *dim_ones)
            batch_m = x.mean(axis=(0, 2, 3), keepdims=True)
            batch_v = x.var(axis=(0, 2, 3), keepdims=True)
            return s * (x - batch_m) / np.sqrt(batch_v + epsilon) + bias

        # input size: (1, 2, 1, 3)
        x = np.array([[[[-1, 0, 1]], [[2, 3, 4]]]]).astype(np.float32)
        s = np.array([1.0, 1.5]).astype(np.float32)
        bias = np.array([0, 1]).astype(np.float32)
        mean = np.array([0, 3]).astype(np.float32)
        var = np.array([1, 1.5]).astype(np.float32)
        y = _batchnorm_test_mode(x, s, bias, mean, var).astype(np.float32)

        node = onnx.helper.make_node(
            'BatchNormalization',
            inputs=['x', 's', 'bias', 'mean', 'var'],
            outputs=['y'],
        )

        # output size: (1, 2, 1, 3)
        expect(node, inputs=[x, s, bias, mean, var], outputs=[y],
               name='test_batchnorm_example')

        # input size: (2, 3, 4, 5)
        x = np.random.randn(2, 3, 4, 5).astype(np.float32)
        s = np.random.randn(3).astype(np.float32)
        bias = np.random.randn(3).astype(np.float32)
        mean = np.random.randn(3).astype(np.float32)
        var = np.random.rand(3).astype(np.float32)
        epsilon = 1e-2
        y = _batchnorm_test_mode(
            x, s, bias, mean, var, epsilon).astype(np.float32)

        node = onnx.helper.make_node(
            'BatchNormalization',
            inputs=['x', 's', 'bias', 'mean', 'var'],
            outputs=['y'],
            epsilon=epsilon,
        )

        # output size: (2, 3, 4, 5)
        expect(node, inputs=[x, s, bias, mean, var], outputs=[y],
               name='test_batchnorm_epsilon')

    def test_softmax(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Softmax',
            inputs=['x'],
            outputs=['y'],
        )
        x = np.array([[-1, 0, 1]]).astype(np.float32)
        # expected output [[0.09003058, 0.24472848, 0.66524094]]
        y = np.exp(x) / np.sum(np.exp(x), axis=1)
        expect(node, inputs=[x], outputs=[y],
               name='test_softmax_example')

    def test_softmax_axis(self):  # type: () -> None
        def softmax_2d(x):  # type: (np.ndarray) -> np.ndarray
            max_x = np.max(x, axis=1).reshape((-1, 1))
            exp_x = np.exp(x - max_x)
            return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))

        x = np.array([[0, 1, 2, 3], [10000, 10001, 10002, 10003]]).astype(np.float32)
        # expected output [[0.0320586, 0.08714432, 0.23688284, 0.64391428],
        #                 [0.0320586, 0.08714432, 0.23688284, 0.64391428]]
        y = softmax_2d(x)

        node = onnx.helper.make_node(
            'Softmax',
            inputs=['x'],
            outputs=['y'],
        )
        expect(node, inputs=[x], outputs=[y],
               name='test_softmax_large_number')

        x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
        node = onnx.helper.make_node(
            'Softmax',
            inputs=['x'],
            outputs=['y'],
            axis=0,
        )
        y = softmax_2d(x.reshape(1, 60)).reshape(3, 4, 5)
        expect(node, inputs=[x], outputs=[y],
               name='test_softmax_axis_0')

        node = onnx.helper.make_node(
            'Softmax',
            inputs=['x'],
            outputs=['y'],
            axis=1,
        )
        y = softmax_2d(x.reshape(3, 20)).reshape(3, 4, 5)
        expect(node, inputs=[x], outputs=[y],
               name='test_softmax_axis_1')

        # default axis is 1
        node = onnx.helper.make_node(
            'Softmax',
            inputs=['x'],
            outputs=['y'],
        )
        expect(node, inputs=[x], outputs=[y],
               name='test_softmax_default_axis')

        node = onnx.helper.make_node(
            'Softmax',
            inputs=['x'],
            outputs=['y'],
            axis=2,
        )
        y = softmax_2d(x.reshape(12, 5)).reshape(3, 4, 5)
        expect(node, inputs=[x], outputs=[y],
               name='test_softmax_axis_2')

        node = onnx.helper.make_node(
            'Softmax',
            inputs=['x'],
            outputs=['y'],
            axis=-1,
        )
        y = softmax_2d(x.reshape(12, 5)).reshape(3, 4, 5)
        expect(node, inputs=[x], outputs=[y],
               name='test_softmax_negative_axis')

    def test_div(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Div',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        x = np.array([3, 4]).astype(np.float32)
        y = np.array([1, 2]).astype(np.float32)
        z = x / y  # expected output [3., 2.]
        expect(node, inputs=[x, y], outputs=[z],
            name='test_div_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.rand(3, 4, 5).astype(np.float32) + 1.0
        z = x / y
        expect(node, inputs=[x, y], outputs=[z],
            name='test_div')

    def test_div_broadcast(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Div',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.rand(5).astype(np.float32) + 1.0
        z = x / y
        expect(node, inputs=[x, y], outputs=[z],
               name='test_div_bcast')

    def test_pow(self):
        node = onnx.helper.make_node(
            'Pow',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        x = np.array([1, 2, 3]).astype(np.float32)
        y = np.array([4, 5, 6]).astype(np.float32)  # todo, not exactly same
        z = np.power(x, y)  # expected output [1., 32., 729.]
        expect(node, inputs=[x, y], outputs=[z],
               name='test_pow_example')

        x = np.arange(24).reshape(2, 3, 4).astype(
            np.float32)  # todo, cannot too big here
        y = np.random.randn(2, 3, 4).astype(np.float32)
        z = np.power(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_pow')

    def test_pow_broadcast(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Pow',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        x = np.array([1, 2, 3]).astype(np.float32)
        y = np.array(2).astype(np.float32)
        z = np.power(x, y)  # expected output [1., 4., 9.]
        expect(node, inputs=[x, y], outputs=[z],
               name='test_pow_bcast_scalar')

        node = onnx.helper.make_node(
            'Pow',
            inputs=['x', 'y'],
            outputs=['z'],
        )
        x = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
        y = np.array([1, 2, 3]).astype(np.float32)
        # expected output [[1, 4, 27], [4, 25, 216]]
        z = np.power(x, y).astype(np.float32)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_pow_bcast_array')

    def test_clip(self):
        node = onnx.helper.make_node(
            'Clip',
            inputs=['x', 'min', 'max'],
            outputs=['y'],
        )

        x = np.array([-2, 0, 2]).astype(np.float32)
        min_val = np.float32(-1)
        max_val = np.float32(1)
        y = np.clip(x, min_val, max_val)  # expected output [-1., 0., 1.]
        expect(node, inputs=[x, min_val, max_val], outputs=[y],
               name='test_clip_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.clip(x, min_val, max_val)
        expect(node, inputs=[x, min_val, max_val], outputs=[y],
               name='test_clip')
        node = onnx.helper.make_node(
            'Clip',
            inputs=['x', 'min', 'max'],
            outputs=['y'],
        )

        min_val = np.float32(-5)
        max_val = np.float32(5)

        x = np.array([-1, 0, 1]).astype(np.float32)
        y = np.array([-1, 0, 1]).astype(np.float32)
        expect(node, inputs=[x, min_val, max_val], outputs=[y],
               name='test_clip_inbounds')

        x = np.array([-6, 0, 6]).astype(np.float32)
        y = np.array([-5, 0, 5]).astype(np.float32)
        expect(node, inputs=[x, min_val, max_val], outputs=[y],
               name='test_clip_outbounds')

        x = np.array([-1, 0, 6]).astype(np.float32)
        y = np.array([-1, 0, 5]).astype(np.float32)
        expect(node, inputs=[x, min_val, max_val], outputs=[y],
               name='test_clip_splitbounds')

    def test_clip_default(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Clip',
            inputs=['x', 'min'],
            outputs=['y'],
        )
        min_val = np.float32(0)
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.clip(x, min_val, np.inf)
        expect(node, inputs=[x, min_val], outputs=[y],
               name='test_clip_default_min')

        no_min = ""  # optional input, not supplied
        node = onnx.helper.make_node(
            'Clip',
            inputs=['x', no_min, 'max'],
            outputs=['y'],
        )
        max_val = np.float32(0)
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.clip(x, -np.inf, max_val)
        expect(node, inputs=[x, max_val], outputs=[y],
               name='test_clip_default_max')

        no_max = ""  # optional input, not supplied
        node = onnx.helper.make_node(
            'Clip',
            inputs=['x', no_min, no_max],
            outputs=['y'],
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        y = np.array([-1, 0, 1]).astype(np.float32)
        expect(node, inputs=[x], outputs=[y],
               name='test_clip_default_inbounds')

    def test_prelu(self):
        node = onnx.helper.make_node(
            'PRelu',
            inputs=['x', 'slope'],
            outputs=['y'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        slope = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * slope

        expect(node, inputs=[x, slope], outputs=[y],
               name='test_prelu_example')

    #todo, not support prelu broadcast
    def test_prelu_broadcast(self):  # type: () -> None
        node = onnx.helper.make_node(
            'PRelu',
            inputs=['x', 'slope'],
            outputs=['y'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        slope = np.random.randn(5).astype(np.float32)
        y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * slope

        expect(node, inputs=[x, slope], outputs=[y],
               name='test_prelu_broadcast')

    def test_mul(self):
        node = onnx.helper.make_node(
            'Mul',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        x = np.array([1, 2, 3]).astype(np.float32)
        y = np.array([4, 5, 6]).astype(np.float32)
        z = x * y  # expected output [4., 10., 18.]
        expect(node, inputs=[x, y], outputs=[z],
               name='test_mul_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(3, 4, 5).astype(np.float32)
        z = x * y
        expect(node, inputs=[x, y], outputs=[z],
               name='test_mul')

    def test_mul_broadcast(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Mul',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(5).astype(np.float32)
        z = x * y
        expect(node, inputs=[x, y], outputs=[z],
               name='test_mul_bcast')


# return padding shape of conv2d or pooling
def get_pad_shape(auto_pad,  # type: Text
                  input_spatial_shape,  # type: Sequence[int]
                  kernel_spatial_shape,  # type: Sequence[int]
                  strides_spatial,  # type: Sequence[int]
                  output_spatial_shape  # type: Sequence[int]
                  ):  # type: (...) -> Sequence[int]
    pad_shape = [0] * len(input_spatial_shape)
    if auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
        for i in range(len(input_spatial_shape)):
            pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial[i] + \
                kernel_spatial_shape[i] - input_spatial_shape[i]
    elif auto_pad == 'VALID':
        pass
    return pad_shape

# return output shape of conv2d or pooling


def get_output_shape(auto_pad,  # type: Text
                     input_spatial_shape,  # type: Sequence[int]
                     kernel_spatial_shape,  # type: Sequence[int]
                     strides_spatial  # type: Sequence[int]
                     ):  # type: (...) -> Sequence[int]
    out_shape = [0] * len(input_spatial_shape)
    if auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
        for i in range(len(input_spatial_shape)):
            out_shape[i] = int(
                np.ceil(
                    float(
                        input_spatial_shape[i])
                    / float(
                        strides_spatial[i])))
    elif auto_pad == 'VALID':
        for i in range(len(input_spatial_shape)):
            out_shape[i] = int(np.ceil(float(
                input_spatial_shape[i] - (kernel_spatial_shape[i] - 1)) / float(strides_spatial[i])))
    return out_shape


def pool(padded,  # type: np.ndarray
         x_shape,  # type: Sequence[int]
         kernel_shape,  # type: Sequence[int]
         strides_shape,  # type: Sequence[int]
         out_shape,  # type: Sequence[int]
         pad_shape,  # type: Sequence[int]
         pooling_type,  # type: Text
         count_include_pad=0  # type: int
         ):  # type: (...) -> np.ndarray
    spatial_size = len(x_shape) - 2
    y = np.zeros([x_shape[0], x_shape[1]] + list(out_shape))

    for shape in itertools.product(range(x_shape[0]), range(x_shape[1]), *[range(int(
            (x_shape[i + 2] + pad_shape[i] - kernel_shape[i]) / strides_shape[i] + 1)) for i in range(spatial_size)]):
        window = padded[shape[0], shape[1]]
        window_vals = np.array([window[i] for i in list(
            itertools.product(
                *[range(strides_shape[i] * shape[i + 2], strides_shape[i] * shape[i + 2] + kernel_shape[i]) for i in
                  range(spatial_size)])
        )])
        if pooling_type == 'AVG':
            f = np.average
        elif pooling_type == 'MAX':
            f = np.max
        else:
            raise NotImplementedError(
                'Pooling type {} does not support. Should be AVG, MAX'.format(pooling_type))

        if count_include_pad == 1 and pooling_type == 'AVG':
            y[shape] = f(window_vals)
        else:
            y[shape] = f(window_vals[np.where(~np.isnan(window_vals))])
    return y.astype(np.float32)


if __name__ == '__main__':
    unittest.main()
