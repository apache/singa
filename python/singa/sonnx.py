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

import warnings
from collections import deque

from . import singa_wrap as singa
from . import autograd
from . import tensor

import collections

from onnx import (checker, helper, numpy_helper, GraphProto, NodeProto, TensorProto, OperatorSetIdProto)
from onnx.backend.base import Backend, BackendRep
import onnx
import numpy as np


def postorderRecursive(root, root_t):
    """
    return a list by the topological ordering (postorder of Depth-first search)
    Args:
        root: singa operator
    Args:
        root_t: tensor
    Returns: 
        deque[int]
    """

    def recursive(root, yid, root_t, res):
        if root:
            for srcop, yid, y, _ in root.src:
                recursive(srcop, yid, y, res)
            res.append((root, yid, root_t))

    res = deque([])
    recursive(root, None, root_t, res)
    return res


def force_unicode(s):
    """
    return string of a bytes
    ! borrow from onnx
    Args:
        s: string or bytes
    Returns: 
        string
    """
    try:
        return s.decode('utf-8')
    except AttributeError:
        return s

def get_pad_shape(auto_pad, input_spatial_shape, kernel_spatial_shape, strides_spatial, output_spatial_shape):
    """
    return padding shape of conv2d or pooling,
    ! borrow from onnx
    Args:
        auto_pad: string
    Args:
        input_spatial_shape: list[int]
    Args:
        kernel_spatial_shape: list[int]
    Args:
        strides_spatial: list[int]
    Args:
        output_spatial_shape: list[int]
    Returns: 
        list[int]
    """
    pad_shape = [0] * len(input_spatial_shape)
    if auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
        for i in range(len(input_spatial_shape)):
            pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial[i] + \
                kernel_spatial_shape[i] - input_spatial_shape[i]
            if (pad_shape[i] % 2) == 0: 
                pad_shape[i] = pad_shape[i] // 2
            else:
                # once the padding is odd, it means we must add extra padding at one end of the input
                raise ValueError("Not implemented two directional padding")
    elif auto_pad == 'VALID':
        pass
    return pad_shape


def get_output_shape(auto_pad, input_spatial_shape, kernel_spatial_shape, strides_spatial):
    """
    return output shape of conv2d or pooling,
    ! borrow from onnx
    Args:
        auto_pad: string
    Args:
        input_spatial_shape: list[int]
    Args:
        kernel_spatial_shape: list[int]
    Args:
        strides_spatial: list[int]
    Returns: 
        list[int]
    """
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
            out_shape[i] = int(np.ceil(float(input_spatial_shape[i] - (kernel_spatial_shape[i] - 1)) / float(strides_spatial[i])))
    return out_shape



class SingaFrontend(object):
    """
    This class provides mthods to convert model from singa to onnx. 
    """

    # This number indicates the target onnx operator set version
    _target_opset_version = 10

    # beceuase singa's operators are different from onnx.
    # we define a dict for the name projection
    # "singa op name": "onnx op name"
    _rename_operators = {
        '_Conv2d': 'Conv',
        'ReLU': 'Relu',
        'Dummy': 'Constant',
        'MaxPool2d': 'MaxPool',
        'AvgPool2d':  'AveragePool',
        'SoftMax': 'Softmax',
        'Sigmoid': 'Sigmoid',
        'Add': 'Add',
        'Matmul': 'MatMul',
        '_BatchNorm2d': 'BatchNormalization',
        'Concat': 'Concat',
        'Flatten': 'Flatten',
        'AddBias': 'Add',
        # 'GEMM': 'Gemm',
        'Reshape': 'Reshape',
        'Sum': 'Sum',
        'cos': 'Cos',
        'cosh': 'Cosh',
        'sin': 'Sin',
        'sinh': 'Sinh',
        'tan': 'Tan',
        'tanh': 'Tanh',
        'acos': 'Acos',
        'acosh': 'Acosh',
        'asin': 'Asin',
        'asinh': 'Asinh',
        'atan': 'Atan',
        'atanh': 'Atanh',
        'SeLU' : 'Selu',
        'Elu' : 'Elu',
        'Equal': 'Equal',
        'Less': 'Less',
    }

    # this dict indicates the operators that need extra handle
    # each indicates a function name
    _special_operators = {
        '_Conv2d': '_create_conv_pool',
        '_Pooling2d': '_create_conv_pool',
        'Dummy': '_create_dummy',
        '_BatchNorm2d': '_create_batch_norm',
        'Concat': '_create_concat',
        'Flatten': '_create_flatten',
        # 'GEMM': '_create_gemm',
        'Reshape': '_create_reshape',
        'SoftMax': '_create_softmax',
        'SeLU': '_create_selu',
        'Elu' : '_create_elu',
    }

    # some ops(such as batchnorm) has inputs we cannot handle directly,
    # so we record these items firstly so that we can handle then
    # at other place.
    _unhandled_operators = {}

    @classmethod
    def _create_elu(cls, op, op_t):
        """
        get a onnx node from singa elu operator
        Args:
            op: a given operator
        Args:
            op_t: the tensor of the operator
        Returns: 
            the onnx node
        """
        node = cls._common_singa_tensor_to_onnx_node(op, op_t)

        node.attribute.extend([
            helper.make_attribute('alpha', op.alpha),
        ])
        return node

    @classmethod
    def _create_selu(cls, op, op_t):
        """
        get a onnx node from singa SeLU operator
        Args:
            op: a given operator
        Args:
            op_t: the tensor of the operator
        Returns: 
            the onnx node
        """
        node = cls._common_singa_tensor_to_onnx_node(op, op_t)

        node.attribute.extend([
            helper.make_attribute('alpha', op.alpha),
            helper.make_attribute('gamma', op.gamma),
        ])
        return node

    @classmethod
    def _create_reshape(cls, op, op_t):
        """
        get a onnx node from singa Concat operator
        Args:
            op: a given operator
        Args:
            op_t: the tensor of the operator
        Returns: 
            the onnx node
        """
        # make the shape node
        # because the reshape in singa does not provide its shape as input tensor
        shape_node_name = op.name+":shape"
        shape_node = NodeProto()
        shape_node.name = shape_node_name
        shape_node.op_type = cls._rename_operators.get("Dummy", "Dummy")
        shape_node.output.extend([shape_node_name])
        shape_node.attribute.extend([helper.make_attribute(
            'value', helper.make_tensor(
                name=shape_node_name,
                data_type=TensorProto.FLOAT,
                dims=[len(op_t.shape)],
                vals=op_t.shape,
            )
        )])
        # make the reshape node
        node = cls._common_singa_tensor_to_onnx_node(op, op_t)
        node.input.extend([shape_node_name])
        return [shape_node, node]

    @classmethod
    def _create_concat(cls, op, op_t):
        """
        get a onnx node from singa Concat operator
        Args:
            op: a given operator
        Args:
            op_t: the tensor of the operator
        Returns: 
            the onnx node
        """
        node = cls._common_singa_tensor_to_onnx_node(op, op_t)

        node.attribute.extend([
            helper.make_attribute('axis', op.axis),
        ])
        return node

    @classmethod
    def _create_softmax(cls, op, op_t):
        """
        get a onnx node from singa Concat operator
        Args:
            op: a given operator
        Args:
            op_t: the tensor of the operator
        Returns: 
            the onnx node
        """
        node = cls._common_singa_tensor_to_onnx_node(op, op_t)

        node.attribute.extend([
            helper.make_attribute('axis', op.axis),
        ])
        return node

    @classmethod
    def _create_flatten(cls, op, op_t):
        """
        get a onnx node from singa flatten operator
        Args:
            op: a given operator
        Args:
            op_t: the tensor of the operator
        Returns: 
            the onnx node
        """
        node = cls._common_singa_tensor_to_onnx_node(op, op_t)

        node.attribute.extend([
            helper.make_attribute('axis', op.start_axis),
        ])
        return node

    # @classmethod
    # def _create_gemm(cls, op, op_t):
    #     """
    #     get a onnx node from singa gemm operator
    #     Args:
    #         op: a given operator
    #     Args:
    #         op_t: the tensor of the operator
    #     Returns: 
    #         the onnx node
    #     """
    #     node = cls._common_singa_tensor_to_onnx_node(op, op_t)

    #     node.attribute.extend([
    #         helper.make_attribute('alpha', float(op.alpha)),
    #         helper.make_attribute('beta', float(op.beta)),
    #         helper.make_attribute('transA', 1 if op.transA else 0),
    #         helper.make_attribute('transB', 1 if op.transB else 0),
    #     ])

    #     return node

    @classmethod
    def _create_batch_norm(cls, op, op_t):
        """
        get a onnx node from singa _BatchNorm2d operator
        Args:
            op: a given operator
        Args:
            op_t: the tensor of the operator
        Returns: 
            the onnx node
        """
        nodes = []
        # firstly we add the running mean and var nodes
        running_values = [op.running_mean, op.running_var]
        for srcop, _, y, _ in op.src:
            if y is None and srcop.name in cls._unhandled_operators:
                node = cls._common_singa_tensor_to_onnx_node(srcop, op_t)
                running_value = running_values.pop(0)
                vals = tensor.to_numpy(tensor.from_raw_tensor(running_value)).astype(float)
                node.attribute.extend([helper.make_attribute(
                    'value', helper.make_tensor(
                        name=srcop.name,
                        data_type=TensorProto.FLOAT,
                        dims=[len(vals)],
                        vals=vals,
                    )
                )])
                nodes.append(node)

        # then we add the batchnorm op itself
        epsilon = 1e-5  # the epsilon value used in singa
        node = cls._common_singa_tensor_to_onnx_node(op, op_t)
        node.attribute.extend([
            helper.make_attribute('momentum', op.handle.factor),
            helper.make_attribute('epsilon', epsilon),
        ])
        nodes.append(node)

        return nodes

    @classmethod
    def _create_conv_pool(cls, op, op_t):
        """
        get a onnx node from singa _Conv2d and _Pooling2d operator
        Args:
            op: a given operator
        Args:
            op_t: the tensor of the operator
        Returns: 
            the onnx node
        """
        node = cls._common_singa_tensor_to_onnx_node(op, op_t)

        k = [op.handle.kernel_h, op.handle.kernel_w]
        s = [op.handle.stride_h, op.handle.stride_w]
        p = [
            op.handle.pad_h,
            op.handle.pad_w,
            op.handle.pad_w,
            op.handle.pad_h,
        ]

        node.attribute.extend([
            helper.make_attribute('kernel_shape', k),
            helper.make_attribute('pads', p),
            helper.make_attribute('strides', s),
        ])
        if cls._get_singa_op_type(op) == '_Conv2d':
            node.op_type = cls._rename_operators.get('_Conv2d')
            node.attribute.extend([
                helper.make_attribute('group', op.handle.group),
            ])

        elif op.handle.is_max_pooling:
            node.op_type = cls._rename_operators.get('MaxPool2d')
        else:
            node.op_type = cls._rename_operators.get('AvgPool2d')
        return node

    @classmethod
    def _create_dummy(cls, op, op_t):
        """
        get a onnx node from singa dummy (constant)
        Args:
            op: a given operator
        Args:
            op_t: the tensor of the operator
        Returns: 
            the onnx node
        """
        # for batchnorm, the running mean and var's op_t is None, we just return
        if op_t is None:
            cls._unhandled_operators[op.name] = op
            return None
        node = cls._common_singa_tensor_to_onnx_node(op, op_t)
        node.attribute.extend([helper.make_attribute(
            'value', helper.make_tensor(
                name=op.name,
                data_type=TensorProto.FLOAT,
                dims=op_t.shape,
                vals=tensor.to_numpy(op_t)
                .flatten()
                .astype(float),
            )
        )])
        del node.input[:]
        return node

    @classmethod
    def _common_singa_tensor_to_onnx_node(cls, op, op_t):
        """
        get a onnx node from a singa operator, prepare its type, inputs and outputs
        Args:
            op: a given operator
        Args:
            op: the tensor of the operator
        Returns: the onnx node
        """
        node_def = NodeProto()
        node_def.name = op.name

        optype = cls._get_singa_op_type(op)
        node_def.op_type = cls._rename_operators.get(optype, optype)

        inputs, outputs = cls._get_singa_op_inputs_outputs(op)
        node_def.input.extend(inputs)
        node_def.output.extend(outputs)

        return node_def

    @classmethod
    def singa_op_to_onnx_node(cls, op, op_t):
        """
        get a onnx node from singa operator
        Args:
            op: a given operator
        Args:
            op_t: the tensor of the operator
        Returns: 
            the onnx node
        """
        optype = cls._get_singa_op_type(op)
        # wether the operator needs special handler
        if optype in cls._special_operators:
            translator = getattr(cls, cls._special_operators[optype])
        else:
            translator = cls._common_singa_tensor_to_onnx_node
        nodes = translator(op, op_t)
        if not isinstance(nodes, collections.Iterable):
            nodes = [nodes]
        nodes = [node for node in nodes if node is not None]
        return nodes

    @classmethod
    def singa_to_onnx_graph(cls, inputs, y, model_name="sonnx"):
        """
        get onnx model from singa computational graph
        Args:
            inputs: a list of input tensors (each is initialized with a name)
        Args:
            y: a list of tensors, usually the outputs of the graph
        Returns: 
            the onnx model
        """
        assert len(y) == 1  # assume there is only one output
        y = y[0]

        graph_def = GraphProto()
        graph_def.name = model_name
        topol = postorderRecursive(y.creator, y)
        # since tensor's name might change
        # we record its id
        input_tensors = {id(x): x for x in inputs}
        # print(input_tensors)
        X = []
        Y = [helper.make_tensor_value_info(y.name, TensorProto.FLOAT, y.shape)]

        for op, yid, op_t in topol:
            optype = cls._get_singa_op_type(op)
            # print(op.name, cls._get_singa_op_type(op), op_t, optype, yid)
            if yid in input_tensors and optype == 'Dummy':
                # find the input by its id
                op_t = input_tensors[yid]
                dtype = TensorProto.FLOAT
                if op_t.dtype == tensor.int32:
                    dtype = TensorProto.INT32
                X.append(helper.make_tensor_value_info(op.name, dtype, op_t.shape))
            else:
                graph_def.node.extend(cls.singa_op_to_onnx_node(op, op_t))

        graph_def.input.extend(X)
        graph_def.output.extend(Y)
        return graph_def

    @classmethod
    def singa_to_onnx_model(cls, inputs, y, model_name="sonnx"):
        """
        get onnx model from singa computational graph
        Args:
            inputs: a list of input tensors (each is initialized with a name)
        Args:
            y: a list of tensors, usually the outputs of the graph
        Returns: 
            the onnx model
        """
        opset_id = OperatorSetIdProto()
        opset_id.version = cls._target_opset_version
        model = helper.make_model(cls.singa_to_onnx_graph(
            inputs, y, model_name="sonnx"), producer_name='sonnx',
            opset_imports=[opset_id])
        # print('The model is:\n{}'.format(model))
        checker.check_model(model)
        return model

    @classmethod
    def _get_singa_op_inputs_outputs(cls, op):
        """
        get inputs and outputs from a given operator
        Args:
            op: a given operator
        Returns: 
            inputs and outputs of the op
        """
        outputs = [op.output_name(idx) for yid, idx in op.y_id2idx.items()]
        inputs = [srcop.output_name(srcop.y_id2idx[yid])
                  for (srcop, yid, _, _) in op.src]
        return inputs, outputs

    @classmethod
    def _get_singa_op_type(cls, op):
        """
        get the operator type from a given operator
        Args:
            op: a given operator
        Returns: 
            operator type
        """
        return type(op).__name__


class OnnxNode(object):
    """
    Reimplementation of NodeProto from ONNX, but in a form
    more convenient to work with from Python.
    We may temporarily edit these nodes to get them into Caffe2 form,
    before actually translating into the Caffe2 protobuf, since this
    is easier than decomposing everything, and putting it back together
    when we're ready.
    """

    def __init__(self, node):
        self.name = str(node.name)
        self.op_type = str(node.op_type)
        self.attrs = OnnxAttributes.from_onnx(node.attribute)
        self.inputs = list(node.input)
        self.outputs = list(node.output)
    
    def getattr(self, key, default=None):
        return self.attrs[key] if key in self.attrs else default


class OnnxAttributes(dict):
    """
    This is a more convenient way to work with ONNX/Caffe2 attributes
    that is not the protobuf representation.
    """
    @staticmethod
    def from_onnx(args):
        d = OnnxAttributes()
        for arg in args:
            d[arg.name] = helper.get_attribute_value(arg)
        return d


class SingaBackend(Backend):

    # This number indicates the onnx operator set version
    _known_opset_version = 10

    # beceuase singa's operators are different from onnx.
    # we define a dict for the name projection
    _rename_operators = {
        'Relu': 'relu',
        'Softmax': 'SoftMax',
        'Sigmoid': 'sigmoid',
        'Add': 'add',
        'MatMul': 'Matmul',
        'Conv': 'conv2d',
        'MaxPool': 'pooling_2d',
        'AveragePool': 'pooling_2d',
        'BatchNormalization': 'batchnorm_2d',
        'Concat': 'Concat',
        'Flatten': 'Flatten',
        # 'Gemm': 'GEMM',
        'Reshape': 'Reshape',
        'Sum': 'sum',
        'Cos': 'cos',
        'Cosh': 'cosh',
        'Sin': 'sin',
        'Sinh': 'sinh',
        'Tan': 'tan',
        'Tanh': 'tanh',
        'Acos': 'acos',
        'Acosh': 'acosh',
        'Asin': 'asin',
        'Asinh': 'asinh',
        'Atan': 'atan',
        'Atanh': 'atanh',
        'Selu' : 'SeLU',
        'Elu' : 'Elu',
        'Equal': 'equal',
        'Less': 'less',
    }

    # this dict indicates the operators that need extra handle
    # each indicates a function name
    _special_operators = {
        'Conv': '_create_conv',
        'MaxPool': '_create_max_avg_pool',
        'AveragePool': '_create_max_avg_pool',
        'BatchNormalization': '_create_batchnorm',
        'Concat': '_create_concat',
        'MatMul': '_create_matmul',
        'Flatten': '_create_flatten',
        # 'Gemm': '_create_gemm',
        'Reshape': '_create_reshape',
        'Softmax': '_create_softmax',
        'Selu': '_create_selu',
        'Elu': '_create_elu',
    }

    @classmethod
    def _create_equal(cls, onnx_node, inputs, opset_version):
        """
        get the equal operator from onnx node
        Args:
            onnx_node: a given onnx node
        Args:
            inputs: the input tensor
        Args:
            opset_version: the opset version
        Returns: 
            handle, the handle of singa operator
        Returns: 
            forward, the autograd of singa operator
        """
        _, forward = cls._common_onnx_node_to_singa_op(
            onnx_node, inputs, opset_version)
        return _, forward()

    @classmethod
    def _create_elu(cls, onnx_node, inputs, opset_version):
        """
        get the elu operator from onnx node
        Args:
            onnx_node: a given onnx node
        Args:
            inputs: the input tensor
        Args:
            opset_version: the opset version
        Returns: 
            handle, the handle of singa operator
        Returns: 
            forward, the autograd of singa operator
        """
        alpha = onnx_node.getattr("alpha", 1.)
        _, forward = cls._common_onnx_node_to_singa_op(
            onnx_node, inputs, opset_version)
        return _, forward(alpha)

    @classmethod
    def _create_selu(cls, onnx_node, inputs, opset_version):
        """
        get the selu operator from onnx node
        Args:
            onnx_node: a given onnx node
        Args:
            inputs: the input tensor
        Args:
            opset_version: the opset version
        Returns: 
            handle, the handle of singa operator
        Returns: 
            forward, the autograd of singa operator
        """
        alpha = onnx_node.getattr("alpha", 1.67326)
        gamma = onnx_node.getattr("gamma", 1.0507)
        _, forward = cls._common_onnx_node_to_singa_op(
            onnx_node, inputs, opset_version)
        return _, forward(alpha, gamma)

    @classmethod
    def _create_reshape(cls, onnx_node, inputs, opset_version):
        """
        get the reshape operator from onnx node
        Args:
            onnx_node: a given onnx node
        Args:
            inputs: the input tensor
        Args:
            opset_version: the opset version
        Returns: 
            the handle of singa operator
        Returns: 
            the autograd of singa operator
        """
        shape = tensor.to_numpy(inputs[1]).astype(np.int32).tolist()
        # handle the shape with -1
        hidden_shape = int(np.prod(inputs[0].shape) // np.abs(np.prod(shape)))
        shape = [s if s != -1 else hidden_shape for s in shape]
        _, forward = cls._common_onnx_node_to_singa_op(
            onnx_node, inputs, opset_version)
        return _, forward(shape)

    @classmethod
    def _create_conv(cls, onnx_node, inputs, opset_version):
        """
        get the conv operator from onnx node
        Args:
            onnx_node: a given onnx node
        Args:
            inputs: the input tensor
        Args:
            opset_version: the opset version
        Returns: 
            handle, the handle of singa operator
        Returns: 
            forward, the autograd of singa operator
        """
        kernel = tuple(onnx_node.attrs["kernel_shape"])
        # todo: we only support the padding with tuple
        padding = tuple(onnx_node.attrs["pads"][0:2]) if "pads" in onnx_node.attrs else (0, 0)
        stride = tuple(onnx_node.getattr('strides', (1, 1)))
        dilation = onnx_node.getattr('dilations', 1)
        group = onnx_node.getattr('group', 1)

        # not support dilation
        if dilation != 1:
            raise ValueError("Not implemented yet")

        # only support 2d
        if len(kernel) != 2:
            raise ValueError("Not implemented yet")

        bias = len(inputs) == 3
        x = inputs[0]
        x_shape = inputs[0].shape
        in_channels = x_shape[1]
        w_shape = inputs[1].shape
        out_channels = w_shape[0]
        assert w_shape[1] == in_channels // group

        if inputs[0].device.id() == -1:
            if group != 1:
                raise NotImplementedError
            else:
                handle = singa.ConvHandle(
                    x.data,
                    kernel,
                    stride,
                    padding,
                    in_channels,
                    out_channels,
                    bias,
                    group
                )
        else:
            handle = singa.CudnnConvHandle(
                x.data,
                kernel,
                stride,
                padding,
                in_channels,
                out_channels,
                bias,
                group
            )

        _, forward = cls._common_onnx_node_to_singa_op(onnx_node, inputs, opset_version)
        return handle, forward

    @classmethod
    def _create_max_avg_pool(cls, onnx_node, inputs, opset_version):
        """
        get the max or avg pool operator from onnx node
        Args:
            onnx_node: a given onnx node
        Args:
            inputs: the input tensor
        Args:
            opset_version: the opset version
        Returns: 
            handle, the handle of singa operator
        Returns: 
            forward, the autograd of singa operator
        """
        kernel = tuple(onnx_node.attrs["kernel_shape"])
        # todo: we only support the padding with tuple
        padding = tuple(onnx_node.attrs["pads"][0:2]) if "pads" in onnx_node.attrs else (0, 0)
        stride = tuple(onnx_node.getattr('strides', (1, 1)))
        if "auto_pad" in onnx_node.attrs:
            auto_pad = force_unicode(onnx_node.attrs['auto_pad'])
            out_shape = get_output_shape(auto_pad,  inputs[0].shape[2:], kernel, stride)
            padding = get_pad_shape(auto_pad, inputs[0].shape[2:], kernel, stride, out_shape)

        # not support count_include_pad and auto_pad
        if "count_include_pad" in onnx_node.attrs or "ceil_mode" in onnx_node.attrs:
            raise ValueError("Not implemented yet")

        # only support 2d
        if len(kernel) != 2:
            raise ValueError("Not implemented yet")

        is_max = onnx_node.op_type == 'MaxPool'
        x = inputs[0]
        if x.device.id() == -1:
            handle = singa.PoolingHandle(x.data, kernel, stride, padding, is_max)
        else:
            handle = singa.CudnnPoolingHandle(
                x.data, kernel, stride, padding, is_max
            )

        _, forward = cls._common_onnx_node_to_singa_op(onnx_node, inputs, opset_version)
        return handle, forward

    @classmethod
    def _create_batchnorm(cls, onnx_node, inputs, opset_version):
        """
        get the batch norm operator from onnx node
        Args:onnx_node: a given onnx node
        Args:inputs: the input tensor
        Args:opset_version: the opset version
        Returns: the handle of singa operator
        Returns: the autograd of singa operator
        """
        x = inputs[0]
        factor = onnx_node.getattr('momentum', 0.9)
        if x.device.id() == -1:
            raise NotImplementedError
        else:
            handle = singa.CudnnBatchNormHandle(factor, x.data)

        _, forward = cls._common_onnx_node_to_singa_op(onnx_node, inputs, opset_version)
        return handle, forward

    @classmethod
    def _create_concat(cls, onnx_node, inputs, opset_version):
        """
        get the concat operator from onnx node
        Args:
            onnx_node: a given onnx node
        Args:
            inputs: the input tensor
        Args:
            opset_version: the opset version
        Returns: 
            the handle of singa operator
        Returns: 
            the autograd of singa operator
        """
        factor = onnx_node.attrs["axis"]
        if factor < 0:
            factor = len(inputs[0].shape) + factor # in order to support the negative axis
        _, forward = cls._common_onnx_node_to_singa_op(onnx_node, inputs, opset_version)
        return None, forward(axis=factor)

    @classmethod
    def _create_softmax(cls, onnx_node, inputs, opset_version):
        """
        get the concat operator from onnx node
        Args:
            onnx_node: a given onnx node
        Args:
            inputs: the input tensor
        Args:
            opset_version: the opset version
        Returns: 
            the handle of singa operator
        Returns: 
            the autograd of singa operator
        """
        factor = onnx_node.getattr('axis', 1)
        if factor < 0:
            factor = len(inputs[0].shape) + factor # in order to support the negative axis
        # alpha = onnx_node.attrs["alpha"]
        # beta = onnx_node.attrs["beta"]
        # transA = False if onnx_node.attrs["transA"] == 0 else True
        # transB = False if onnx_node.attrs["transB"] == 0 else True
        _, forward = cls._common_onnx_node_to_singa_op(onnx_node, inputs, opset_version)
        return None, forward(axis=factor)

    # @classmethod
    # def _create_gemm(cls, onnx_node, inputs, opset_version):
    #     """
    #     get the gemm operator from onnx node
    #     Args:
    #         onnx_node: a given onnx node
    #     Args:
    #         inputs: the input tensor
    #     Args:
    #         opset_version: the opset version
    #     Returns: 
    #         the handle of singa operator
    #     Returns: 
    #         the autograd of singa operator
    #     """
    #     x = inputs[0]
    #     alpha = onnx_node.attrs["alpha"]
    #     beta = onnx_node.attrs["beta"]
    #     transA = False if onnx_node.attrs["transA"] == 0 else True
    #     transB = False if onnx_node.attrs["transB"] == 0 else True
    #     _, forward = cls._common_onnx_node_to_singa_op(onnx_node, inputs, opset_version)
    #     return None, forward(alpha=alpha, beta=beta, transA=transA, transB=transB)

    @classmethod
    def _create_flatten(cls, onnx_node, inputs, opset_version):
        """
        get the flatten operator from onnx node
        Args:
            onnx_node: a given onnx node
        Args:
            inputs: the input tensor
        Args:
            opset_version: the opset version
        Returns: 
            the handle of singa operator
        Returns: 
            the autograd of singa operator
        """
        factor = onnx_node.getattr('axis', 1)
        if factor < 0:
            factor = len(inputs[0].shape) + factor # in order to support the negative axis
        
        _, forward = cls._common_onnx_node_to_singa_op(onnx_node, inputs, opset_version)
        return None, forward(start_axis=factor)

    @classmethod
    def _create_matmul(cls, onnx_node, inputs, opset_version):
        """
        get the matmul operator from onnx node
        Args:
            onnx_node: a given onnx node
        Args:
            inputs: the input tensor
        Args:
            opset_version: the opset version
        Returns: 
            the handle of singa operator
        Returns: 
            the autograd of singa operator
        """
        _, forward = cls._common_onnx_node_to_singa_op(onnx_node, inputs, opset_version)
        return None, forward()

    @classmethod
    def _common_onnx_node_to_singa_op(cls, onnx_node, inputs, opset_version):
        """
        get a common singa operator(only autograd) from a onnx node
        other special operators also can call this func to get autograd
        Args:
            onnx_node: a given onnx node
        Args:
            tensor_map: the input tensor
        Args:
            opset_version: the opset version
        Returns: 
            a dict of tensors
        Returns: 
            a list of SingaOps('name', 'op', 'handle', 'forward')
        """
        onnx_op_type = onnx_node.op_type
        autograd_op = getattr(autograd, cls._rename_operators.get(onnx_op_type, onnx_op_type))
        return None, autograd_op

    @classmethod
    def _onnx_node_to_singa_op(cls, onnx_node, tensor_map, opset_version):
        """
        get a singa operator(handle and autograd) from a onnx node
        Args:
            onnx_node: a given onnx node
        Args:
            tensor_map: the input tensor
        Args:
            opset_version: the opset version
        Returns: 
            a dict of tensors
        Returns: 
            a list of SingaOps('name', 'op', 'handle', 'forward')
        """
        if onnx_node.op_type in cls._special_operators:
            translator = getattr(cls, cls._special_operators[onnx_node.op_type])
        else:
            translator = cls._common_onnx_node_to_singa_op
        inputs = [tensor_map[in_name] for in_name in onnx_node.inputs]
        return translator(onnx_node, inputs, opset_version)

    @classmethod
    def run_node(cls, onnx_node, inputs, opset_version=_known_opset_version):
        """
        run a single singa operator from a onnx node
        Args:
            onnx_node: a given onnx node
        Args:
            inputs: the input tensor
        Args:
            device: the used device
        Args:
            opset_version: the opset version
        Returns: 
            list, the output of the 
        """
        assert len(onnx_node.inputs) == len(inputs), "{}: expected {} but got {}".format(
            onnx_node.op_type, len(onnx_node.inputs), len(inputs))

        handle, forward = cls._onnx_node_to_singa_op(onnx_node, inputs, opset_version)
        inputs = [inputs[x] for x in onnx_node.inputs]
        return cls._run_node(onnx_node, inputs, handle, forward, opset_version)

    @classmethod
    def _run_node(cls, onnx_node, inputs, handle, forward, opset_version=_known_opset_version):
        """
        run a single singa operator from a onnx node
        Args:inputs: 
            the input tensor
        Args:handle: 
            the handle of singa operator
        Args:forward: 
            the forward of singa operator
        Args:
            opset_version: the opset version
        Returns: 
            list, the output of the
        """
        # since reshape acutally only needs one input tensor
        # but onnx regard its shape as another tensor, we need to ommit it
        if onnx_node.op_type == 'Reshape':
            inputs = [inputs[0]]
        outputs = forward(*inputs) if handle is None else forward(handle, *inputs)
        if not isinstance(outputs, collections.Iterable):
            outputs = [outputs]
        outputs_dict = collections.OrderedDict()
        for (key, val) in zip(onnx_node.outputs, outputs):
            outputs_dict[key] = val
        return outputs_dict

    @classmethod
    def _onnx_model_to_singa_net(cls, onnx_model, device, opset_version):
        """
        get all intermediate tensors and operators from onnx model
        Args:
            onnx_model: a given onnx model
        Args:
            device: the used device
        Args:
            opset_version: the opset version
        Returns:
            a dict of tensors
        Returns:
            a list of SingaOps('name', 'op', 'handle', 'forward')
        """
        # todo check the reason of Segmentation fault (core dumped)
        # optimized_model = optimizer.optimize(onnx_model)
        optimized_model = onnx_model
        # this tensor_nap contains all tensors, including outputs of each op
        tensor_map = {}
        # this weights only contains the tensors which have stored the gradients
        weights = {}
        singa_ops = []
        singa_op = collections.namedtuple('SingaOps', ['name', 'op', 'handle', 'forward'])
        # init the input as tensors
        for x in optimized_model.graph.input:
            x_shape = tuple(dim.dim_value for dim in x.type.tensor_type.shape.dim)
            # tmp_tensor = tensor.Tensor(shape=x_shape, device=device)
            tmp_tensor = tensor.from_numpy(np.zeros(x_shape, dtype=np.float32))
            tmp_tensor.to_device(device)
            tensor_map[x.name] = tmp_tensor
        # convert constant nodes to tensor, other nodes to handler
        for node in optimized_model.graph.node:
            node = OnnxNode(node)
            if node.op_type == "Constant":
                requires_grad, stores_grad = False, False
                tmp_tensor = tensor.Tensor(
                    device=device,
                    data=numpy_helper.to_array(node.attrs['value']),
                    requires_grad=requires_grad,
                    stores_grad=stores_grad,
                )
                tensor_map[node.name] = tmp_tensor
                weights[node.name] = tmp_tensor
            else:
                handle, forward = cls._onnx_node_to_singa_op(node, tensor_map, opset_version)
                singa_ops.extend([singa_op(node.name, node, handle, forward)])
                # we must know the shape of ouput
                # becasue it will become the input of next layer
                # so we need to init a new tensor with the same shape with the output
                inputs = [tensor_map[x].clone() for x in node.inputs]
                outputs = cls._run_node(node, inputs, handle, forward, opset_version)
                tensor_map.update(outputs)
        return weights, singa_ops

    @classmethod
    def prepare(cls, model, device, **kwargs):
        """
        get the batch norm operator from onnx node
        Args:
            onnx_node: a given onnx node
        Args:
            tensor_map: the input tensor
        Args:
            device: the used device
        Args:
            opset_version: the opset version
        Returns: 
            a list of output values
        """
        super(SingaBackend, cls).prepare(model, device, **kwargs)
        # check the opset version and ir version
        opset_version = None
        for imp in model.opset_import:
            if not imp.HasField("domain") or imp.domain == "":
                opset_version = imp.version
                if imp.version > cls._known_opset_version:
                    warnings.warn("This version of singa targets ONNX operator set version {}, but the model we are trying to import uses version {}.  We will try to import it anyway, but if the model uses operators which had BC-breaking changes in the intervening versions, import will fail.".format(cls._known_opset_version, imp.version))
            else:
                warnings.warn(
                    "Unrecognized operator set {}".format(imp.domain))
        if opset_version is None:
            if model.ir_version >= 0x00000003:
                raise RuntimeError(
                    "Model with IR version >= 3 did not specify ONNX operator set version (singa requires it)")
            else:
                opset_version = 1
        tensor_map, singa_ops = cls._onnx_model_to_singa_net(
            model, device, opset_version)
        return SingaRep(model, tensor_map, singa_ops)


class SingaRep(BackendRep):
    def __init__(self, model, tensor_map, singa_ops):
        """
        SingaRep provides the intermediate representation of Singa,
        the user can run the forward of the singa model by run func,
        or, the user can append more layers after the singa_ops to do
        the transfer learning
        Args:
            model: a given operator
        Args:
            tensor_map: the tensor of the operator
        Args:
            singa_ops: the tensor of the operator
        """
        super(SingaRep, self).__init__()
        self.model = model
        self.tensor_map = tensor_map
        # this each item of singa_ops is: ('name', 'op', 'handle', 'forward')
        # the name is a string, op is OnnxNode,
        # handle is Singa handle to store the tensor into singa operator
        # the forward is singa autograd operator
        self.singa_ops = singa_ops

    def run(self, inputs, **kwargs):
        """
        run the forward of singa model
        Args:
            inputs: a given operator
        Returns: 
            the onnx node
        """
        # last_layers means we run this model until the last #N layers
        last_layers = kwargs.get('last_layers', len(self.singa_ops))
        # whether return all outputs
        all_outputs = kwargs.get('all_outputs', False)
        # get a specific op by its name
        op_name = kwargs.get('op_name', None)

        # the dict will be returned
        ret_outputs = collections.OrderedDict()
        # run the handle by the order of the list(the list is Topological Sorting)
        for x, val in zip(self.model.graph.input, inputs):
            self.tensor_map[x.name] = val
        for _, op, handle, forward in self.singa_ops[:last_layers]:
            inputs = [self.tensor_map[x] for x in op.inputs]
            if op.op_type == 'Reshape':
                inputs = [inputs[0]]
            outputs = forward(*inputs) if handle is None else forward(handle, *inputs)
            if not isinstance(outputs, collections.Iterable):
                outputs = [outputs]
            for (key, val) in zip(op.outputs, outputs):
                self.tensor_map[key] = val
                ret_outputs[key] = val

        if op_name is not None:
            if op_name in outputs:
                return outputs[op_name]
            else:
                raise RuntimeError(
                    "The op_name {} does not exist, please check. The available op_names are: {}".format(
                        op_name, [val for key, val in op_name.items()]))

        # return all outputs if all_outputs==True
        # else return last outputs
        if all_outputs:
            return ret_outputs
        else:
            return outputs


run_node = SingaBackend.run_node
prepare = SingaBackend.prepare
to_onnx = SingaFrontend.singa_to_onnx_model
save = onnx.save
load = onnx.load
