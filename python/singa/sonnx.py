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

import numpy as np
import onnx.utils
import onnx
from onnx.backend.base import Backend, BackendRep
from onnx import (checker, helper, numpy_helper, GraphProto, NodeProto,
                  TensorProto, OperatorSetIdProto, optimizer)
import warnings

from . import singa_wrap as singa
from . import autograd
from . import tensor
from singa import utils

import collections
OrderedDict = collections.OrderedDict
namedtuple = collections.namedtuple


class SingaFrontend(object):
    """
    This class provides mthods to convert model from singa to onnx. 
    """

    # This number indicates the target onnx operator set version
    _target_opset_version = 11

    # beceuase singa's operators are different from onnx.
    # we define a dict for the name projection
    # "singa op name": "onnx op name"
    _rename_operators = {
        '_Conv2d': 'Conv',
        'ReLU': 'Relu',
        'MaxPool2d': 'MaxPool',
        'AvgPool2d': 'AveragePool',
        'SoftMax': 'Softmax',
        'Sigmoid': 'Sigmoid',
        'Add': 'Add',
        'Matmul': 'MatMul',
        '_BatchNorm2d': 'BatchNormalization',
        'Concat': 'Concat',
        'Flatten': 'Flatten',
        'AddBias': 'Add',
        'Gemm': 'Gemm',
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
        'SeLU': 'Selu',
        'Elu': 'Elu',
        'Equal': 'equal',
        'Less': 'Less',
        'Sign': 'Sign',
        'Div': 'Div',
        'Sub': 'Sub',
        'Sqrt': 'Sqrt',
        'Log': 'Log',
        'Greater': 'Greater',
        'HardSigmoid': 'HardSigmoid',
        'Identity': 'Identity',
        'SoftPlus': 'Softplus',
        'SoftSign': 'Softsign',
        'Mean': 'Mean',
        'Pow': 'Pow',
        'Clip': 'Clip',
        'PRelu': 'PRelu',
        'Mul': 'Mul',
        'Transpose': 'Transpose',
        'Max': 'Max',
        'Min': 'Min',
        'Shape': 'Shape',
        'And': 'And',
        'Or': 'Or',
        'Xor': 'Xor',
        'Not': 'Not',
        'Negative': 'Neg',
        'Reciprocal': 'Reciprocal',
        'ConstantOfShape': 'ConstantOfShape',
        'Dropout': 'Dropout',
        'ReduceSum': 'ReduceSum',
        'ReduceMean': 'ReduceMean',
        'LeakyRelu': 'LeakyRelu',
        'GlobalAveragePool': 'GlobalAveragePool',
        'Squeeze': 'Squeeze',
        'Unsqueeze': 'Unsqueeze',
        'Slice': 'Slice',
        'Ceil': 'Ceil',
        'Split': 'Split',
        'Gather': 'Gather',
        'Tile': 'Tile',
        'NonZero': 'NonZero',
        'Cast': 'Cast',
        'OneHot': 'OneHot',
    }

    # this dict indicates the operators that need extra handle
    # each indicates a function name
    _special_operators = {
        '_Conv2d': '_create_conv_pool',
        '_Pooling2d': '_create_conv_pool',
        '_BatchNorm2d': '_create_batchnorm',
        'Concat': '_create_concat',
        'Flatten': '_create_flatten',
        'Gemm': '_create_gemm',
        'Reshape': '_create_reshape',
        'SoftMax': '_create_softmax',
        'SeLU': '_create_selu',
        'Elu': '_create_elu',
        'HardSigmoid': '_create_hardsigmoid',
        'Clip': '_create_clip',
        'Transpose': '_create_transpose',
        'ConstantOfShape': '_create_constantOfShape',
        'Dropout': '_create_dropout',
        'ReduceSum': '_create_reduceOp',
        'ReduceMean': '_create_reduceOp',
        'Squeeze': '_create_squeeze',
        'Unsqueeze': '_create_squeeze',
        'Slice': '_create_slice',
        'Split': '_create_split',
        'Gather': '_create_gather',
        'Tile': '_create_tile',
        'Cast': '_create_cast',
        'OneHot': '_create_onehot',
    }

    # operators with bool output
    _bool_operators = {
        'Equal': TensorProto.BOOL,
        'Greater': TensorProto.BOOL,
        'Less': TensorProto.BOOL,
        'And': TensorProto.BOOL,
        'Not': TensorProto.BOOL,
        'Or': TensorProto.BOOL,
        'Xor': TensorProto.BOOL,
        'Shape': TensorProto.INT64,
        'NonZero': TensorProto.INT64,
    }

    # some ops(such as batchnorm) has inputs we cannot handle directly,
    # so we record these items firstly so that we can handle then
    # at other place.
    _unhandled_operators = {
        "_BatchNorm2d": "_special_handle_batchnorm",
        "Reshape": "_special_handle_reshape",
        "Clip": "_special_handle_clip",
        "Slice": "_special_handle_slice",
        "Gather": "_special_handle_gather",
        "Tile": "_special_handle_tile",
        "OneHot": "_special_handle_onehot",
    }

    @classmethod
    def _create_onehot(cls, op, op_t):
        """
        get a onnx node from singa onthot
        Args:
            op: a given operator
        Args:
            op_t: the tensor of the operator
        Returns: 
            the onnx node
        """
        node = cls._common_singa_tensor_to_onnx_node(op, op_t)
        # axis, indices, depth, values
        node.attribute.extend([
            helper.make_attribute('axis', op.axis),
        ])
        for attr in ['depth', 'values']:
            node.input.append(op.name + ":" + attr)
        return node

    @classmethod
    def _create_cast(cls, op, op_t):
        """
        get a onnx node from singa cast
        Args:
            op: a given operator
        Args:
            op_t: the tensor of the operator
        Returns: 
            the onnx node
        """
        node = cls._common_singa_tensor_to_onnx_node(op, op_t)

        map_dict = {
            tensor.float32: TensorProto.FLOAT,  # FLOAT to float32
            tensor.int32: TensorProto.INT32,  # INT32 to int32
        }
        node.attribute.extend([
            helper.make_attribute('to', map_dict[op.to]),
        ])
        return node

    @classmethod
    def _create_tile(cls, op, op_t):
        """
        get a onnx node from singa tile
        Args:
            op: a given operator
        Args:
            op_t: the tensor of the operator
        Returns: 
            the onnx node
        """
        node = cls._common_singa_tensor_to_onnx_node(op, op_t)

        node.input.append(op.name + ":repeats")
        return node

    @classmethod
    def _create_gather(cls, op, op_t):
        """
        get a onnx node from singa gather
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
        node.input.append(op.name + ":indices")
        return node

    @classmethod
    def _create_split(cls, op, op_t):
        """
        get a onnx node from singa split
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
            helper.make_attribute('split', op.parts),
        ])
        return node

    @classmethod
    def _create_slice(cls, op, op_t):
        """
        get a onnx node from singa slice
        Args:
            op: a given operator
        Args:
            op_t: the tensor of the operator
        Returns: 
            the onnx node
        """
        node = cls._common_singa_tensor_to_onnx_node(op, op_t)
        for attr in ['starts', 'ends', 'axes', 'steps']:
            node.input.append(op.name + ":" + attr)
        return node

    @classmethod
    def _create_squeeze(cls, op, op_t):
        """
        get a onnx node from singa squeeze and unsqueeze
        Args:
            op: a given operator
        Args:
            op_t: the tensor of the operator
        Returns: 
            the onnx node
        """
        node = cls._common_singa_tensor_to_onnx_node(op, op_t)

        node.attribute.extend([
            helper.make_attribute('axes', list(op.axis)),
        ])
        return node

    @classmethod
    def _create_reduceOp(cls, op, op_t):
        """
        get a onnx node from singa ReduceSum, ReduceMean, ReduceMax, ReduceMin, etc.
        Args:
            op: a given operator
        Args:
            op_t: the tensor of the operator
        Returns: 
            the onnx node
        """
        node = cls._common_singa_tensor_to_onnx_node(op, op_t)

        node.attribute.extend([
            helper.make_attribute('axes', list(op.axes)),
            helper.make_attribute('keepdims', op.keepdims),
        ])
        return node

    @classmethod
    def _create_dropout(cls, op, op_t):
        """
        get a onnx node from singa Dropout operator
        Args:
            op: a given operator
        Args:
            op_t: the tensor of the operator
        Returns: 
            the onnx node
        """
        node = cls._common_singa_tensor_to_onnx_node(op, op_t)

        node.attribute.extend([
            helper.make_attribute('ratio', op.ratio),
        ])
        return node

    @classmethod
    def _create_constantOfShape(cls, op, op_t):
        """
        get a onnx node from singa ConstantOfShape operator
        Args:
            op: a given operator
        Args:
            op_t: the tensor of the operator
        Returns: 
            the onnx node
        """
        node = cls._common_singa_tensor_to_onnx_node(op, op_t)
        tensor_type = onnx.TensorProto.FLOAT if isinstance(
            op.value, float) else onnx.TensorProto.INT32
        tensor_value = onnx.helper.make_tensor("value", tensor_type, [1],
                                               [op.value])
        node.attribute.extend([
            helper.make_attribute('value', tensor_value),
        ])
        return node

    @classmethod
    def _create_transpose(cls, op, op_t):
        """
        get a onnx node from singa Transpose operator
        Args:
            op: a given operator
        Args:
            op_t: the tensor of the operator
        Returns: 
            the onnx node
        """
        node = cls._common_singa_tensor_to_onnx_node(op, op_t)

        node.attribute.extend([
            helper.make_attribute('perm', op.perm),
        ])
        return node

    @classmethod
    def _create_clip(cls, op, op_t):
        """
        get a onnx node from singa clip operator
        Args:
            op: a given operator
        Args:
            op_t: the tensor of the operator
        Returns: 
            the onnx node
        """
        node = cls._common_singa_tensor_to_onnx_node(op, op_t)
        if op.min is not None:
            node.input.append(op.name + ":min")
        else:
            node.input.append("")
        if op.max is not None:
            node.input.append(op.name + ":max")
        else:
            node.input.append("")
        return node

    @classmethod
    def _create_hardsigmoid(cls, op, op_t):
        """
        get a onnx node from singa HardSigmoid operator
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
            helper.make_attribute('beta', op.gamma),
        ])
        return node

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
        shape_node_name = op.name + ":shape"
        node = cls._common_singa_tensor_to_onnx_node(op, op_t)
        node.input.extend([shape_node_name])
        return node

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
            helper.make_attribute('axis', op.axis),
        ])
        return node

    @classmethod
    def _create_gemm(cls, op, op_t):
        """
        get a onnx node from singa gemm operator
        Args:
            op: a given operator
        Args:
            op_t: the tensor of the operator
        Returns: 
            the onnx node
        """
        node = cls._common_singa_tensor_to_onnx_node(op, op_t)

        node.attribute.extend([
            helper.make_attribute('alpha', float(op.alpha)),
            helper.make_attribute('beta', float(op.beta)),
            helper.make_attribute('transA', op.transA),
            helper.make_attribute('transB', op.transB),
        ])

        return node

    @classmethod
    def _create_batchnorm(cls, op, op_t):
        """
        get a onnx node from singa _BatchNorm2d operator
        Args:
            op: a given operator
        Args:
            op_t: the tensor of the operator
        Returns: 
            the onnx node
        """
        # first, we init batchnorm node
        epsilon = 1e-5  # the epsilon value used in singa
        bn_node = cls._common_singa_tensor_to_onnx_node(op, op_t)
        bn_node.attribute.extend([
            helper.make_attribute('momentum', op.handle.factor),
            helper.make_attribute('epsilon', epsilon),
        ])
        # then we add nodes of scal, bias, mean, var
        nodes = []
        running_values = {"mean": op.running_mean, "var": op.running_var}
        for tmp_name, running_value in running_values.items():
            node_name = op.name + ":" + tmp_name
            bn_node.input.append(node_name)

        nodes.append(bn_node)
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
        oddp = op.odd_padding
        p = [
            op.handle.pad_h + oddp[0],
            op.handle.pad_w + oddp[1],
            op.handle.pad_w + oddp[2],
            op.handle.pad_h + oddp[3],
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
                helper.make_attribute('auto_pad', 'NOTSET'),
            ])

        elif op.handle.is_max_pooling:
            node.op_type = cls._rename_operators.get('MaxPool2d')
        else:
            node.op_type = cls._rename_operators.get('AvgPool2d')
        return node

    @classmethod
    def _get_singa_op_inputs_outputs(cls, op):
        """
        get inputs and outputs from a given operator
        Args:
            op: a given operator
        Returns: 
            inputs and outputs of the op
        """
        outputs = [op.output_name(idx) for _, idx in op.y_id2idx.items()]
        inputs = [
            srcop.output_name(srcop.y_id2idx[yid])
            for (srcop, yid, _, _) in op.src
        ]
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

    @classmethod
    def _special_handle_batchnorm(cls, op, X, W):
        """
        hanlde the special operators
        Args:
            op: a given operator
        Args:
            op_t: the tensor of the operator
        Returns: 
            onnx tensor list
        """
        # for singa, x, scale, bias is input
        # and mean and var is attribute
        # so we add the mean and var to W
        tensor_list = []
        append_inputs = {"mean": op.running_mean, "var": op.running_var}
        for tmp_name, append_input in append_inputs.items():
            node_name = op.name + ":" + tmp_name
            append_input = tensor.to_numpy(tensor.from_raw_tensor(append_input))
            tensor_list.append(numpy_helper.from_array(append_input, node_name))
        return tensor_list

    @classmethod
    def _special_handle_reshape(cls, op, X, W):
        """
        hanlde the special operators
        Args:
            op: a given operator
        Args:
            op_t: the tensor of the operator
        Returns: 
            onnx tensor list
        """
        node_name = op.name + ":shape"
        return [
            numpy_helper.from_array(np.array(op.shape, dtype=np.int64),
                                    node_name)
        ]

    @classmethod
    def _special_handle_clip(cls, op, X, W):
        """
        hanlde the special operators
        Args:
            op: a given operator
        Args:
            op_t: the tensor of the operator
        Returns: 
            onnx tensor list
        """
        tensor_list = []
        # clip add min and max
        append_inputs = {"min": op.min, "max": op.max}
        for tmp_name, append_input in append_inputs.items():
            node_name = op.name + ":" + tmp_name
            tensor_list.append(
                helper.make_tensor(node_name, TensorProto.FLOAT, [],
                                   [append_input]))
        return tensor_list

    @classmethod
    def _special_handle_slice(cls, op, X, W):
        """
        hanlde the special operators
        Args:
            op: a given operator
        Args:
            op_t: the tensor of the operator
        Returns: 
            onnx tensor list
        """
        tensor_list = []
        # slice add starts, ends, axes, steps
        append_inputs = {
            "starts": op.starts,
            "ends": op.ends,
            "axes": op.axes,
            "steps": op.steps,
        }
        for tmp_name, append_input in append_inputs.items():
            node_name = op.name + ":" + tmp_name
            tensor_list.append(
                numpy_helper.from_array(np.array(append_input), node_name))
        return tensor_list

    @classmethod
    def _special_handle_gather(cls, op, X, W):
        """
        hanlde the special operators
        Args:
            op: a given operator
        Args:
            op_t: the tensor of the operator
        Returns: 
            onnx tensor list
        """
        tensor_list = []
        append_inputs = {
            "indices": op.indices,
        }
        for tmp_name, append_input in append_inputs.items():
            node_name = op.name + ":" + tmp_name
            tensor_list.append(
                numpy_helper.from_array(np.array(append_input), node_name))
        return tensor_list

    @classmethod
    def _special_handle_tile(cls, op, X, W):
        """
        hanlde the special operators
        Args:
            op: a given operator
        Args:
            op_t: the tensor of the operator
        Returns: 
            onnx tensor list
        """
        tensor_list = []
        append_inputs = {
            "repeats": op.repeats,
        }
        for tmp_name, append_input in append_inputs.items():
            node_name = op.name + ":" + tmp_name
            tensor_list.append(
                numpy_helper.from_array(np.array(append_input), node_name))
        return tensor_list

    @classmethod
    def _special_handle_onehot(cls, op, X, W):
        """
        hanlde the special operators
        Args:
            op: a given operator
        Args:
            op_t: the tensor of the operator
        Returns: 
            onnx tensor list
        """
        tensor_list = []
        append_inputs = {
            "depth": op.depth,
            "values": op.values,
        }
        for tmp_name, append_input in append_inputs.items():
            node_name = op.name + ":" + tmp_name
            tensor_list.append(
                numpy_helper.from_array(np.array(append_input), node_name))
        return tensor_list

    @classmethod
    def handle_special_ops(cls, op, X, W):
        """
        hanlde the special operators, 
        because the inputs of batchnorm and reshape are differnet with onnx
        we need to add these inputs into onnx model mannully
        Args:
            op: a given operator
        Args:
            X: onnx input list
        Args:
            X: onnx weight list
        Returns: the onnx node
        """
        optype = cls._get_singa_op_type(op)
        translator = getattr(cls, cls._unhandled_operators[optype])
        tensor_list = translator(op, X, W)
        for tensor in tensor_list:
            X.append(
                helper.make_tensor_value_info(tensor.name, tensor.data_type,
                                              tensor.dims))
            W.append(tensor)
        # return X, W

    @classmethod
    def _common_singa_tensor_to_onnx_node(cls, op, op_t):
        """
        get a onnx node from a singa operator, prepare its type, inputs and outputs
        Args:
            op: a given operator
        Args:
            op_t: the tensor of the operator
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
        assert len(
            y
        ) == 1, "Not support multiple output now."  # assume there is only one output
        y = y[0]

        graph_def = GraphProto()
        graph_def.name = model_name
        topol, ws, ins = utils.post_order_recursive(y.creator, y)

        # prepare the input
        X = []
        for op_name, op_t in ins.items():
            op_t = inputs.pop(0)
            dtype = TensorProto.INT32 if op_t.dtype == tensor.int32 else TensorProto.FLOAT
            X.append(helper.make_tensor_value_info(op_name, dtype, op_t.shape))

        # prepare the output
        y_optype = cls._get_singa_op_type(y.creator)
        if y_optype in cls._bool_operators:
            y_dtype = cls._bool_operators[y_optype]
        elif y.dtype == tensor.int32:
            y_dtype = TensorProto.INT32
        else:
            y_dtype = TensorProto.FLOAT
        Y = [helper.make_tensor_value_info(y.name, y_dtype, y.shape)]

        # prepare the weight
        W = []
        for op_name, op_t in ws.items():
            dtype = TensorProto.INT32 if op_t.dtype == tensor.int32 else TensorProto.FLOAT
            wt = tensor.to_numpy(op_t)
            wt = numpy_helper.from_array(wt)
            wt.name = op_name
            W.append(wt)
            X.append(helper.make_tensor_value_info(op_name, dtype, op_t.shape))

        # iterate the node graph
        for op_name, op in topol.items():
            optype = cls._get_singa_op_type(op)
            if optype in cls._unhandled_operators:
                cls.handle_special_ops(op, X, W)
            graph_def.node.extend(cls.singa_op_to_onnx_node(op, op_t))

        graph_def.input.extend(X)
        graph_def.output.extend(Y)
        graph_def.initializer.extend(W)
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
        model = helper.make_model(cls.singa_to_onnx_graph(inputs,
                                                          y,
                                                          model_name="sonnx"),
                                  producer_name='sonnx',
                                  opset_imports=[opset_id])
        model = optimizer.optimize(model)
        checker.check_model(model)
        return model


class OnnxNode(object):
    """
    Reimplementation of NodeProto from ONNX, but in a form
    more convenient to work with from Python.
    """

    def __init__(self, node):
        self.name = str(node.name)
        self.op_type = str(node.op_type)
        self.attrs = OnnxAttributes.from_onnx(node.attribute)
        # there may some inputs which we regard as attribute, so we mark them there
        self.consumed_inputs = list()
        self.inputs = list(node.input)
        self.outputs = list(node.output)

    def getattr(self, key, default=None):
        return self.attrs[key] if key in self.attrs else default


class OnnxAttributes(dict):
    """
    This is a more convenient way to work with ONNX attributes
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
    _known_opset_version = 11

    # beceuase singa's operators are different from onnx.
    # we define a dict for the name projection
    _rename_operators = {
        'Relu': 'relu',
        'Softmax': 'SoftMax',
        'Sigmoid': 'sigmoid',
        'Add': 'add',
        'MatMul': 'matmul',
        'Conv': '_Conv2d',
        'MaxPool': '_Pooling2d',
        'AveragePool': '_Pooling2d',
        'BatchNormalization': 'batchnorm_2d',
        'Concat': 'Concat',
        'Flatten': 'Flatten',
        'Gemm': 'Gemm',
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
        'Selu': 'SeLU',
        'Elu': 'Elu',
        'Equal': 'equal',
        'Less': 'less',
        'Sign': 'sign',
        'Div': 'div',
        'Sub': 'sub',
        'Sqrt': 'sqrt',
        'Log': 'log',
        'Greater': 'greater',
        'HardSigmoid': 'HardSigmoid',
        'Identity': 'identity',
        'Softplus': 'softplus',
        'Softsign': 'softsign',
        'Mean': 'mean',
        'Pow': 'pow',
        'Clip': 'Clip',
        'PRelu': 'prelu',
        'Mul': 'mul',
        'Transpose': 'Transpose',
        'Max': 'max',
        'Min': 'min',
        'Shape': 'shape',
        'And': '_and',
        'Or': '_or',
        'Xor': '_xor',
        'Not': '_not',
        'Neg': 'negative',
        'Reciprocal': 'reciprocal',
        'ConstantOfShape': 'ConstantOfShape',
        'Dropout': 'Dropout',
        'ReduceSum': 'ReduceSum',
        'ReduceMean': 'ReduceMean',
        'LeakyRelu': 'LeakyRelu',
        'GlobalAveragePool': 'GlobalAveragePool',
        'Squeeze': 'Squeeze',
        'Unsqueeze': 'Unsqueeze',
        'Slice': 'Slice',
        'Ceil': 'Ceil',
        'Split': 'Split',
        'Gather': 'Gather',
        'Tile': 'Tile',
        'NonZero': 'nonzero',
        'Cast': 'Cast',
        'OneHot': 'OneHot',
    }

    # this dict indicates the operators that need extra handle
    # each indicates a function name
    _special_operators = {
        'Conv': '_create_conv',
        'MaxPool': '_create_max_avg_pool',
        'AveragePool': '_create_max_avg_pool',
        'BatchNormalization': '_create_batchnorm',
        'Concat': '_create_concat',
        'Flatten': '_create_flatten',
        'Gemm': '_create_gemm',
        'Reshape': '_create_reshape',
        'Softmax': '_create_softmax',
        'Selu': '_create_selu',
        'Elu': '_create_elu',
        'HardSigmoid': '_create_hardsigmoid',
        'Clip': '_create_clip',
        'Transpose': '_create_transpose',
        'ConstantOfShape': '_create_constantOfShape',
        'Dropout': '_create_dropout',
        'ReduceSum': '_create_reduceOp',
        'ReduceMean': '_create_reduceOp',
        'LeakyRelu': '_create_leakyrelu',
        'GlobalAveragePool': '_create_globalaveragepool',
        'Squeeze': '_create_squeeze',
        'Unsqueeze': '_create_squeeze',
        'Slice': '_create_slice',
        'Split': '_create_split',
        'Gather': '_create_gather',
        'Tile': '_create_tile',
        'Cast': '_create_cast',
        'OneHot': '_create_onehot',
        'Constant': "_create_constant"
    }

    @classmethod
    def _create_constant(cls, onnx_node, inputs, opset_version):
        """
        parse onnx constatn node to weights
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
        tmp_tensor = onnx_node.getattr('value')
        np_dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[tmp_tensor.data_type]
        np_tensor = np.frombuffer(tmp_tensor.raw_data, dtype=np_dtype)
        if np_tensor.dtype == "int64":
            np_tensor = np_tensor.astype(np.int32)
        # todo, we cannot support scalar tensor
        if np.ndim(np_tensor) == 0:
            np_tensor = np.array(np_tensor, ndmin=1)
        return None, np_tensor

    @classmethod
    def _create_onehot(cls, onnx_node, inputs, opset_version):
        """
        get the OneHot operator from onnx node
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
        axis = onnx_node.getattr("axis", -1)
        # we move several inputs to singa's attribuates
        # and mark them so we don't use them when we run this operator
        depth = tensor.to_numpy(inputs.pop(1)).astype(np.int32)
        value = tensor.to_numpy(inputs.pop(1))
        onnx_node.consumed_inputs.extend(onnx_node.inputs[1:])
        _, forward = cls._common_onnx_node_to_singa_op(onnx_node, inputs,
                                                       opset_version)
        return _, forward(axis, depth, value)

    @classmethod
    def _create_cast(cls, onnx_node, inputs, opset_version):
        """
        get the Cast operator from onnx node
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
        to = onnx_node.getattr("to")
        # singa only supports float32 and int32
        map_dict = {
            TensorProto.FLOAT: tensor.float32,  # FLOAT to float32
            TensorProto.UINT8: None,  # UINT8
            TensorProto.INT8: tensor.int32,  # INT8 to int32
            TensorProto.UINT16: None,  # UINT16
            TensorProto.INT16: tensor.int32,  # INT16 to int32
            TensorProto.INT32: tensor.int32,  # INT32 to int32
            TensorProto.INT64: tensor.int32,  # INT64 to int32
            TensorProto.STRING: None,  # stirng
            TensorProto.BOOL: None,  # bool
        }
        to = map_dict[to]
        assert to != None, "not support cast type: {}".format(to)
        _, forward = cls._common_onnx_node_to_singa_op(onnx_node, inputs,
                                                       opset_version)
        return _, forward(to)

    @classmethod
    def _create_tile(cls, onnx_node, inputs, opset_version):
        """
        get the Tile operator from onnx node
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
        # we move several inputs to singa's attribuates
        # and mark them so we don't use them when we run this operator
        repeats = tensor.to_numpy(inputs.pop(1)).astype(np.int32).tolist()
        onnx_node.consumed_inputs.append(onnx_node.inputs[1])
        _, forward = cls._common_onnx_node_to_singa_op(onnx_node, inputs,
                                                       opset_version)
        return _, forward(repeats)

    @classmethod
    def _create_gather(cls, onnx_node, inputs, opset_version):
        """
        get the Gather operator from onnx node
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
        axis = onnx_node.getattr("axis", 0)
        # we move several inputs to singa's attribuates
        # and mark them so we don't use them when we run this operator
        indices = tensor.to_numpy(inputs.pop(1)).astype(np.int32).tolist()
        onnx_node.consumed_inputs.append(onnx_node.inputs[1])
        _, forward = cls._common_onnx_node_to_singa_op(onnx_node, inputs,
                                                       opset_version)
        return _, forward(axis, indices)

    @classmethod
    def _create_split(cls, onnx_node, inputs, opset_version):
        """
        get the Split operator from onnx node
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
        axis = onnx_node.getattr("axis", 0)
        split = onnx_node.getattr("split", None)
        num_output = len(onnx_node.outputs)
        _, forward = cls._common_onnx_node_to_singa_op(onnx_node, inputs,
                                                       opset_version)
        return _, forward(axis, split, num_output)

    @classmethod
    def _create_slice(cls, onnx_node, inputs, opset_version):
        """
        get the Slice operator from onnx node
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
        # we move several inputs to singa's attribuates
        # and mark them so we don't use them when we run this operator
        starts = tensor.to_numpy(inputs.pop(1)).astype(np.int32).tolist()
        ends = tensor.to_numpy(inputs.pop(1)).astype(np.int32).tolist()
        # sometime onnx may ignore these two inputs, axes and step
        if len(inputs) >= 2 and onnx_node.inputs[3] != '':
            axes = tensor.to_numpy(inputs.pop(1)).astype(np.int32).tolist()
        else:
            axes = None
        steps = tensor.to_numpy(inputs.pop(1)).astype(
            np.int32).tolist() if len(inputs) >= 2 else None
        onnx_node.consumed_inputs.extend(onnx_node.inputs[1:])
        _, forward = cls._common_onnx_node_to_singa_op(onnx_node, inputs,
                                                       opset_version)
        return _, forward(starts, ends, axes, steps)

    @classmethod
    def _create_squeeze(cls, onnx_node, inputs, opset_version):
        """
        get the Squeeze and Unsqueeze operator from onnx node
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
        axes = onnx_node.getattr("axes")
        _, forward = cls._common_onnx_node_to_singa_op(onnx_node, inputs,
                                                       opset_version)
        return _, forward(axes)

    @classmethod
    def _create_globalaveragepool(cls, onnx_node, inputs, opset_version):
        """
        get the GlobalAveragePool operator from onnx node
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
        data_format = onnx_node.getattr("data_format", 'channels_first')
        _, forward = cls._common_onnx_node_to_singa_op(onnx_node, inputs,
                                                       opset_version)
        return _, forward(data_format)

    @classmethod
    def _create_leakyrelu(cls, onnx_node, inputs, opset_version):
        """
        get the LeakyRelu operator from onnx node
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
        alpha = onnx_node.getattr("alpha", 0.01)
        _, forward = cls._common_onnx_node_to_singa_op(onnx_node, inputs,
                                                       opset_version)
        return _, forward(alpha)

    @classmethod
    def _create_reduceOp(cls, onnx_node, inputs, opset_version):
        """
        get the ReduceSum, ReduceMean, ReduceMax, ReduceMin, etc, operator from onnx node
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
        axes = onnx_node.getattr("axes", None)
        keepdims = onnx_node.getattr("keepdims", 1)
        _, forward = cls._common_onnx_node_to_singa_op(onnx_node, inputs,
                                                       opset_version)
        return _, forward(axes, keepdims)

    @classmethod
    def _create_dropout(cls, onnx_node, inputs, opset_version):
        """
        get the Dropout operator from onnx node
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
        ratio = onnx_node.getattr("ratio", 0)
        _, forward = cls._common_onnx_node_to_singa_op(onnx_node, inputs,
                                                       opset_version)
        return _, forward(ratio)

    @classmethod
    def _create_constantOfShape(cls, onnx_node, inputs, opset_version):
        """
        get the ConstantOfShape operator from onnx node
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
        value = onnx_node.getattr("value", 0)
        if isinstance(value, onnx.TensorProto):
            value = numpy_helper.to_array(value)[0].item()
        _, forward = cls._common_onnx_node_to_singa_op(onnx_node, inputs,
                                                       opset_version)
        return _, forward(value)

    @classmethod
    def _create_transpose(cls, onnx_node, inputs, opset_version):
        """
        get the Transpose operator from onnx node
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
        shape = inputs[0].shape
        perm = onnx_node.getattr("perm", list(range(len(shape) - 1, -1, -1)))
        _, forward = cls._common_onnx_node_to_singa_op(onnx_node, inputs,
                                                       opset_version)
        return _, forward(perm)

    @classmethod
    def _create_clip(cls, onnx_node, inputs, opset_version):
        """
        get the clip operator from onnx node
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
        # sometime onnx may ignore these two inputs, min or max or both
        if len(inputs) >= 2 and onnx_node.inputs[1] != '':
            min_v = tensor.to_numpy(inputs.pop(1)).tolist()[0]
        else:
            min_v = None
        if len(inputs) >= 2 and onnx_node.inputs[2] != '':
            max_v = tensor.to_numpy(inputs.pop(1)).tolist()[0]
        else:
            max_v = None
        onnx_node.consumed_inputs.extend(onnx_node.inputs[1:])
        _, forward = cls._common_onnx_node_to_singa_op(onnx_node, inputs,
                                                       opset_version)
        return _, forward(min_v, max_v)

    @classmethod
    def _create_hardsigmoid(cls, onnx_node, inputs, opset_version):
        """
        get the HardSigmoid operator from onnx node
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
        alpha = onnx_node.getattr("alpha", 0.2)
        beta = onnx_node.getattr("beta", 0.5)
        _, forward = cls._common_onnx_node_to_singa_op(onnx_node, inputs,
                                                       opset_version)
        return _, forward(alpha, beta)

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
        _, forward = cls._common_onnx_node_to_singa_op(onnx_node, inputs,
                                                       opset_version)
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
        _, forward = cls._common_onnx_node_to_singa_op(onnx_node, inputs,
                                                       opset_version)
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
        shape = tensor.to_numpy(inputs.pop(1)).astype(np.int32).tolist()
        onnx_node.consumed_inputs.append(onnx_node.inputs[1])
        _, forward = cls._common_onnx_node_to_singa_op(onnx_node, inputs,
                                                       opset_version)
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
        padding = tuple(
            onnx_node.attrs["pads"]) if "pads" in onnx_node.attrs else (0, 0)
        stride = tuple(onnx_node.getattr('strides', (1, 1)))
        # default the odd_padding is 0, once there are same pad mode, we modify it
        # for odd_padding, please refer the autegrade.py
        odd_padding = (0, 0, 0, 0)
        if "auto_pad" in onnx_node.attrs:
            auto_pad = utils.force_unicode(onnx_node.attrs['auto_pad'])
            if auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
                padding, odd_padding = utils.get_padding_shape(
                    auto_pad, inputs[0].shape[2:], kernel, stride)

        # not support dilation
        dilation = onnx_node.getattr('dilations', 1)
        if dilation != 1 and list(dilation) != [1, 1]:
            raise ValueError("Not implemented yet for dilation")
        group = onnx_node.getattr('group', 1)

        # only support 1d or 2d
        if len(kernel) > 2:
            raise ValueError("Only implemented for 1d or 2d")

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
                handle = singa.ConvHandle(x.data, kernel, stride, padding,
                                          in_channels, out_channels, bias,
                                          group)
        else:
            handle = singa.CudnnConvHandle(x.data, kernel, stride, padding,
                                           in_channels, out_channels, bias,
                                           group)

        _, forward = cls._common_onnx_node_to_singa_op(onnx_node, inputs,
                                                       opset_version)
        return _, forward(handle, odd_padding)

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
        padding = tuple(
            onnx_node.attrs["pads"]) if "pads" in onnx_node.attrs else (0, 0)
        stride = tuple(onnx_node.getattr('strides', (1, 1)))
        # default the odd_padding is 0, once there are same pad mode, we modify it
        # for odd_padding, please refer the autegrade.py
        odd_padding = (0, 0, 0, 0)
        if "auto_pad" in onnx_node.attrs:
            auto_pad = utils.force_unicode(onnx_node.attrs['auto_pad'])
            if auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
                padding, odd_padding = utils.get_padding_shape(
                    auto_pad, inputs[0].shape[2:], kernel, stride)

        # not support count_include_pad and auto_pad
        if "count_include_pad" in onnx_node.attrs or "ceil_mode" in onnx_node.attrs:
            raise ValueError(
                "Not implemented yet for count_include_pad or ceil_mode")

        # only support 2d
        if len(kernel) != 2:
            raise ValueError("Not implemented yet")

        is_max = onnx_node.op_type == 'MaxPool'
        x = inputs[0]
        if x.device.id() == -1:
            handle = singa.PoolingHandle(x.data, kernel, stride, padding,
                                         is_max)
        else:
            handle = singa.CudnnPoolingHandle(x.data, kernel, stride, padding,
                                              is_max)

        _, forward = cls._common_onnx_node_to_singa_op(onnx_node, inputs,
                                                       opset_version)
        return _, forward(handle, odd_padding)

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
            handle = singa.BatchNormHandle(factor, x.data)
        else:
            handle = singa.CudnnBatchNormHandle(factor, x.data)

        _, forward = cls._common_onnx_node_to_singa_op(onnx_node, inputs,
                                                       opset_version)
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
            factor = len(inputs[0].shape
                        ) + factor  # in order to support the negative axis
        _, forward = cls._common_onnx_node_to_singa_op(onnx_node, inputs,
                                                       opset_version)
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
            # in order to support the negative axis
            factor = len(inputs[0].shape) + factor
        _, forward = cls._common_onnx_node_to_singa_op(onnx_node, inputs,
                                                       opset_version)
        return None, forward(axis=factor)

    @classmethod
    def _create_gemm(cls, onnx_node, inputs, opset_version):
        """
        get the gemm operator from onnx node
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
        x = inputs[0]
        alpha = onnx_node.getattr('alpha', 1.)
        beta = onnx_node.getattr('beta', 1.)
        transA = onnx_node.getattr('transA', 0)
        transB = onnx_node.getattr('transB', 0)
        _, forward = cls._common_onnx_node_to_singa_op(onnx_node, inputs,
                                                       opset_version)
        return None, forward(alpha=alpha,
                             beta=beta,
                             transA=transA,
                             transB=transB)

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
            # in order to support the negative axis
            factor = len(inputs[0].shape) + factor

        _, forward = cls._common_onnx_node_to_singa_op(onnx_node, inputs,
                                                       opset_version)
        return None, forward(axis=factor)

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
        assert onnx_op_type in cls._rename_operators, "not support operator: {}".format(
            onnx_op_type)
        autograd_op = getattr(autograd, cls._rename_operators[onnx_op_type])
        return None, autograd_op

    @classmethod
    def _onnx_node_to_singa_op(cls,
                               onnx_node,
                               inputs,
                               opset_version=_known_opset_version):
        """
        get a singa operator(handle and autograd) from a onnx node
        Args:
            onnx_node: a given onnx node
        Args:
            inputs: the input list
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
        valid_inputs = [x for x in onnx_node.inputs if x != ""]
        assert len(valid_inputs) == len(
            inputs), "{}: expected {} but got {}".format(
                onnx_node.op_type, len(valid_inputs), len(inputs))

        tmp_inputs = [inputs[x] for x in onnx_node.inputs if x != ""]
        handle, forward = cls._onnx_node_to_singa_op(onnx_node, tmp_inputs,
                                                     opset_version)
        # only give the inputs it needs
        # consumed_inputs are the inputs marked as attributes
        # so we remove it here
        tmp_inputs = [
            inputs[x]
            for x in onnx_node.inputs
            if x not in onnx_node.consumed_inputs
        ]
        return cls._run_node(onnx_node, tmp_inputs, handle, forward,
                             opset_version)

    @classmethod
    def _run_node(cls,
                  onnx_node,
                  inputs,
                  handle,
                  forward,
                  opset_version=_known_opset_version):
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
        outputs = forward(*inputs) if handle is None else forward(
            handle, *inputs)
        if not isinstance(outputs, collections.Iterable):
            outputs = [outputs]
        outputs_dict = OrderedDict()
        for (key, val) in zip(onnx_node.outputs, outputs):
            outputs_dict[key] = val
        return outputs_dict

    @classmethod
    def _init_graph_parameter(cls, graph, init_inputs, device):
        """
        init the singa tensor from onnx infos
        Args:
            graph: a given onnx graph
        Args:
            init_inputs: a list of inputs, which used to init the operators
        Args:
            device: the used device
        Returns:
            a dict of tensors
        """
        tensor_map = {}
        # due to https://github.com/onnx/onnx/issues/2417
        # sometimes, input contains all initializer's info
        # sometimes, may not
        all_inputs = OrderedDict()
        for t in graph.input:
            all_inputs[t.name] = t
        # so we refresh the input by the initializer
        for t in graph.initializer:
            all_inputs[t.name] = t
        initializers = {t.name for t in graph.initializer}
        inp_idx = 0
        for name, x in all_inputs.items():
            if name in initializers:
                # if it has initializer, we use its value as the input
                np_tensor = numpy_helper.to_array(x)
                if np_tensor.dtype == "int64":
                    np_tensor = np_tensor.astype(np.int32)
                # todo, we cannot support scalar tensor
                if np.ndim(np_tensor) == 0:
                    np_tensor = np.array(np_tensor, ndmin=1)
            else:
                # if not, means it's a input rather than a inner weight
                # so if the user gives values, we use these values
                # if not, we just use the shape of input gived by onnx to init a random value
                # HOWEVER, the random value may not be correct for some inputs, such as gather which needs indices
                # so if have operators, the user must give inputs
                x_shape = tuple(
                    dim.dim_value for dim in x.type.tensor_type.shape.dim)
                if init_inputs is not None:
                    np_tensor = init_inputs[inp_idx]
                    inp_idx += 1
                else:
                    np_tensor = np.random.randn(*x_shape).astype(np.float32)
            tmp_tensor = tensor.from_numpy(np_tensor)
            tmp_tensor.to_device(device)
            # todo, for backward
            tmp_tensor.stores_grad = (name in initializers)
            tensor_map[x.name] = tmp_tensor
        return tensor_map

    @classmethod
    def _onnx_model_to_singa_net(cls, model, init_inputs, device,
                                 opset_version):
        """
        get all intermediate tensors and operators from onnx model
        Args:
            model: a given onnx model
        Args:
            init_inputs: a list of inputs, which used to init the operators
        Args:
            device: the used device
        Args:
            opset_version: the opset version
        Returns:
            a dict of tensors
        Returns:
            a list of SingaOps('name', 'op', 'handle', 'forward')
        """
        # init all tensor input and weight as a tensor map
        tensor_map = cls._init_graph_parameter(model.graph, init_inputs, device)
        # only weights tensor
        weights = {x.name: tensor_map[x.name] for x in model.graph.initializer}
        # the parsed operators queue
        singa_ops = []
        singa_op = namedtuple('SingaOps', ['name', 'op', 'handle', 'forward'])
        for node in model.graph.node:
            node = OnnxNode(node)
            # only give the inputs it needs
            # consumed_inputs are the inputs marked as attributes
            # so we remove it here
            inputs = [
                tensor_map[x]
                for x in node.inputs
                if x not in node.consumed_inputs
            ]
            handle, forward = cls._onnx_node_to_singa_op(
                node, inputs, opset_version)
            # if it is Constant, we hanlde it as a weight
            # otherwise, we run it and add its output into map for being used by later operators
            if node.op_type == 'Constant':
                tmp_tensor = tensor.from_numpy(forward)
                tmp_tensor.to_device(device)
                tmp_name = node.outputs.pop(0)
                weights[tmp_name] = tmp_tensor
                tensor_map[tmp_name] = tmp_tensor
            else:
                outputs = cls._run_node(node, inputs, handle, forward)
                for key, val in outputs.items():
                    tensor_map[key] = val
                singa_ops.extend([singa_op(node.name, node, handle, forward)])
        return weights, singa_ops

    @classmethod
    def prepare(cls, model, device, **kwargs):
        """
        get the batch norm operator from onnx node
        Args:
            model: a given onnx node
        Args:
            device: the used device
        Returns: 
            a list of output values
        """
        super(SingaBackend, cls).prepare(model, device, **kwargs)
        # when parsing graph, we use the shape of input gived by onnx to init a random value
        # HOWEVER, the random value may not be correct for some inputs, such as gather which needs indices
        # so if have operators, the user must give inputs
        init_inputs = kwargs.get("init_inputs", None)
        # whether initializers are moved into inputs, due to https://github.com/onnx/onnx/issues/2417
        # sometimes, input contains all initializer's info, sometimes, may not
        cls.keep_initializers_as_inputs = kwargs.get(
            'keep_initializers_as_inputs', True)
        # optimize and infer the shape of the model
        try:
            model = onnx.utils.polish_model(model)
        except IndexError as err:
            # due to https://github.com/onnx/onnx/issues/2417
            model = onnx.shape_inference.infer_shapes(model)

        # check the opset version and ir version
        opset_version = None
        for imp in model.opset_import:
            if not imp.HasField("domain") or imp.domain == "":
                opset_version = imp.version
                if imp.version > cls._known_opset_version:
                    warnings.warn(
                        "This version of singa targets ONNX operator set version {}, but the model we are trying to import uses version {}.  We will try to import it anyway, but if the model uses operators which had BC-breaking changes in the intervening versions, import will fail."
                        .format(cls._known_opset_version, imp.version))
            else:
                warnings.warn("Unrecognized operator set {}".format(imp.domain))
        if opset_version is None:
            if model.ir_version >= 0x00000003:
                raise RuntimeError(
                    "Model with IR version >= 3 did not specify ONNX operator set version (singa requires it)"
                )
            else:
                opset_version = 1
        weights, singa_ops = cls._onnx_model_to_singa_net(
            model, init_inputs, device, opset_version)
        return SingaRep(model, weights, singa_ops,
                        cls.keep_initializers_as_inputs)


class SingaRep(BackendRep):

    def __init__(self,
                 model,
                 weights,
                 singa_ops,
                 keep_initializers_as_inputs=True):
        """
        SingaRep provides the intermediate representation of Singa,
        the user can run the forward of the singa model by run func,
        or, the user can append more layers after the singa_ops to do
        the transfer learning
        Args:
            model: a given operator
        Args:
            weights: the tensor of weights
        Args:
            singa_ops: the tensor of the operator
        """
        super(SingaRep, self).__init__()
        self.model = model
        self.tensor_map = weights
        self.keep_initializers_as_inputs = keep_initializers_as_inputs
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
        graph = self.model.graph
        # last_layers means we run this model until the last #N layers
        last_layers = kwargs.get('last_layers', len(self.singa_ops))
        if last_layers != len(self.singa_ops):
            final_outputs = self.singa_ops[last_layers-1].op.outputs
        else:
            final_outputs =  [outp.name for outp in graph.output]
        # whether return all outputs
        all_outputs = kwargs.get('all_outputs', False)
        # get a specific op by its name
        op_name = kwargs.get('op_name', None)
        # record the tensor we added from input
        tmp_tensor_map = {name: val for name, val in self.tensor_map.items()}

        # the dict will be returned
        ret_outputs = OrderedDict()
        if self.keep_initializers_as_inputs:
            require_input_len = len(graph.input) - len(graph.initializer)
            actual_input_len = len(inputs)
        else:
            require_input_len = len(graph.input)
            actual_input_len = len(inputs)
        assert require_input_len == actual_input_len, "The length of graph input is different from the tensor input: %d, %d" % (
            require_input_len, actual_input_len)
        # run the handle by the order of the list(the list is Topological Sorting)
        for inp in graph.input:
            if inp.name not in tmp_tensor_map:
                tmp_tensor_map[inp.name] = inputs.pop(0)

        for _, op, handle, forward in self.singa_ops[:last_layers]:
            if len(op.consumed_inputs) != 0:
                # because if op has consumed_inputs, it means it moved some inputs into attributes
                # so when running, we should update these attributes
                handle, forward = get_op(op,
                                         [tmp_tensor_map[x] for x in op.inputs])
            inputs = [
                tmp_tensor_map[x]
                for x in op.inputs
                if x not in op.consumed_inputs
            ]
            outputs = _run_node(op, inputs, handle, forward)
            for key, val in outputs.items():
                tmp_tensor_map[key] = val
                ret_outputs[key] = val

        if op_name is not None:
            if op_name in outputs:
                return outputs[op_name]
            else:
                raise RuntimeError(
                    "The op_name {} does not exist, please check. The available op_names are: {}"
                    .format(op_name, [val for key, val in op_name.items()]))

        # return all outputs if all_outputs==True
        # else return last outputs
        if all_outputs:
            return ret_outputs
        else:
            return [ret_outputs[outp] for outp in final_outputs]


run_node = SingaBackend.run_node
_run_node = SingaBackend._run_node
prepare = SingaBackend.prepare
get_op = SingaBackend._onnx_node_to_singa_op
to_onnx = SingaFrontend.singa_to_onnx_model
save = onnx.save
load = onnx.load
