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
                  TensorProto, OperatorSetIdProto, optimizer, mapping)
import warnings

from singa import device
from . import singa_wrap as singa
from . import autograd, layer
from . import tensor
from . import model
from singa import utils

import collections
OrderedDict = collections.OrderedDict
namedtuple = collections.namedtuple

# singa only supports float32 and int32
NP_TYPE_TO_SINGA_SUPPORT_TYPE = {
    np.dtype('float32'): np.dtype('float32'),
    np.dtype('uint8'): None,
    np.dtype('int8'): np.dtype('int32'),
    np.dtype('uint16'): None,
    np.dtype('int16'): np.dtype('int32'),
    np.dtype('int32'): np.dtype('int32'),
    np.dtype('int64'): np.dtype('int32'),
    np.dtype('bool'): np.dtype('float32'),
    np.dtype('float16'): np.dtype('float32'),
    np.dtype('float64'): np.dtype('float32'),
    np.dtype('complex64'): None,
    np.dtype('complex128'): None,
    np.dtype('uint32'): None,
    np.dtype('uint64'): None,
    np.dtype(np.object): None
}


def onnx_type_to_singa_type(onnx_type):
    np_type = mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_type]
    return NP_TYPE_TO_SINGA_SUPPORT_TYPE[np_type]


gpu_dev = None
if singa.USE_CUDA:
    gpu_dev = device.create_cuda_gpu(set_default=False)
cpu_dev = device.get_default_device()


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
        get a onnx node from singa operator, prepare its type, inputs and outputs
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
        self.name = str(node.name).replace(".", "_")
        self.op_type = str(node.op_type)
        self.attrs = OnnxAttributes.from_onnx(node.attribute)
        # inputs as attributes in singa
        self.attr_inputs = {}
        # inputs as weights in singa
        self.weight_inputs = {}
        self.inputs = list(node.input)
        self.outputs = list(node.output)

    def getattr(self, key, default=None):
        return self.attrs[key] if key in self.attrs else default

    def set_attr_inputs(self, key, name):
        self.attr_inputs[key] = name

    def del_attr_inputs(self, key):
        del self.attr_inputs[key]

    def set_weight_inputs(self, key, name):
        self.weight_inputs[key] = name

    def del_weight_inputs(self, key):
        del self.weight_inputs[key]


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
    _opset_version = 11

    _ir_version = 0x0000000000000006

    # beceuase singa's operators are different from onnx.
    # we define a dict for the name projection
    _rename_operators = {
        # common op
        'Relu': 'ReLU',
        'Sigmoid': 'Sigmoid',
        'Add': 'Add',
        'MatMul': 'Matmul',
        'Sum': 'Sum',
        'Cos': 'Cos',
        'Cosh': 'Cosh',
        'Sin': 'Sin',
        'Sinh': 'Sinh',
        'Tan': 'Tan',
        'Tanh': 'Tanh',
        'Acos': 'Acos',
        'Acosh': 'Acosh',
        'Asin': 'Asin',
        'Asinh': 'Asinh',
        'Atan': 'Atan',
        'Atanh': 'Atanh',
        'Equal': 'Equal',
        'Less': 'Less',
        'Sign': 'Sign',
        'Div': 'Div',
        'Sub': 'Sub',
        'Sqrt': 'Sqrt',
        'Log': 'Log',
        'Greater': 'Greater',
        'Identity': 'Identity',
        'Softplus': 'SoftPlus',
        'Softsign': 'SoftSign',
        'Mean': 'Mean',
        'Pow': 'Pow',
        'PRelu': 'PRelu',
        'Mul': 'Mul',
        'Max': 'Max',
        'Min': 'Min',
        'Shape': 'Shape',
        'And': 'And',
        'Or': 'Or',
        'Xor': 'Xor',
        'Not': 'Not',
        'Neg': 'Negative',
        'Reciprocal': 'Reciprocal',
        'Unsqueeze': 'Unsqueeze',
        'NonZero': 'NonZero',
        'Ceil': 'Ceil',
        # # special op
        'Cast': 'Cast',
        'Split': 'Split',
        'Squeeze': 'Squeeze',
        'GlobalAveragePool': 'GlobalAveragePool',
        'LeakyRelu': 'LeakyRelu',
        'ReduceSum': 'ReduceSum',
        'ReduceMean': 'ReduceMean',
        'Dropout': 'Dropout',
        'ConstantOfShape': 'ConstantOfShape',
        'Transpose': 'Transpose',
        'HardSigmoid': 'HardSigmoid',
        'Elu': 'Elu',
        'Selu': 'SeLU',
        'Concat': 'Concat',
        'Softmax': 'SoftMax',
        'Flatten': 'Flatten',
        'OneHot': 'OneHot',
        'Tile': 'Tile',
        'Gather': 'Gather',
        'Reshape': 'Reshape',
        'Slice': 'Slice',
        'Clip': 'Clip',
        'Gemm': 'layer.Gemm',  # layer
        'BatchNormalization': 'layer.BatchNorm2d',  # layer
        'Conv': 'layer.Conv2d',  # layer
        'MaxPool': 'layer.Pooling2d',  # layer
        'AveragePool': 'layer.Pooling2d',  # layer
    }

    # this dict indicates the operators that need extra handle
    # each indicates a function name
    _special_operators = {
        'Cast': '_create_cast',
        'Split': '_create_split',
        'Squeeze': '_create_squeeze_unsqueeze',
        'Unsqueeze': '_create_squeeze_unsqueeze',
        'GlobalAveragePool': '_create_global_average_pool',
        'LeakyRelu': '_create_leakyrelu',
        'ReduceSum': '_create_reduce_ops',
        'ReduceMean': '_create_reduce_ops',
        'Dropout': '_create_dropout',
        'ConstantOfShape': '_create_constant_of_shape',
        'Transpose': '_create_transpose',
        'HardSigmoid': '_create_hardsigmoid',
        'Elu': '_create_elu',
        'Selu': '_create_selu',
        'Concat': '_create_concat',
        'Softmax': '_create_softmax',
        'Gemm': '_create_gemm',
        'Flatten': '_create_flatten',
        'OneHot': '_create_onehot',
        'Tile': '_create_tile',
        'Gather': '_create_gather',
        'Reshape': '_create_reshape',
        'Slice': '_create_slice',
        'Clip': '_create_clip',
        'BatchNormalization': '_create_batch_norm',
        'Conv': '_create_conv',
        'MaxPool': '_create_max_avg_pool',
        'AveragePool': '_create_max_avg_pool',
    }

    @classmethod
    def _create_cast(cls, onnx_node, operator, opset_version=_opset_version):
        """
        get the Cast operator from onnx node
        Args:
            onnx_node (OnnxNode): a given onnx node
            operator (Operator Class): a singa operator class
            opset_version (int): the opset version
        Returns: 
            singa operator instance
        """
        to_type = onnx_type_to_singa_type(onnx_node.getattr("to"))
        assert to_type != None, "not support cast type: {}".format(to_type)
        if to_type == np.dtype('float32'):
            return operator(tensor.float32)
        else:
            return operator(tensor.int32)

    @classmethod
    def _create_split(cls, onnx_node, operator, opset_version=_opset_version):
        """
        get the Split operator from onnx node
        Args:
            onnx_node (OnnxNode): a given onnx node
            operator (Operator Class): a singa operator class
            opset_version (int): the opset version
        Returns: 
            singa operator instance
        """
        axis = onnx_node.getattr("axis", 0)
        split = onnx_node.getattr("split", None)
        num_output = len(onnx_node.outputs)
        return operator(axis, split, num_output)

    @classmethod
    def _create_squeeze_unsqueeze(cls,
                                  onnx_node,
                                  operator,
                                  opset_version=_opset_version):
        """
        get the Squeeze and Unsqueeze operator from onnx node
        Args:
            onnx_node (OnnxNode): a given onnx node
            operator (Operator Class): a singa operator class
            opset_version (int): the opset version
        Returns: 
            singa operator instance
        """
        axes = onnx_node.getattr("axes")
        return operator(axes)

    @classmethod
    def _create_global_average_pool(cls,
                                    onnx_node,
                                    operator,
                                    opset_version=_opset_version):
        """
        get the GlobalAveragePool operator from onnx node
        Args:
            onnx_node (OnnxNode): a given onnx node
            operator (Operator Class): a singa operator class
            opset_version (int): the opset version
        Returns: 
            singa operator instance
        """
        data_format = onnx_node.getattr("data_format", 'channels_first')
        return operator(data_format)

    @classmethod
    def _create_leakyrelu(cls,
                          onnx_node,
                          operator,
                          opset_version=_opset_version):
        """
        get the LeakyRelu operator from onnx node
        Args:
            onnx_node (OnnxNode): a given onnx node
            operator (Operator Class): a singa operator class
            opset_version (int): the opset version
        Returns: 
            singa operator instance
        """
        alpha = onnx_node.getattr("alpha", 0.01)
        return operator(alpha)

    @classmethod
    def _create_reduce_ops(cls,
                           onnx_node,
                           operator,
                           opset_version=_opset_version):
        """
        get the ReduceSum, ReduceMean, ReduceMax, ReduceMin, etc, operator from onnx node
        Args:
            onnx_node (OnnxNode): a given onnx node
            operator (Operator Class): a singa operator class
            opset_version (int): the opset version
        Returns: 
            singa operator instance
        """
        axes = onnx_node.getattr("axes", None)
        keepdims = onnx_node.getattr("keepdims", 1)
        return operator(axes, keepdims)

    @classmethod
    def _create_dropout(cls, onnx_node, operator, opset_version=_opset_version):
        """
        get the Dropout operator from onnx node
        Args:
            onnx_node (OnnxNode): a given onnx node
            operator (Operator Class): a singa operator class
            opset_version (int): the opset version
        Returns: 
            singa operator instance
        """
        ratio = onnx_node.getattr("ratio", 0)
        return operator(ratio)

    @classmethod
    def _create_constant_of_shape(cls,
                                  onnx_node,
                                  operator,
                                  opset_version=_opset_version):
        """
        get the ConstantOfShape operator from onnx node
        Args:
            onnx_node (OnnxNode): a given onnx node
            operator (Operator Class): a singa operator class
            opset_version (int): the opset version
        Returns: 
            singa operator instance
        """
        value = onnx_node.getattr("value", 0)
        if isinstance(value, onnx.TensorProto):
            value = numpy_helper.to_array(value)[0].item()
        return operator(value)

    @classmethod
    def _create_transpose(cls,
                          onnx_node,
                          operator,
                          opset_version=_opset_version):
        """
        get the Transpose operator from onnx node
        Args:
            onnx_node (OnnxNode): a given onnx node
            operator (Operator Class): a singa operator class
            opset_version (int): the opset version
        Returns: 
            singa operator instance
        """
        perm = onnx_node.getattr("perm")
        return operator(perm)

    @classmethod
    def _create_hardsigmoid(cls,
                            onnx_node,
                            operator,
                            opset_version=_opset_version):
        """
        get the hardsigmoid operator from onnx node
        Args:
            onnx_node (OnnxNode): a given onnx node
            operator (Operator Class): a singa operator class
            
            opset_version (int): the opset version
        Returns: 
            singa operator instance
        """
        alpha = onnx_node.getattr("alpha", 0.2)
        beta = onnx_node.getattr("beta", 0.5)
        return operator(alpha, beta)

    @classmethod
    def _create_elu(cls, onnx_node, operator, opset_version=_opset_version):
        """
        get the elu operator from onnx node
        Args:
            onnx_node (OnnxNode): a given onnx node
            operator (Operator Class): a singa operator class
            opset_version (int): the opset version
        Returns: 
            singa operator instance
        """
        alpha = onnx_node.getattr("alpha", 1.)
        return operator(alpha)

    @classmethod
    def _create_selu(cls, onnx_node, operator, opset_version=_opset_version):
        """
        get the selu operator from onnx node
        Args:
            onnx_node (OnnxNode): a given onnx node
            operator (Operator Class): a singa operator class
            opset_version (int): the opset version
        Returns: 
            singa operator instance
        """
        alpha = onnx_node.getattr("alpha", 1.67326)
        gamma = onnx_node.getattr("gamma", 1.0507)
        return operator(alpha, gamma)

    @classmethod
    def _create_concat(cls, onnx_node, operator, opset_version=_opset_version):
        """
        get the concat operator from onnx node
        Args:
            onnx_node (OnnxNode): a given onnx node
            operator (Operator Class): a singa operator class
            opset_version (int): the opset version
        Returns: 
            singa operator instance
        """
        factor = onnx_node.getattr('axis')
        return operator(axis=factor)

    @classmethod
    def _create_softmax(cls, onnx_node, operator, opset_version=_opset_version):
        """
        get the softmax operator from onnx node
        Args:
            onnx_node (OnnxNode): a given onnx node
            operator (Operator Class): a singa operator class
            opset_version (int): the opset version
        Returns: 
            singa operator instance
        """
        factor = onnx_node.getattr('axis', 1)
        return operator(axis=factor)

    @classmethod
    def _create_gemm(cls, onnx_node, operator, opset_version=_opset_version):
        """
        get the gemm operator from onnx node
        Args:
            onnx_node (OnnxNode): a given onnx node
            operator (Operator Class): a singa operator class
            opset_version (int): the opset version
        Returns: 
            singa operator instance
        """
        alpha = onnx_node.getattr('alpha', 1.)
        beta = onnx_node.getattr('beta', 1.)
        transA = onnx_node.getattr('transA', 0)
        transB = onnx_node.getattr('transB', 0)
        onnx_node.set_weight_inputs(onnx_node.inputs[1], 'W')
        bias = False
        if len(onnx_node.inputs) == 3:
            onnx_node.set_weight_inputs(onnx_node.inputs[2], 'b')
            bias = True
        return operator(None,
                        alpha=alpha,
                        beta=beta,
                        transA=transA,
                        transB=transB,
                        bias=bias)

    @classmethod
    def _create_flatten(cls, onnx_node, operator, opset_version=_opset_version):
        """
        get the flatten operator from onnx node
        Args:
            onnx_node (OnnxNode): a given onnx node
            operator (Operator Class): a singa operator class
            opset_version (int): the opset version
        Returns: 
            singa operator instance
        """
        factor = onnx_node.getattr('axis', 1)
        return operator(axis=factor)

    @classmethod
    def _create_onehot(cls, onnx_node, operator, opset_version=_opset_version):
        """
        get the OneHot operator from onnx node
        Args:
            onnx_node (OnnxNode): a given onnx node
            operator (Operator Class): a singa operator class
            opset_version (int): the opset version
        Returns: 
            singa operator instance
        """
        axis = onnx_node.getattr("axis", -1)
        onnx_node.set_attr_inputs(onnx_node.inputs[1], 'depth')
        onnx_node.set_attr_inputs(onnx_node.inputs[2], 'values')
        return operator(axis, None, None)

    @classmethod
    def _create_tile(cls, onnx_node, operator, opset_version=_opset_version):
        """
        get the Tile operator from onnx node
        Args:
            onnx_node (OnnxNode): a given onnx node
            operator (Operator Class): a singa operator class
            opset_version (int): the opset version
        Returns: 
            singa operator instance
        """
        onnx_node.set_attr_inputs(onnx_node.inputs[1], 'repeats')
        return operator(None)

    @classmethod
    def _create_gather(cls, onnx_node, operator, opset_version=_opset_version):
        """
        get the Gather operator from onnx node
        Args:
            onnx_node (OnnxNode): a given onnx node
            operator (Operator Class): a singa operator class
            opset_version (int): the opset version
        Returns: 
            singa operator instance
        """
        axis = onnx_node.getattr("axis", 0)
        onnx_node.set_attr_inputs(onnx_node.inputs[1], 'indices')
        return operator(axis, None)

    @classmethod
    def _create_reshape(cls, onnx_node, operator, opset_version=_opset_version):
        """
        get the reshape operator from onnx node
        Args:
            onnx_node (OnnxNode): a given onnx node
            operator (Operator Class): a singa operator class
            opset_version (int): the opset version
        Returns: 
            singa operator instance
        """
        onnx_node.set_attr_inputs(onnx_node.inputs[1], 'shape')
        return operator(None)

    @classmethod
    def _create_slice(cls, onnx_node, operator, opset_version=_opset_version):
        """
        get the Slice operator from onnx node
        Args:
            onnx_node (OnnxNode): a given onnx node
            operator (Operator Class): a singa operator class
            opset_version (int): the opset version
        Returns: 
            singa operator instance
        """
        onnx_node.set_attr_inputs(onnx_node.inputs[1], 'starts')
        onnx_node.set_attr_inputs(onnx_node.inputs[2], 'ends')
        if len(onnx_node.inputs) >= 4 and onnx_node.inputs[3] != '':
            onnx_node.set_attr_inputs(onnx_node.inputs[3], 'axes')
        if len(onnx_node.inputs) == 5 and onnx_node.inputs[4] != '':
            onnx_node.set_attr_inputs(onnx_node.inputs[4], 'steps')
        return operator(None, None, None, None)

    @classmethod
    def _create_clip(cls, onnx_node, operator, opset_version=_opset_version):
        """
        get the clip operator from onnx node
        Args:
            onnx_node (OnnxNode): a given onnx node
            operator (Operator Class): a singa operator class
            opset_version (int): the opset version
        Returns: 
            singa operator instance
        """
        if len(onnx_node.inputs) >= 2 and onnx_node.inputs[1] != '':
            onnx_node.set_attr_inputs(onnx_node.inputs[1], 'min')
        if len(onnx_node.inputs) == 3 and onnx_node.inputs[2] != '':
            onnx_node.set_attr_inputs(onnx_node.inputs[2], 'max')
        return operator(None, None)

    @classmethod
    def _create_batch_norm(cls,
                           onnx_node,
                           operator,
                           opset_version=_opset_version):
        """
        get the clip operator from onnx node
        Args:
            onnx_node (OnnxNode): a given onnx node
            operator (Operator Class): a singa operator class
            opset_version (int): the opset version
        Returns: 
            singa operator instance
        """
        factor = onnx_node.getattr('momentum', 0.9)
        onnx_node.set_weight_inputs(onnx_node.inputs[1], 'scale')
        onnx_node.set_weight_inputs(onnx_node.inputs[2], 'bias')
        onnx_node.set_weight_inputs(onnx_node.inputs[3], 'running_mean')
        onnx_node.set_weight_inputs(onnx_node.inputs[4], 'running_var')
        return operator(factor)

    @classmethod
    def _create_conv(cls, onnx_node, operator, opset_version=_opset_version):
        """
        get the clip operator from onnx node
        Args:
            onnx_node (OnnxNode): a given onnx node
            operator (Operator Class): a singa operator class
            opset_version (int): the opset version
        Returns: 
            singa operator instance
        """
        kernel_size = tuple(onnx_node.getattr('kernel_shape'))
        padding = tuple(onnx_node.getattr('pads', (0, 0)))
        stride = tuple(onnx_node.getattr('strides', (1, 1)))
        auto_pad = utils.force_unicode(onnx_node.getattr('auto_pad', 'NOTSET'))

        # not support dilation
        dilation = onnx_node.getattr('dilations', 1)
        if dilation != 1 and list(dilation) != [1, 1]:
            raise ValueError("Not implemented yet for dilation")
        group = onnx_node.getattr('group', 1)

        # only support 1d or 2d
        if len(kernel_size) > 2:
            raise ValueError("Only implemented for 1d or 2d")

        onnx_node.set_weight_inputs(onnx_node.inputs[1], 'W')
        bias = False
        if len(onnx_node.inputs) == 3:
            onnx_node.set_weight_inputs(onnx_node.inputs[2], 'b')
            bias = True
        return operator(None,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        group=group,
                        bias=bias,
                        pad_mode=auto_pad)

    @classmethod
    def _create_max_avg_pool(cls,
                             onnx_node,
                             operator,
                             opset_version=_opset_version):
        """
        get the clip operator from onnx node
        Args:
            onnx_node (OnnxNode): a given onnx node
            operator (Operator Class): a singa operator class
            opset_version (int): the opset version
        Returns: 
            singa operator instance
        """
        kernel_size = tuple(onnx_node.getattr('kernel_shape'))
        padding = tuple(onnx_node.getattr('pads', (0, 0)))
        stride = tuple(onnx_node.getattr('strides', (1, 1)))
        auto_pad = utils.force_unicode(onnx_node.getattr('auto_pad', 'NOTSET'))

        # not support count_include_pad and auto_pad
        ceil_mode = onnx_node.getattr('ceil_mode', 0)
        count_include_pad = onnx_node.getattr('count_include_pad', 0)
        if ceil_mode != 0 or count_include_pad != 0:
            raise ValueError(
                "Not implemented yet for count_include_pad or ceil_mode")

        # only support 1d or 2d
        if len(kernel_size) > 2:
            raise ValueError("Only implemented for 1d or 2d")

        is_max = onnx_node.op_type == 'MaxPool'
        return operator(kernel_size, stride, padding, is_max, auto_pad)

    @classmethod
    def _onnx_constant_to_np(cls, onnx_node, opset_version=_opset_version):
        """
        parse onnx constatn node to numpy array
        Args:
            onnx_node (OnnxNode): a given onnx node
            opset_version (int): the opset version
        Returns: 
            a numpy ndarray
        """
        onnx_tensor = onnx_node.getattr('value')
        np_dtype = mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_tensor.data_type]
        np_tensor = np.frombuffer(onnx_tensor.raw_data, dtype=np_dtype)
        return tensor.from_numpy(np_tensor)

    @classmethod
    def _onnx_node_to_singa_op(cls, onnx_node, opset_version=_opset_version):
        """
        get singa operator from a onnx node
        Args:
            onnx_node (OnnxNode): a given onnx node
            opset_version (int): the opset version
        Returns: 
            singa operator instance
        """
        onnx_op_type = onnx_node.op_type
        assert onnx_op_type in cls._rename_operators, "not support operator: {}".format(
            onnx_op_type)
        renamed_op = cls._rename_operators[onnx_op_type]
        if renamed_op.startswith('layer.'):
            op_class = getattr(layer, renamed_op[6:])
        else:
            op_class = getattr(autograd, renamed_op)
        if onnx_node.op_type in cls._special_operators:
            translator = getattr(cls, cls._special_operators[onnx_node.op_type])
            op = translator(onnx_node, op_class, opset_version)
            op.name = onnx_node.name
        else:
            op = op_class()
        # refine the ONNXNode
        onnx_node.inputs = [inp for inp in onnx_node.inputs if inp != '']
        return op

    @classmethod
    def run_node(cls, node, inputs, device='CPU', opset_version=_opset_version):
        """
        run a single singa operator from a onnx node
        Args:
            node (NodeProto): a given onnx node
            inputs (ndarray[]): a list of numpy ndarray
            device (string): CPU or CUDA
            opset_version (int): the opset version
        Returns:
            list, the output
        """
        node = OnnxNode(node)
        valid_inputs = [x for x in node.inputs if x != ""]
        assert len(valid_inputs) == len(
            inputs), "{}: expected {} inputs, but got {}. ".format(
                node.op_type, len(valid_inputs), len(inputs))

        operator = cls._onnx_node_to_singa_op(node, opset_version)
        # seperate weights with inputs, and init inputs as Tensor
        weights = {}
        _inputs = []
        for (key, val) in zip(valid_inputs, inputs):
            val = val.astype(onnx_type_to_singa_type(val.dtype))
            if key in node.weight_inputs:
                weights[key] = val
            else:
                x = tensor.from_numpy(val)
                if device == 'CPU':
                    assert singa.USE_CUDA, "Your SINGA doesn't compile GPU module."
                    dev = device.create_cuda_gpu(set_default=False)
                else:
                    dev = device.get_default_device()
                x.to_device(dev)
                _inputs.append(x)
        inputs = _inputs
        # set params
        params = {}
        for key, name in node.weight_inputs.items():
            params[name] = weights[key]
        operator.set_params(params)
        outputs = cls._run_node(operator, inputs)
        outputs_dict = OrderedDict()
        for (key, val) in zip(node.outputs, outputs):
            outputs_dict[key] = val
        return outputs_dict

    @classmethod
    def _run_node(cls, operator, inputs):
        """
        run a single singa operator from singa operator
        Args:
            operator (Operator): the Operator instance
            inputs (Tensor[]): a list of SINGA Tensor
        Returns:
            list, the output
        """
        outputs = operator(*inputs)
        if not isinstance(outputs, collections.Iterable):
            outputs = [outputs]
        return outputs

    @classmethod
    def _parse_graph_params(cls, graph, device):
        """
        parse the parameters from onnx graph
        Args:
            graph (Graph): a given onnx graph
            device (string): CPU or CUDA
        Returns:
            a dict of numpy ndarray
        """
        params = {}
        for tp in graph.initializer:
            val = numpy_helper.to_array(tp)
            val = val.astype(onnx_type_to_singa_type(tp.data_type))
            params[tp.name] = val
        return params

    @classmethod
    def _parse_graph_inputs_outputs(cls, graph, params, device):
        """
        parse the inits, outputs from onnx graph
        Args:
            graph (Graph): a given onnx graph
            device (string): # CPU or CUDA
        Returns:
            a dict of ValueInfo
            a dict of ValueInfo
        """
        inputs = []
        outputs = []
        info_tuple = namedtuple('info_tuple', ['name', 'dtype', 'shape'])
        for t in graph.input:
            if t.name not in params:
                dtype = t.type.tensor_type.elem_type
                shape = [dim.dim_value for dim in t.type.tensor_type.shape.dim]
                inputs.extend([info_tuple(t.name, dtype, shape)])
        for t in graph.output:
            dtype = t.type.tensor_type.elem_type
            shape = [dim.dim_value for dim in t.type.tensor_type.shape.dim]
            outputs.extend([info_tuple(t.name, dtype, shape)])
        return inputs, outputs

    @classmethod
    def _onnx_model_to_singa_ops(cls, graph, device, opset_version=_opset_version):
        """
        get all intermediate params, operators, and input info from onnx model
        Args:
            graph (Graph): the loaded ONNX graph
            device (string): CPU or CUDA
            opset_version (int): the opset version
        Returns:
            a dict of weights
            a dict of ValueInfo
            a dict of ValueInfo
            a list of SingaOps('node', 'forward')
        """
        # init all tensor input and params as a tensor map
        params = cls._parse_graph_params(graph, device)
        inputs, outputs = cls._parse_graph_inputs_outputs(graph, params, device)
        # the parsed operators queue
        operators = []
        operator_tuple = namedtuple('operator_tuple', ['node', 'operator'])
        for node in graph.node:
            node = OnnxNode(node)
            # convert Constant to param
            if node.op_type == 'Constant':
                params[node.outputs[0]] = cls._onnx_constant_to_np(node)
            else:
                op = cls._onnx_node_to_singa_op(node, opset_version)
                operators.extend([operator_tuple(node, op)])
        return params, inputs, outputs, operators

    @classmethod
    def prepare(cls, model, device='CPU', **kwargs):
        """
        parse the ONNX and to create layers
        Args:
            model (ModelProto): the loaded ONNX model
            device (string): CPU or CUDA
        Returns:
            a SingaRep instance to stores the layers and weights
        """
        super(SingaBackend, cls).prepare(model, device, **kwargs)
        # optimize and infer the shape of the model
        try:
            model = onnx.utils.polish_model(model)
        except IndexError as err:
            model = onnx.shape_inference.infer_shapes(model)

        # check the opset version and ir version
        # SINGA supports opset version(11), ir version(1.6.0 -> 6)
        opset_version = None
        for imp in model.opset_import:
            if not imp.HasField("domain") or imp.domain == "":
                opset_version = imp.version
                if imp.version > cls._opset_version:
                    warnings.warn(
                        "The imported opertor set verion {} is larger than the supported version {}."
                        .format(imp.version, cls._opset_version))
            else:
                warnings.warn("Unrecognized operator set {}".format(imp.domain))

        if model.ir_version > cls._ir_version:
            warnings.warn(
                "The imported ir verion {} is larger than the supported version {}."
                .format(cls._ir_version, imp.version))

        graph = model.graph
        params, inputs, outputs, layers = cls._onnx_model_to_singa_ops(
            graph, device, opset_version)
        return SingaRep(params, inputs, outputs, layers, device)


class SingaRep(BackendRep):

    def __init__(self, params, inputs, outputs, layers, device):
        """
        https://github.com/onnx/onnx/blob/master/docs/ImplementingAnOnnxBackend.md
        SingaRep provides the intermediate representation of Singa,
        the user can run the forward of the singa model by run func,
        or, the user can append more layers after the singa_ops to do
        the transfer learning
        Args:
            params (dict{}): a dict of params, data type is numpy ndarray
            inputs (ValueInfo): a dict of inputs
            outputs (ValueInfo): a dict of outputs
            layers (namedtuple('operator_tuple', ['node', 'operator'])[]): a list of singa operator
            device (string): CPU or CUDA
        """
        super(SingaRep, self).__init__()
        self.inputs = inputs
        self.states = params
        self.outputs = outputs
        self.dev = cpu_dev if device == "CPU" else gpu_dev
        self.layers = layers
        self.tensor_count = {}
        self.has_initialized = False
        self.is_graph = False

    def initialize(self):
        """
        Init the instance
        """
        self.outputs_info = {outp.name: outp for outp in self.outputs}
        _layers = []  # layers by topo order
        for node, operator in self.layers:
            _del_keys = []
            for key, name in node.weight_inputs.items():
                if key not in self.states:
                    # cannot find the weights, try to find it from input
                    node.set_attr_inputs(key, name)
                    _del_keys.append(key)
            for key in _del_keys:
                node.del_weight_inputs(key)
            self.__dict__[node.name] = operator
            _layers.append(node)
        self._layers = _layers

    def init_tensor_count(self):
        """
        Init the tensor count dict
        """
        self.tensor_count = {}
        for node, operator in self.layers:
            # init the tensor count
            all_possible_inputs = node.inputs + list(
                node.attr_inputs.keys()) + list(node.weight_inputs.keys())
            for inp in all_possible_inputs:
                if inp not in self.tensor_count:
                    self.tensor_count[inp] = 1
                else:
                    self.tensor_count[inp] += 1

    def to_input_tensor(self, x):
        """
        convert the input to tensors
        Args:
            x (np.ndarray[]): a list of numpy ndarray as inputs
        Returns: 
            a dict of SINGA Tensors
        """
        tensor_dict = {}
        # init inputs as Tensor
        for (key, val) in zip(self.inputs, x):
            if not self.is_graph:
                val = val.astype(onnx_type_to_singa_type(key.dtype))
                # todo, scalar
                val = np.atleast_1d(val)
                val = tensor.from_numpy(val)
                val.to_device(self.dev)
            tensor_dict[key.name] = val
        return tensor_dict

    def to_output_tensor(self, y, out_name):
        """
        convert the tensors to input
        Args:
            x (np.ndarray[]): a list of numpy ndarray as inputs
        Returns: 
            a dict of SINGA Tensors
        """
        if not self.is_graph:
            y = tensor.to_numpy(y)
            if out_name in self.outputs_info:
                np_dtyp = mapping.TENSOR_TYPE_TO_NP_TYPE[
                    self.outputs_info[out_name].dtype]
                y = y.astype(np_dtyp)
        return y

    def get_s(self, name, node, tensor_dict):
        """
        get state from the node's weights or tensor_dict
        Args:
            name (str): name of the state
            node (ONNXNode): ONNX node
            tensor_dict ({}): tensor dict
        Returns: 
            the states
        """
        if name in node.attr_inputs:
            return tensor_dict[name]
        else:
            return self.states[name]

    def handle_special_ops(self, node, op, tensor_dict):
        """
        hanlde some special operations
        Args:
            name (str): name of the state
            node (ONNXNode): ONNX node
            tensor_dict ({}): tensor dict
        Returns: 
            the states
        """
        # todo, hard code
        # Conv2d nb_kernels
        if node.op_type == "Conv":
            shape = self.get_s(node.inputs[1], node, tensor_dict).shape
            op.nb_kernels = shape[0]
        # Gemm nb_kernels and bias_shape
        elif node.op_type == "Gemm":
            nb_kernels_flag = 0 if op.transB == 1 else 1
            shape = self.get_s(node.inputs[1], node, tensor_dict).shape
            op.nb_kernels = shape[nb_kernels_flag]
            if op.bias:
                shape = self.get_s(node.inputs[2], node, tensor_dict).shape
                op.bias_shape = shape

    def run(self, x, **kwargs):
        """
        run the forward of singa model
        Args:
            x (np.ndarray[]): a list of numpy ndarray as inputs
        Returns: 
            a list of outputs
        """
        if not self.has_initialized:
            self.initialize()
            if isinstance(x[0], tensor.Tensor):
                self.dev = x[0].device

        outputs_dict = OrderedDict([(outp.name, None) for outp in self.outputs])

        # last_layers means we run this model until the last #N layers
        last_layers = kwargs.get('last_layers', len(self._layers))
        if last_layers != len(self._layers):
            for outp in self._layers[last_layers - 1].outputs:
                outputs_dict[outp] = None

        aux_output = kwargs.get('aux_output', ())
        for outp in aux_output:
            outputs_dict[outp] = None

        tensor_dict = self.to_input_tensor(x)
        self.init_tensor_count()

        # run the layer by the topo order
        for node in self._layers[:last_layers]:
            op = self.__dict__[node.name]
            self.handle_special_ops(node, op, tensor_dict)
            # make input
            inputs = []
            for inp in node.inputs:
                if inp not in node.weight_inputs and inp not in node.attr_inputs:
                    if inp in tensor_dict:
                        inputs.append(tensor_dict[inp])
                    elif inp in self.states:
                        # todo, scalar
                        val = np.atleast_1d(self.states[inp])
                        val = tensor.from_numpy(val)
                        val.to_device(self.dev)
                        inputs.append(val)
                    else:
                        raise KeyError(
                            "Not found the input {} for operation {}".format(
                                inp, node.name))
            states = {}
            if callable(getattr(op, "initialize",
                                None)) and not op._initialized:
                # init the operator
                op.initialize(*inputs)
                op._initialized = True
                for key, name in node.weight_inputs.items():
                    if key not in node.attr_inputs:
                        # find the weights and not in the inputs
                        states[name] = self.states[key]

            # replace attrs by inputs
            for key, name in node.attr_inputs.items():
                if key in tensor_dict:
                    ts = tensor_dict[key]
                    if isinstance(ts, tensor.Tensor):
                        ts = tensor.to_numpy(ts)
                    states[name] = ts
                elif key in self.states:
                    states[name] = self.states[key]
            # set states
            if states:
                if callable(getattr(op, "set_states", None)):
                    # rename the layer's states
                    states = {
                        getattr(op, key).name: val
                        for (key, val) in states.items()
                    }
                    if self.is_graph and not self.has_initialized:
                        prev_state = self.dev.graph_enabled()
                        self.dev.EnableGraph(False)
                        op.set_states(states)
                        self.dev.EnableGraph(prev_state)
                    else:
                        op.set_states(states)
                else:
                    for key, value in states.items():
                        setattr(op, key, value)
            # run the node
            outputs = _run_node(op, inputs)
            # release the input tensor
            for inp in node.inputs:
                if inp in self.tensor_count:
                    self.tensor_count[inp] -= 1
                if self.tensor_count[inp] == 0:
                    if inp in tensor_dict:
                        del tensor_dict[inp]
                    del self.tensor_count[inp]
            # store the output
            for (outp, val) in zip(node.outputs, outputs):
                tensor_dict[outp] = val
                if outp in outputs_dict:
                    outputs_dict[outp] = self.to_output_tensor(val, outp)
        self.has_initialized = True
        return list(outputs_dict.values())


class SONNXModel(model.Model):

    def __init__(self, onnx_model):
        """
        Init a SIGNA Model
        Args:
            onnx_model (ModelProto): a loaded onnx model
        """
        super(SONNXModel, self).__init__()
        self.sg_ir = prepare(onnx_model)
        for node, operator in self.sg_ir.layers:
            self.__dict__[node.name] = operator
        self.sg_ir.is_graph = True

    def forward(self, *input, aux_output=()):
        """
        The forward of the SINGA model
        Args:
            input (Tensors[]): a list of Tensor
            aux_output (string()): a set of required output name
        Returns:
            a OrderedDict of Tensor
        """
        return self.sg_ir.run(input, aux_output=aux_output)


run_node = SingaBackend.run_node
_run_node = SingaBackend._run_node
prepare = SingaBackend.prepare
get_op = SingaBackend._onnx_node_to_singa_op
to_onnx = SingaFrontend.singa_to_onnx_model
save = onnx.save
load = onnx.load
