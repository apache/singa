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
from onnx import helper, checker
from onnx import TensorProto
from onnx import numpy_helper
from onnx.backend.base import BackendRep as backendRep
from onnx.backend.base import Backend as backend
import onnx

from . import singa_wrap as singa
from . import autograd
from . import tensor


class Handle(object):
    @staticmethod
    def conv(inputs, attrs):
        # inputs: a list of the input tensors
        kernel = tuple(attrs["kernel_shape"])
        padding = tuple(attrs["pads"][0:2])
        stride = tuple(attrs["strides"])
        group = attrs["group"]

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
        return handle

    @staticmethod
    def max_pool(inputs, attrs):
        x = inputs[0]
        kernel = tuple(attrs["kernel_shape"])
        padding = tuple(attrs["pads"][0:2])
        stride = tuple(attrs["strides"])
        if x.device.id() == -1:
            handle = singa.PoolingHandle(x.data, kernel, stride, padding, True)
        else:
            handle = singa.CudnnPoolingHandle(
                x.data, kernel, stride, padding, True
            )
        return handle

    @staticmethod
    def avg_pool(inputs, attrs):
        x = inputs[0]
        kernel = tuple(attrs["kernel_shape"])
        padding = tuple(attrs["pads"][0:2])
        stride = tuple(attrs["strides"])
        if x.device.id() == -1:
            handle = singa.PoolingHandle(
                x.data, kernel, stride, padding, False
            )
        else:
            handle = singa.CudnnPoolingHandle(
                x.data, kernel, stride, padding, False
            )
        return handle

    @staticmethod
    def batchnorm(inputs, attrs):
        x = inputs[0]
        factor = attrs["momentum"]
        if x.device.id() == -1:
            raise NotImplementedError
        else:
            handle = singa.CudnnBatchNormHandle(factor, x.data)
        return handle



UnaryOp = {
    "Relu": autograd.relu,
    "Softmax": autograd.softmax,
    "Flatten": autograd.flatten,
    "Tanh": autograd.tanh,
    "Sigmoid": autograd.sigmoid,
}
BinaryOp = {
    "Add": autograd.add_bias,
    "Mul": autograd.mul,
    "MatMul": autograd.matmul,
}

OtherOp = {
    "Conv": (Handle.conv, autograd.conv2d),
    "MaxPool": (Handle.max_pool, autograd.pooling_2d),
    "AveragePool": (Handle.avg_pool, autograd.pooling_2d),
    "BatchNormalization": (Handle.batchnorm, autograd.batchnorm_2d),
}

class SingaBackendRep(backendRep):
    def __init__(self, model, device, tensor_dict):
        """
        Args:
            model: onnx model proto
            device: singa device
            tensor_dict: dict for weight tensors
        """
        self.model = model
        self.device = device
        self.tensor_dict = tensor_dict
        self.handle_dict = {}

    @staticmethod
    def run_node(node, inputs, handles):
        """
        Args:
            node: onnx node proto
            inputs: a list of input tensors
            handles: dict from node name to handle

        Return:
            a list out output tensors
        """
        attrs = attribute2dict(node)
        op = node.op_type
        if op in UnaryOp:
            out = UnaryOp[op](inputs[0])
        elif op in BinaryOp:
            out = BinaryOp[op](inputs[0], inputs[1])
        elif op in OtherOp:
            handle, forward = OtherOp[op]
            if node.name not in handles:
                handles[node.name] = handle(inputs, attrs)
            out = forward(handles[node.name], *inputs)
        elif op == "Concat":
            out = autograd.cat(tuple(inputs), attrs["axis"])
        else:
            raise NotImplementedError("Not supported op: {}".format(op))
        return [out]

    def run(self, inputs):
        """
        Run the graph with given inputs.

        Args:
            inputs: a list of tensors whose name and order match the
                graph inputs.

        Return:
            a list of output tensors whose order match the graph outputs.
        """
        # input_dict: dict from input name to numpy array
        tensors = self.tensor_dict.copy()
        for i, x in enumerate(inputs):
            tensors[x.name] = x
            if x.name != self.model.graph.input[i].name:
                warnings.warn("the inputs do not match the graph inputs")

        for node in self.model.graph.node:
            if node.op_type != "Constant":
                inputs = [tensors[x] for x in node.input]
                outputs = SingaBackendRep.run_node(
                    node, inputs, self.handle_dict
                )
                for (key, val) in zip(node.output, outputs):
                    tensors[key] = val
        y = []
        for i in self.model.graph.output:
            y.append(tensors[i.name])
        return y


def attribute2dict(node):
    # create a dictionary from the node attribute name to value
    attr = {}
    for a in node.attribute:
        attr[a.name] = helper.get_attribute_value(a)
    return attr


class SingaBackend(backend):
    @classmethod
    def prepare(
        cls,
        model,  # type: ModelProto
        device,  # type: singa device
        **kwargs  # type: Any
    ):  # type: (...) -> Optional[BackendRep]
        """
        Args:
            model: onnx model proto
            device: singa device
        Return:
            SingaBackendRep instance
        """
        super(SingaBackend, cls).prepare(model, device, **kwargs)
        name2tensor = {}
        for node in model.graph.node:
            if node.op_type == "Constant":
                data = helper.get_attribute_value(node.attribute[0])
                requires_grad, stores_grad = True, True
                if len(node.attribute) == 3:
                    requires_grad = helper.get_attribute_value(
                        node.attribute[1]
                    )
                    stores_grad = helper.get_attribute_value(node.attribute[2])
                t = tensor.Tensor(
                    device=device,
                    data=numpy_helper.to_array(data),
                    requires_grad=requires_grad,
                    stores_grad=stores_grad,
                )

                name2tensor[node.output[0]] = t

        return SingaBackendRep(model, device, name2tensor)

    @classmethod
    def run_node(cls, node, inputs, device, outputs_info=None, **kwargs):
        """
        Args:
            node: onnx node proto
            inputs: list of singa tensors; the names should match
                node inputs
        Return:
            a list of singa tensors as the node outputs
        """
        super(SingaBackend, cls).run_node(node, inputs, device)
        handles = {}
        outputs = SingaBackendRep.run_node(node, inputs, handles)
        return outputs


def to_onnx_model(inputs, y, model_name="sonnx"):
    """
    get onnx model from singa computational graph
    Args:
        inputs: a list of input tensors (each is initialized with a name)
        y: a list of tensors, usually the outputs of the graph
    Return:
        the onnx model
    """
    assert len(y) == 1  # assume there is only one output
    y = y[0]  
    node = []
    dependency, _ = autograd.infer_dependency(y.creator)

    input_ids = set(id(x) for x in inputs)
    X = []
    for x in inputs:
        dtype = TensorProto.FLOAT
        if y.dtype == tensor.int32:
            dtype = TensorProto.INT
        X.append(helper.make_tensor_value_info(x.name, dtype, x.shape))
    Y = [helper.make_tensor_value_info(y.name, TensorProto.FLOAT, y.shape)]
    ready = deque([y.creator])

    while len(ready) > 0:
        op = ready.pop()
        assert not isinstance(op, autograd.Dummy)
        outputs = [op.output_name(idx) for yid, idx in op.y_id2idx.items()]
        inputs = [
            srcop.output_name(srcop.y_id2idx[yid])
            for (srcop, yid, _, _) in op.src
        ]
        opname = op.name 
        optype = str(op).split(".")[-1].split(" ")[0]
        if isinstance(op, autograd.Concat):
            node.append(
                helper.make_node(
                    "Concat",
                    inputs=inputs,
                    outputs=outputs,
                    name=opname,
                    axis=op.axis,
                )
            )
        elif isinstance(op, autograd._Conv2d):
            pads = [
                op.handle.pad_h,
                op.handle.pad_w,
                op.handle.pad_w,
                op.handle.pad_h,
            ]
            stride = [op.handle.stride_h, op.handle.stride_w]
            k = [op.handle.kernel_h, op.handle.kernel_w]
            node.append(
                helper.make_node(
                    "Conv",
                    inputs=inputs,
                    outputs=outputs,
                    name=opname,
                    kernel_shape=k,
                    pads=pads,
                    strides=stride,
                    group=op.handle.group,
                )
            )
        elif isinstance(op, autograd._Pooling2d):
            k = [op.handle.kernel_h, op.handle.kernel_w]
            s = [op.handle.stride_h, op.handle.stride_w]
            p = [
                op.handle.pad_h,
                op.handle.pad_w,
                op.handle.pad_w,
                op.handle.pad_h,
            ]
            if op.handle.is_max_pooling:
                node.append(
                    helper.make_node(
                        "MaxPool",
                        inputs=inputs,
                        outputs=outputs,
                        name=opname,
                        kernel_shape=k,
                        pads=p,
                        strides=s,
                    )
                )
            else:
                node.append(
                    helper.make_node(
                        "AveragePool",
                        inputs=inputs,
                        outputs=outputs,
                        name=opname,
                        kernel_shape=k,
                        pads=p,
                        strides=s,
                    )
                )
        elif isinstance(op, autograd._BatchNorm2d):
            node.append(
                helper.make_node(
                    "BatchNormalization",
                    inputs=inputs,
                    outputs=outputs,
                    name=opname,
                    momentum=op.handle.factor,
                )
            )
            # [(<singa.autograd.Sigmoid object at 0x7fd5ec09cb90>, 140556764852432, None, False),
            # (<singa.autograd.Dummy object at 0x7fd5ec09c390>, 140556764824208,
            # <singa.tensor.Tensor object at 0x7fd5ec09c290>, True),
            # (<singa.autograd.Dummy object at 0x7fd5ec09c490>, 140556764824528,
            # <singa.tensor.Tensor object at 0x7fd5ec09c3d0>, True),
            # (<singa.autograd.Dummy object at 0x7fd5ec09c590>, 140556764824784, None, False),
            # (<singa.autograd.Dummy object at 0x7fd5ec09c690>, 140556764825040, None, False)])
            # two dummy operators do not have values, so take the values from handle
            """
            dummy0 = tensor.to_numpy(
                tensor.Tensor(
                    device=op.running_mean.device(), data=op.running_mean
                )
            )
            dummy1 = tensor.to_numpy(
                tensor.Tensor(
                    device=op.running_var.device(), data=op.running_var
                )
            )
            dummy0 = helper.make_node(
                "Constant",
                inputs=[],
                outputs=[inputs[3]],
                value=numpy_helper.from_array(dummy0),
            )
            dummy1 = helper.make_node(
                "Constant",
                inputs=[],
                outputs=[inputs[4]],
                value=numpy_helper.from_array(dummy1),
            )
            node.append(dummy0)
            node.append(dummy1)
            """
        else:
            singa2onnx = {
                "SoftMax": "Softmax",
                "AddBias": "Add",
                "Add": "Add",
                "Matmul": "MatMul",
                "ReLU": "Relu",
                "ElemMatmul": "Mul",
                "Flatten": "Flatten",
                "Tanh": "Tanh",
                "Sigmoid": "Sigmoid"
            }
            assert optype in singa2onnx, "Unsupported op:{}".format(optype)
            onnx_op = singa2onnx[optype]
            node.append(
                helper.make_node(
                    onnx_op, inputs=inputs, outputs=outputs, name=opname
                )
            )

        for srcop, yid, y, _ in op.src:
            dependency[srcop] -= 1
            if dependency[srcop] == 0:
                if isinstance(srcop, autograd.Dummy):
                    if yid not in input_ids:
                        tmp = helper.make_node(
                            "Constant",
                            inputs=[],
                            outputs=[srcop.output_name(0)],
                            value=helper.make_tensor(
                                name=opname,
                                data_type=TensorProto.FLOAT,
                                dims=y.shape,
                                vals=tensor.to_numpy(y)
                                .flatten()
                                .astype(float),
                            ),
                        )
                        node.append(tmp)
                else:
                    ready.append(srcop)

    # print(node)
    onnx_model = helper.make_model(
        helper.make_graph(node[::-1], model_name, X, Y)
    )
    checker.check_model(onnx_model)
    return onnx_model


def export(inputs, y, file_path, model_name="sonnx"):
    onnx_model = to_onnx_model(inputs, y, model_name)
    onnx.save(onnx_model, file_path)


run_model = SingaBackend.run_model
run_node = SingaBackend.run_node
supports_device = SingaBackend.supports_device
prepare = SingaBackend.prepare
save = onnx.save
load = onnx.load
