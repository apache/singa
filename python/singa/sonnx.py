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

from collections import deque
from onnx import helper, checker
from onnx import TensorProto
from onnx import numpy_helper
from onnx.backend.base import BackendRep as backendRep
from onnx.backend.base import Backend as backend
from onnx.backend.base import namedtupledict

from . import singa_wrap as singa
import autograd

import tensor
from device import create_cuda_gpu_on, get_default_device


class Handle(object):

    @staticmethod
    def conv(inputs, attrs):
        # inputs: a list of the input tensors
        kernel = tuple(attrs['kernel_shape'])
        padding = tuple(attrs['pads'])
        stride = tuple(attrs['strides'])
        group = 1
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
                                          in_channels, out_channels, bias)
            handle.device_id = inputs[0].device.id()
        else:
            handle = singa.CudnnConvHandle(x.data, kernel, stride, padding,
                                           in_channels, out_channels, bias,
                                           group)
            handle.device_id = inputs[0].device.id()
        return handle

    @staticmethod
    def max_pool(inputs, attrs):
        x = inputs[0]
        kernel = tuple(attrs['kernel_shape'])
        padding = tuple(attrs['pads'])
        stride = tuple(attrs['strides'])
        if x.device.id() == -1:
            handle = singa.PoolingHandle(x.data, kernel, stride, padding, True)
            handle.device_id = inputs[0].device.id()
        else:
            handle = singa.CudnnPoolingHandle(x.data, kernel, stride, padding,
                                              True)
            handle.device_id = inputs[0].device.id()
        return handle

    @staticmethod
    def avg_pool(inputs, attrs):
        x = inputs[0]
        kernel = tuple(attrs['kernel_shape'])
        padding = tuple(attrs['pads'])
        stride = tuple(attrs['strides'])
        if x.device.id() == -1:
            handle = singa.PoolingHandle(x.data, kernel, stride, padding, False)
            handle.device_id = inputs[0].device.id()
        else:
            handle = singa.CudnnPoolingHandle(x.data, kernel, stride, padding,
                                              False)
            handle.device_id = inputs[0].device.id()
        return handle

    @staticmethod
    def batchnorm(inputs, attrs):
        x = inputs[0]
        factor = attrs['momentum']
        if x.device.id() == -1:
            raise NotImplementedError
        else:
            handle = singa.CudnnBatchNormHandle(factor, x.data)
            handle.device_id = inputs[0].device.id()
        return handle

UnaryOp = {'Relu': autograd.relu,
           'Softmax': autograd.softmax,
           'Flatten': autograd.flatten,
           'Tanh': autograd.tanh,
           'Sigmoid': autograd.sigmoid}
BinaryOp = {'Add': autograd.add,
            'Mul': autograd.mul,
            'MatMul': autograd.matmul}

OtherOp = {'Conv': (Handle.conv, autograd.conv2d),
           'MaxPool': (Handle.max_pool, autograd.pooling_2d),
           'AveragePool': (Handle.avg_pool, autograd.pooling_2d),
           'BatchNormalization': (Handle.batchnorm, autograd.batchnorm_2d)
           }


class SingaBackendRep(backendRep):

    def __init__(self, model, device, tensor_dict):
        '''
        Args:
            model: onnx model proto
            device: singa device
            tensor_dict: dict for weight tensors
        '''
        self.model = model
        self.device = device
        self.tensor_dict = tensor_dict
        self.handle_dict = {}

    @staticmethod
    def run_node(node, tensors, handles):
        '''
        Args:
            node: onnx node proto
            tensors: dict from tensor name to tensor
            handles: dict from node name to handle
        '''
        inputs = [tensors[x] for x in node.input]
        outputs = node.output
        attrs = attribute2dict(node)
        op = node.op_type
        if op in UnaryOp:
            tensors[outputs[0]] = UnaryOp[op](inputs[0])
        elif op in BinaryOp:
            tensors[outputs[0]] = BinaryOp[op](inputs[0], inputs[1])
        elif op in OtherOp:
            handle, forward = OtherOp[op]
            if node.name not in handles:
                handles[node.name] = handle(inputs, attrs)
            tensors[outputs[0]] = forward(handles[node.name], *inputs)
        elif op == 'Concat':
            tensors[outputs[0]] = autograd.cat(tuple(inputs), attrs['axis'])
        else:
            raise NotImplementedError('Not supported op: {}'.format(op))

    def run(self, input):
        # input_dict: dict from input name to numpy array
        tensors = self.tensor_dict.copy()
        for i in range(len(input)):
            key=self.model.graph.input[i].name
            tensors[key] = input[i]

        for node in self.model.graph.node:
            if(node.op_type!="Constant"):
                SingaBackendRep.run_node(node, tensors, self.handle_dict)
        y=[]
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
    def prepare(cls,
                model,  # type: ModelProto
                device,  # type: singa device
                **kwargs  # type: Any
                ):  # type: (...) -> Optional[BackendRep]
        '''
        Args:
            model: onnx model proto
            device: singa device
        Return:
            SingaBackendRep instance
        '''
        super(SingaBackend, cls).prepare(model, device, **kwargs)
        name2tensor = {}
        for node in model.graph.node:
            if (node.op_type == 'Constant'):
                data = helper.get_attribute_value(node.attribute[0])
                requires_grad, stores_grad = True, True
                if len(node.attribute) == 3:
                    requires_grad = helper.get_attribute_value(
                        node.attribute[1])
                    stores_grad = helper.get_attribute_value(node.attribute[2])
                t = tensor.Tensor(device=device,
                                  data=numpy_helper.to_array(data),
                                  requires_grad=requires_grad,
                                  stores_grad=stores_grad)

                name2tensor[node.output[0]] = t

        return SingaBackendRep(model, device, name2tensor)

    @classmethod
    def run_node(cls, node, inputs, device, outputs_info=None, **kwargs):
        '''
        Args:
            node: onnx node proto
            inputs: list of singa tensor array; the names should match
                node inputs
        Return:
            a named tuple for the output tensors
        '''
        super(SingaBackend, cls).run_node(node, inputs, device)
        handles = {}
        SingaBackendRep.run_node(node, inputs, handles)
        output_vals = [tensors[x] for x in node.outputs]
        return namedtupledict('Outputs', node.outputs)(*output_vals)



run_model = SingaBackend.run_model
run_node = SingaBackend.run_node
supports_device = SingaBackend.supports_device
prepare = SingaBackend.prepare


def to_onnx_model(inputs, y, model_name='sonnx'):
    '''
    get onnx model from singa computational graph
    Args:
        inputs: a list of input tensors (each is initialized with a name)
        y: a Tensor instance, usually the output of the graph
    Return:
        the onnx model
    '''
    node = []
    dependency = autograd.infer_dependency(y.creator)
    ready = deque([y.creator])

    def output_name(op,extra=''):
        return '{}'.format(op.name+str(extra))

    input_ids = set(id(x) for x in inputs)
    X = []
    for x in inputs:
        dtype = TensorProto.FLOAT
        if y.dtype == tensor.int32:
            dtype = TensorProto.INT
        X.append(helper.make_tensor_value_info(x.name, dtype, x.shape))
    Y = [helper.make_tensor_value_info(
        output_name(y.creator), TensorProto.FLOAT, y.shape)]

    while len(ready) > 0:
        op = ready.pop()
        assert not isinstance(op, autograd.Dummy)
        outputs = [output_name(op)for _, idx in op.y_id2idx.items()]
        if(len(outputs)!=1):
            outputs = [output_name(op,idx)for _, idx in op.y_id2idx.items()]
        inputs = [output_name(srcop) for (srcop, yid, _, _) in op.src]
        curop = str(op).split('.')[-1].split(' ')[0]
        if isinstance(op, autograd.Concat):
            node.append(helper.make_node('Concat',
                                         inputs=inputs,
                                         outputs=outputs,
                                         name=op.name,
                                         axis=op.axis))
        elif isinstance(op, autograd._Conv2d):
            pads = [op.handle.padding_h, op.handle.padding_h,
                    op.handle.padding_h, op.handle.padding_h]
            stride = [op.handle.stride_h, op.handle.stride_w]
            k = [op.handle.kernel_h, op.handle.kernel_w]
            node.append(helper.make_node('Conv',
                                         inputs=inputs,
                                         outputs=outputs,
                                         name=op.name,
                                         kernel_shape=k,
                                         pads=pads,
                                         strides=stride))
            # TODO groups
        elif isinstance(op, autograd._Pooling2d):
            k = [op.handle.kernel_h, op.handle.kernel_w]
            s = [op.handle.stride_h, op.handle.stride_w]
            p = [op.handle.pad_h, op.handle.pad_h,
                 op.handle.pad_w, op.handle.pad_w]
            if (op.handle.is_max_pooling):
                node.append(helper.make_node('MaxPool',
                                             inputs=inputs,
                                             outputs=outputs,
                                             name=op.name,
                                             kernel_shape=k,
                                             pads=p,
                                             strides=s))
            else:
                node.append(helper.make_node('AveragePool',
                                             inputs=inputs,
                                             outputs=outputs,
                                             name=op.name,
                                             kernel_shape=k,
                                             pads=p,
                                             strides=s))
        elif (isinstance(op, autograd._BatchNorm2d)):
            #[(<singa.autograd.Sigmoid object at 0x7fd5ec09cb90>, 140556764852432, None, False),
            # (<singa.autograd.Dummy object at 0x7fd5ec09c390>, 140556764824208,
            # <singa.tensor.Tensor object at 0x7fd5ec09c290>, True),
            # (<singa.autograd.Dummy object at 0x7fd5ec09c490>, 140556764824528,
            # <singa.tensor.Tensor object at 0x7fd5ec09c3d0>, True),
            # (<singa.autograd.Dummy object at 0x7fd5ec09c590>, 140556764824784, None, False),
            # (<singa.autograd.Dummy object at 0x7fd5ec09c690>, 140556764825040, None, False)])
            # two dummy operators do not have values, so take the values from handle
            dummy0 = tensor.to_numpy(tensor.Tensor(device=op.running_mean.device(), data=op.running_mean))
            dummy1 = tensor.to_numpy(tensor.Tensor(device=op.running_var.device(), data=op.running_var))
            node.append(helper.make_node('BatchNormalization',
                                         inputs=inputs,
                                         outputs=outputs,
                                         name=op.name,
                                         momentum=op.handle.factor))
            dummy0=helper.make_node('Constant', inputs=[], outputs=[inputs[3]],
                                    value=numpy_helper.from_array(dummy0))
            dummy1=helper.make_node('Constant', inputs=[], outputs=[inputs[4]],
                                    value=numpy_helper.from_array(dummy1))
            node.append(dummy0)
            node.append(dummy1)
        else:
            singa2onnx = {'SoftMax': 'Softmax',
                          'AddBias': 'Add',
                          'Add': 'Add',
                          'Matmul': 'MatMul',
                          'ReLU': 'Relu',
                          'ElemMatmul': 'Mul',
                          'Flatten':'Flatten',
                          'Tanh':'Tanh'}
            if(curop in singa2onnx):onnx_op = singa2onnx[curop]
            else:onnx_op = curop
            node.append(helper.make_node(onnx_op,
                                         inputs=inputs,
                                         outputs=outputs,
                                         name=op.name))


        for srcop, yid, y, _ in op.src:
            dependency[srcop] -= 1
            if dependency[srcop] == 0:
                if isinstance(srcop, autograd.Dummy):
                    if yid not in input_ids:
                        tmp = helper.make_node(
                            'Constant',
                            inputs=[],
                            outputs=[output_name(srcop)],
                            value=helper.make_tensor(
                                name=op.name,
                                data_type=TensorProto.FLOAT,
                                dims=y.shape,
                                vals=tensor.to_numpy(y).flatten().astype(float))
                        )
                        node.append(tmp)
                else:
                    ready.append(srcop)

    onnx_model = helper.make_model(helper.make_graph(node[::-1],model_name, X, Y))
    checker.check_model(onnx_model)
    return onnx_model