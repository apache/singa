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
import pickle
from singa import tensor
from singa.tensor import Tensor
from singa import autograd
from singa import optimizer
import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
from onnx import numpy_helper
from collections import Counter, deque
import math

from .tensor import Tensor
from . import layer
from singa.proto import model_pb2
from . import singa_wrap as singa
#from .tensor import einsum
from autograd import *
from singa.tensor import to_numpy
from autograd import _Conv2d,_Pooling2d,_BatchNorm2d

def onnx_model_init(path):
    '''
    input path

    return: model and model dictionary
    '''
    print('path:', path)
    model = onnx.load(path)
    modeldic = {}
    for i in model.graph.node:
        if (i.op_type == 'Constant'):
            modeldic[str(i.output[0])] = tensor.from_numpy(onnx.numpy_helper.to_array(i.attribute[0].t))
            modeldic[str(i.output[0])].stores_grad = True
    return modeldic, model


def find_add(output, model):
    '''
    #utils for combine operators to layers
    '''
    ans = []
    for idx, i in enumerate(model.graph.node):
        for j in i.input:
            if j == output and i.op_type == 'Add':
                ans.append(idx)
    return ans


def find_shape(input, model):
    '''
    # find weight shape for layers
    '''
    for i in model.graph.node:
        if (i.op_type == 'Constant' and i.output[0] == input):
            return onnx.numpy_helper.to_array(i.attribute[0].t).shape



def combine_node(modeldic, model):
    '''
    # for combine operators to layers
    '''
    for idx, i in enumerate(model.graph.node):
        if (i.op_type == 'MatMul'):
            addlist = find_add(i.output[0], model)
            if (len(addlist) > 1 or len(addlist) == 0): continue
            addidx = addlist[0]
            if (i.name == "not_requires_grad" and model.graph.node[addidx].name == "not_requires_grad"): continue
            model.graph.node[idx].output[0] = model.graph.node[addidx].output[0]
            model.graph.node[idx].input.append(model.graph.node[addidx].input[1])
            model.graph.node[idx].op_type = 'Linear'
            model.graph.node[addidx].op_type = 'removed'

    layer = {}
    for i in model.graph.node:
        if (i.op_type == 'Linear'):
            shape = find_shape(i.input[1], model)
            layer[str(i.output[0])] = autograd.Linear(shape[0], shape[1])
            layer[str(i.output[0])].set_params(W=modeldic[str(i.input[1])])
            layer[str(i.output[0])].set_params(b=modeldic[str(i.input[2])])

    for i in model.graph.node:
        if (i.op_type == 'Conv'):
            shape = find_shape(i.input[1], model)
            layer[str(i.output[0])] = autograd.Conv2d(shape[1], shape[0],shape[2],padding=int(i.attribute[0].ints[0]))
            layer[str(i.output[0])].set_params(W=modeldic[str(i.input[1])])
            layer[str(i.output[0])].set_params(b=modeldic[str(i.input[2])])

    for i in model.graph.node:
        if (i.op_type == 'MaxPool'):
            k = (int(i.attribute[0].ints[0]),int(i.attribute[0].ints[0]))
            layer[str(i.output[0])] = autograd.MaxPool2d(k, int(i.attribute[2].ints[0]),padding=int(i.attribute[1].ints[0]))
    for i in model.graph.node:
        if (i.op_type == 'AveragePool'):
            k = (int(i.attribute[0].ints[0]),int(i.attribute[0].ints[0]))
            layer[str(i.output[0])] = autograd.AvgPool2d(k, int(i.attribute[2].ints[0]),padding=int(i.attribute[1].ints[0]))
    for i in model.graph.node:
        if (i.op_type == 'BatchNormalization'):
            shape = find_shape(i.input[1], model)
            layer[str(i.output[0])] = autograd.BatchNorm2d(shape[0])
            layer[str(i.output[0])].set_params(scale=modeldic[str(i.input[1])])
            layer[str(i.output[0])].set_params(bias=modeldic[str(i.input[2])])

    return layer, model

class ONNXm(Layer):
    def __init__(self,path):
        super(ONNXm, self).__init__()
        self.modeldic, self.model = onnx_model_init(path)
        self.layer, self.model = combine_node(self.modeldic, self.model)

    def __call__(self,inputs):
        '''
            input: input for singa model
            load other nodes of onnx
            '''
        supportLayer = ['Linear','Conv','MaxPool','AveragePool','BatchNormalization']
        layer, model,oper = self.layer, self.model,self.modeldic
        self.modeldic['X'] = inputs
        for i in model.graph.node:
            if (i.op_type == 'Relu'):
                oper[str(i.output[0])] = autograd.relu(oper[str(i.input[0])])
            elif (i.op_type == 'Softmax'):
                oper[str(i.output[0])] = autograd.softmax(oper[str(i.input[0])])
            elif (i.op_type == 'Add'):
                oper[str(i.output[0])] = autograd.add(oper[str(i.input[0])], oper[str(i.input[1])])
            elif (i.op_type == 'MatMul'):
                oper[str(i.output[0])] = autograd.matmul(oper[str(i.input[0])], oper[str(i.input[1])])
            elif (i.op_type == 'Flatten'):
                oper[str(i.output[0])] = autograd.flatten(oper[str(i.input[0])])
            elif(i.op_type == 'Concat'):
                oper[str(i.output[0])] = autograd.cat((oper[str(i.input[0])], oper[str(i.input[1])]),int(i.attribute[0].i))
            elif(i.op_type == 'Tanh'):
                oper[str(i.output[0])] = autograd.tanh(oper[str(i.input[0])])
            elif (i.op_type == 'Sigmoid'):
                oper[str(i.output[0])] = autograd.sigmoid(oper[str(i.input[0])])
            elif (i.op_type == 'Mul'):
                oper[str(i.output[0])] = autograd.mul(oper[str(i.input[0])],oper[str(i.input[1])])
            elif (i.op_type in supportLayer):
                oper[str(i.output[0])] = layer[str(i.output[0])](oper[str(i.input[0])])
        print('finish farward')
        return oper['Y']



def from_onnx_model(path):
    return ONNXm(path)

def get_onnx_model(y,inputs,target):
    '''
	get onnx model from singa computational graph
	Args:
        y: a Tensor instance, usually the loss
        Return:
        loss for onnx model
    '''
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT,inputs.shape)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT,target.shape)
    node = []
    dependency = infer_dependency(y.creator)

    assert y.size() == 1, 'y must be a Tensor with a single value;' \
                          'size of y is % d' % y.size()

    ready = deque([y.creator])

    supportOp = set(['ReLU', 'SoftMax', 'Add', 'AddBias', 'Matmul', 'Flatten', '_Conv2d', 'Concat', 'ElemMatmul','Sigmoid','Tanh','_Pooling2d','_BatchNorm2d'])
    singatoonnx = {'SoftMax':'Softmax','AddBias':'Add','Matmul':'MatMul','ReLU':'Relu','_Conv2d':'Conv','ElemMatmul':'Mul','_Pooling2d':'MaxPool','_BatchNorm2d':'BatchNormalization'}
    lastop=True
    while len(ready) > 0:
        op = ready.pop()
        if isinstance(op, Dummy):continue
        curop = str(op).split('.')[-1].split(' ')[0]
        cur = str(op)
        pre = [str(i[0]) for i in op.src]
        preop = [str(i[0]).split('.')[-1].split(' ')[0] for i in op.src]
        #print('op',op)
        #print('op scr', op.src)
        #print('-------')
        if curop in supportOp:
            if not op.requires_grad:name = "not_requires_grad"
            else:name=''
            if (isinstance(op.src[0][0], Dummy)): pre[0] = 'X'
            if (curop in singatoonnx): curop = singatoonnx[curop]
            if (lastop):
                node = [onnx.helper.make_node(curop, inputs=pre, outputs=['Y'],name=name )] + node
                lastop = False
            else:
                if(isinstance(op,Concat)):
                    node = [onnx.helper.make_node(curop, inputs=pre, outputs=[cur], name=name,axis=int(op.axis))] + node
                elif(isinstance(op,_Conv2d)):
                    pads=[op.handle.padding_h,op.handle.padding_h,op.handle.padding_h,op.handle.padding_h]
                    stride=[op.handle.stride_h,op.handle.stride_w]
                    node = [onnx.helper.make_node(curop, inputs=pre, outputs=[cur], name=name, pads=pads,strides=stride)] + node
                elif(isinstance(op,_Pooling2d)):

                    k = [op.handle.kernel_h, op.handle.kernel_w]
                    s = [op.handle.stride_h, op.handle.stride_w]
                    p = [op.handle.pad_h,op.handle.pad_h, op.handle.pad_w,op.handle.pad_w]
                    if (op.handle.is_max_pooling):
                        node = [onnx.helper.make_node(curop, inputs=pre, outputs=[cur], name=name,kernel_shape=k,pads=p,strides=s)] + node
                    else:
                        node = [onnx.helper.make_node('AveragePool', inputs=pre, outputs=[cur], name=name, kernel_shape=k,
                                                      pads=p, strides=s)] + node
                elif (isinstance(op, _BatchNorm2d)):
                    pre.append(cur + 'op.running_mean')
                    pre.append(cur + 'op.running_var')
                    dummy0 = to_numpy(tensor.Tensor(device=op.running_mean.device(), data=op.running_mean))
                    dummy1 = to_numpy(tensor.Tensor(device=op.running_mean.device(), data=op.running_mean))
                    node = [onnx.helper.make_node(curop, inputs=pre, outputs=[cur], name=name)] + node

                    node = [onnx.helper.make_node('Constant', inputs=[], outputs=[pre[3]],
                                                  value=numpy_helper.from_array(dummy0))] + node
                    node = [onnx.helper.make_node('Constant', inputs=[], outputs=[pre[4]],
                                                  value=numpy_helper.from_array(dummy1))] + node
                else:
                    node = [onnx.helper.make_node(curop, inputs=pre, outputs=[cur],name=name )] + node
            num = 1
            while(True):
                if (len(op.src) > num and isinstance(op.src[num][0],Dummy) and op.src[num][2] is not None):
                    dummy = to_numpy(op.src[num][2])
                    node = [onnx.helper.make_node('Constant', inputs=[], outputs=[pre[num]],
                                                  value=numpy_helper.from_array(dummy))] + node
                    num+=1
                else:break
        if not op.requires_grad:continue
        for (src_op, x_id, y, y_stores_grad) in op.src:
            dependency[src_op] -= 1
            if src_op.requires_grad is True:
                if dependency[src_op] == 0:
                    if not isinstance(src_op, Dummy):ready.append((src_op))
    model_def = helper.make_model(helper.make_graph(node, "t", [X], [Y], ), producer_name='o')
    onnx.checker.check_model(model_def)
    return model_def


