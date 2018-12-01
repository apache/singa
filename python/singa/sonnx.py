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

def onnx_model_init(inputs,name):
    '''
    load onnx model graph and load weights
    input:
    input data and file name of onnx model

    return:
     a graph node dictionary
     model: graph model
    '''
    model = onnx.load('singa.onnx')
    a = {}
    a['X'] = inputs
    for i in model.graph.node:
        if (i.op_type == 'Constant'):
            a[str(i.output[0])] = tensor.from_numpy(onnx.numpy_helper.to_array(i.attribute[0].t))
            a[str(i.output[0])].stores_grad = True
    return a,model

def find_add(output,model):
    ans = []
    for idx, i in enumerate(model.graph.node):
        for j in i.input:
            if j == output and i.op_type == 'Add':
                ans.append(idx)
    return ans

def find_shape(input,model):
    for i in model.graph.node:
        if(i.op_type == 'Constant' and i.output[0]==input):
            return onnx.numpy_helper.to_array(i.attribute[0].t).shape


def combine_node(a,model):
    for idx,i in enumerate(model.graph.node):
        if (i.op_type == 'MatMul'):
            addlist = find_add(i.output[0],model)
            if(len(addlist) > 1 or len(addlist)==0):continue
            addidx = addlist[0]
            if(i.name == "not_requires_grad" and model.graph.node[addidx].name == "not_requires_grad"):continue
            model.graph.node[idx].output[0] = model.graph.node[addidx].output[0]
            model.graph.node[idx].input.append(model.graph.node[addidx].input[1])
            model.graph.node[idx].op_type = 'Linear'
            model.graph.node[addidx].op_type='removed'

    linear={}
    for i in model.graph.node:
        if (i.op_type == 'Linear'):
            shape = find_shape(i.input[1],model)
            linear[str(i.output[0])] = autograd.Linear(shape[0], shape[1])
            linear[str(i.output[0])].w = a[str(i.input[1])]
            linear[str(i.output[0])].b = a[str(i.input[2])]


    return linear,model

def onnx_loss(a,model,target):
    '''
    input:
    a graph node dictionary
    model: graph model
    target: label

    load other nodes of onnx
    '''

    linear, model = combine_node(a,model)
    for i in model.graph.node:
        if (i.op_type == 'Constant'):
            pass
            # do nothing
        if (i.op_type == 'Relu'):
            a[str(i.output[0])] = autograd.relu(a[str(i.input[0])])
        elif (i.op_type == 'Softmax'):
            a[str(i.output[0])] = autograd.softmax(a[str(i.input[0])])
        elif (i.op_type == 'Add'):
            a[str(i.output[0])] = autograd.add(a[str(i.input[0])],a[str(i.input[1])])
        elif (i.op_type == 'MatMul'):
            a[str(i.output[0])] = autograd.matmul(a[str(i.input[0])], a[str(i.input[1])])
        elif (i.op_type == 'Linear'):
            a[str(i.output[0])] = linear[str(i.output[0])](a[str(i.input[0])])

    loss = autograd.cross_entropy(a['Y'], target)
    return loss



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

    supportOp = set(['ReLU','SoftMax','Add','AddBias','Matmul','Flatten'])
    singatoonnx = {'SoftMax':'Softmax','AddBias':'Add','Matmul':'MatMul','ReLU':'Relu'}
    lastop=True
    while len(ready) > 0:
        op = ready.pop()
        if isinstance(op, Dummy):continue
        curop = str(op).split('.')[-1].split(' ')[0]
        cur = str(op)
        pre = [str(i[0]) for i in op.src]
        preop = [str(i[0]).split('.')[-1].split(' ')[0] for i in op.src]
        prefname = preop[0]
        if curop in supportOp:
            if not op.requires_grad:name = "not_requires_grad"
            else:name=''
            if (prefname == 'Dummy'): pre[0] = 'X'
            if (curop in singatoonnx): curop = singatoonnx[curop]
            if (lastop):
                node = [onnx.helper.make_node(curop, inputs=pre, outputs=['Y'],name=name )] + node
                lastop = False
            else:
                node = [onnx.helper.make_node(curop, inputs=pre, outputs=[cur],name=name )] + node
                num = 1
                while(True):
                    if (len(pre) > num and preop[num] == 'Dummy' and op.src[num][2] is not None):
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


