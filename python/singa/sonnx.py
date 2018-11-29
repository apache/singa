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

def onnx_loss(a,model,target):
    '''
    input:
    a graph node dictionary
    model: graph model
    target: label

    load other nodes of onnx
    '''
    for i in model.graph.node:
        if (i.op_type == 'Constant'):
            pass
            # do nothing
        if (i.op_type == 'LeakyRelu'):
            a[str(i.output[0])] = autograd.relu(a[str(i.input[0])])
        elif (i.op_type == 'Relu'):
            a[str(i.output[0])] = autograd.relu(a[str(i.input[0])])
        elif (i.op_type == 'Softmax'):
            a[str(i.output[0])] = autograd.softmax(a[str(i.input[0])])
        elif (i.op_type == 'Add'):
            if(str(i.input[1])[-1] == 'b'):
                a[str(i.output[0])] = autograd.add_bias(a[str(i.input[0])], a[str(i.input[1])])
            else:
                a[str(i.output[0])] = autograd.add(a[str(i.input[0])],a[str(i.input[1])])
        elif (i.op_type == 'MatMul'):
            a[str(i.output[0])] = autograd.matmul(a[str(i.input[0])], a[str(i.input[1])])

    loss = autograd.cross_entropy(a['Y'], target)
    return loss



def get_onnx_model(y,inputs,target, dy=None):
    '''
	get onnx model from singa computational graph
	Args:
        y: a Tensor instance, usually the loss
        dy: a number or a Tensor instance, for the gradient of the
            objective/loss w.r.t y, usually 1.0
        Return:
        loss for onnx model
    '''
    ######################

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT,inputs.shape)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT,target.shape)
    node = []
    ######################

    dependency = infer_dependency(y.creator)

    assert y.size() == 1, 'y must be a Tensor with a single value;' \
                          'size of y is % d' % y.size()

    # by default the dy is a tensor with 1.0 for each sample;
    if dy is None:
        dy = float(1.0)
    elif isinstance(dy, Tensor):
        dy = dy.data
    else:
        dy = float(dy)

    # ready is a queue of (operation, dy list)
    ready = deque([(y.creator, (dy,))])
    not_ready = {}  # mapping: op->[dy]
    gradients = {}  # mapping: x->dx if x.stores_grad
    if y.stores_grad:
        gradients[y] = dy

    supportOp = set(['LeakyRelu','Softmax','Add','MatMul','Flatten'])

    while len(ready) > 0:
        op, dys = ready.pop()
        if not op.requires_grad or isinstance(op, Dummy):
            continue
        # if not isinstance(op, tensor.Dummy):
        dxs = op._do_backward(*dys)
        ##############################
        curname = str(op).split('.')[-1].split(' ')[0]
        prefname = str(op.src[0][0]).split('.')[-1].split(' ')[0]
        cur = str(op)
        pre = [str(i[0]) for i in op.src]
        if op.param['name'] in supportOp:
            if (prefname == 'Dummy'): pre[0] = 'X'
            if (op.param['name'] == 'Softmax'):
                node = [onnx.helper.make_node('Softmax', inputs=pre, outputs=['Y'], )] + node
            else:
                node = [onnx.helper.make_node(op.param['name'], inputs=pre, outputs=[cur], )] + node
                num = 1
                if 'b' in op.param:
                    b = ctensor2numpy(op.param['b'])
                    node = [onnx.helper.make_node('Constant', inputs=[], outputs=[pre[num]],
                                                  value=numpy_helper.from_array(b))] + node
                    num+=1
                if 'w' in op.param:
                    w = ctensor2numpy(op.param['w'])
                    node = [onnx.helper.make_node('Constant', inputs=[], outputs=[pre[num]],
                                                  value=numpy_helper.from_array(w))] + node
                    num+=1
        ##################################
        # TODO src and dx must match
        assert len(op.src) == len(dxs), \
            'the number of src ops (=%d) and dx (=%d) not match' \
            % (len(op.src), len(dxs))
        for (src_op, x_id, y, y_stores_grad), dx in zip(op.src, dxs):
            # prefix x is w.r.t op; prefix y is w.r.t src_op.
            # x_id is the python id of one input arg of src_op, denoted as x.
            # y_idx (below) is the index of x among the outputs of src_op.
            # not_ready[src_op][y_idx] records the intermediate gradient
            # of the y_idx'th output of src_op. 'intermediate gradient'
            # indicates that if this output is used in multiple children
            # operations, then we have to add the graident (dx) from all these
            # children operations. When src_op is ready, it means that
            # the gradient of all its outputs are available, i.e. all children
            # operations have been backwarded.
            # y is None if y.stores_grad is false; otherwise it is a Tensor
            y_idx = src_op.y_id2idx[x_id]
            if src_op not in not_ready:
                # src_op may have mulitple outputs
                not_ready[src_op] = [None for _ in src_op.y_id2idx]
                not_ready[src_op][y_idx] = dx
            else:
                pass
                #dxs = not_ready[src_op]
                #if dxs[y_idx] is None:
                #    dxs[y_idx] = dx
                #else:
                    # add the gradient from another children operation that
                    # uses y_idx'th output of src_op as input arg
                #    dxs[y_idx] += dx
            if y_stores_grad:
                pass
            dependency[src_op] -= 1
            if src_op.requires_grad is True:
                if dependency[src_op] == 0:
                    if not isinstance(src_op, Dummy):
                        ready.append((src_op, not_ready[src_op]))
                    del not_ready[src_op]
    ###############################################
    model_def = helper.make_model(helper.make_graph(node, "t", [X], [Y], ), producer_name='o')
    onnx.checker.check_model(model_def)
    ###############################################
    return model_def

