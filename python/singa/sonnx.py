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
import numpy as np
import onnx
from onnx import numpy_helper

from collections import Counter, deque
import numpy as np
import math

from .tensor import Tensor
from . import layer
from singa.proto import model_pb2
from . import singa_wrap as singa
#from .tensor import einsum
from autograd import *

def load_onnx_model(name = 'singonnx.pkl'):
    '''
        load onnx model
        Args:
            name:name of onnx model file
        Return:
            onnx model
    '''
    with open(name, 'rb') as f:
        model = pickle.load(f)
    return model

def onnx_model_init(inputs,model):
    '''
    init onnx model
    Args:
        inputs: input data for model
        model: onnx model
    Return:
        a dictionary for save all of the node of singa computation graph
    '''
    a = {}
    a['X'] = inputs
    for i in model.graph.node:
        if (i.op_type == 'Constant'):
            a[str(i.output[0])] = tensor.from_numpy(numpy_helper.to_array(i.attribute[0].t))
            a[str(i.output[0])].stores_grad = True
    return a

def onnx_loss(a,model,target):
    '''
    get onnx model loss
    Args:
        a: singa computational graph nodes dictionary
        model: onnx model
        target:
    Return:
        loss for onnx model
    '''
    for i in model.graph.node:
        if (i.op_type == 'Constant'):
            pass
            # do nothing
        if (i.op_type == 'LeakyRelu'):
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



def get_onnx_model(y, dy=None):
    '''
    from singa computational graph(loss) to get onnx model
    Args:
        y:loss of singa autograd
    Return:
        onnx model
    '''
    ######################
    import onnx
    from onnx import helper
    from onnx import AttributeProto, TensorProto, GraphProto
    from onnx import numpy_helper
    import numpy as np
    from singa import tensor
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2])
    node = []
    ######################

    dependency = infer_dependency(y.creator)
    #get singa computational graph dependency

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
    # using bfs iterative method to check all computational graph nodes
    not_ready = {}  # mapping: op->[dy]
    gradients = {}  # mapping: x->dx if x.stores_grad
    if y.stores_grad:
        gradients[y] = dy

    while len(ready) > 0:
        op, dys = ready.pop()
        #check all children nodes of this node
        if not op.requires_grad or isinstance(op, Dummy):
            continue
        # if not isinstance(op, tensor.Dummy):
        dxs = op._do_backward(*dys)
        ##############################
        cur = str(op).split('.')[-1].split(' ')[0]
        pre = str(op.src[0][0]).split('.')[-1].split(' ')[0]
        cc = str(op).split(' ')[-1]
        pc = str(op.src[0][0]).split(' ')[-1]
        cstrd = str(op) + str(dependency[op])
        pstrd = str(op.src[0][0]) + str(dependency[op.src[0][0]])
        cstr = str(op)
        pstr = str(op.src[0][0])
        #get father node's name and children nodes' names

        if (pre == 'Dummy'): pstr = pre = 'X'
        # if it is father node is dummy, the node will be input x
        if (op.param['name'] == 'LeakyRelu'):
        # save autograd layers param name, and use here for saving onnx model
            node = [onnx.helper.make_node('LeakyRelu', inputs=[pstr], outputs=[cstr], )] + node
        elif (op.param['name'] == 'Softmax'):
            node = [onnx.helper.make_node('Softmax', inputs=[pstr], outputs=['Y'], )] + node
        elif (op.param['name'] == 'AddBias'):
            node = [onnx.helper.make_node('Add', inputs=[pstr, pstrd + 'b'], outputs=[cstr], )] + node
            b = ctensor2numpy(op.param['b'])
            node = [onnx.helper.make_node('Constant', inputs=[], outputs=[pstrd + 'b'],value=numpy_helper.from_array(b))] + node
        elif (op.param['name'] == 'Add'):
            node = [onnx.helper.make_node('Add', inputs=[pstr, str(op.src[1][0])], outputs=[cstr], )] + node
        elif (op.param['name'] == 'MatMul'):
            node = [onnx.helper.make_node('MatMul', inputs=[pstr, pstrd + 'w'], outputs=[cstr], )] + node
            w = ctensor2numpy(op.param['w'])
            node = [onnx.helper.make_node('Constant', inputs=[], outputs=[pstrd + 'w'],value=numpy_helper.from_array(w))] + node
        elif (op.param['name'] == 'linear'):
            pass
            '''
            node = [onnx.helper.make_node('MatMul', inputs=[pstr, pstrd + 'w'], outputs=[cstr], )] + node
            w = tensor.to_numpy(Tensor(data=op.param['w'], device=op.param['w'].device))
            node = [onnx.helper.make_node('Constant', inputs=[], outputs=[pstrd + 'w'],
                                          value=numpy_helper.from_array(w))] + node
            b = tensor.to_numpy(Tensor(data=op.param['w'], device=op.param['w'].device))
            node = [onnx.helper.make_node('Constant', inputs=[], outputs=[pstrd + 'w'],
                                          value=numpy_helper.from_array(w))] + node
            '''
        elif (op.param['name'] == 'Flatten'):
            node = [onnx.helper.make_node('Flatten', inputs=[pstr], outputs=[cstr], )] + node
        else:
            pass
        ##get all computational graph nodes and save for onnx nodes
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
                dxs = not_ready[src_op]
                if dxs[y_idx] is None:
                    dxs[y_idx] = dx
                else:
                    # add the gradient from another children operation that
                    # uses y_idx'th output of src_op as input arg
                    dxs[y_idx] += dx
            if y_stores_grad:
                # store the gradient for final return, e.g. if x is parameter
                g = not_ready[src_op][y_idx]
                #gradients[y] = Tensor(device=g.device, data=g)
            dependency[src_op] -= 1
            if src_op.requires_grad is True:
                if dependency[src_op] == 0:
                    if not isinstance(src_op, Dummy):
                        ready.append((src_op, not_ready[src_op]))
                    del not_ready[src_op]
    ###############################################
    # specific input data x, output data y and all intermediary nodes
    model_def = helper.make_model(helper.make_graph(node, "t", [X], [Y], ), producer_name='o')
    onnx.checker.check_model(model_def)
    # check is there any problem of onnx model
    ###############################################
    return model_def

