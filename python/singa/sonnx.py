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


class ONNXm(Layer):

    @staticmethod
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

    @staticmethod
    def combine_node(self,modeldic):
        '''
        # for combine operators to layers
        '''

        for idx, i in enumerate(self.model.graph.node):
            if (i.op_type == 'MatMul'):
                addlist = self.find_add(i.output[0])
                if (len(addlist) == 0): continue
                if (len(addlist) > 1): continue
                addidx = addlist[0]
                if (i.name == "not_requires_grad" and self.model.graph.node[addidx].name == "not_requires_grad"): continue
                self.model.graph.node[idx].output[0] = self.model.graph.node[addidx].output[0]
                self.model.graph.node[idx].input.append(self.model.graph.node[addidx].input[1])
                self.model.graph.node[idx].op_type = 'Linear'
                self.model.graph.node[addidx].op_type = 'removed'

        self.layer = {}
        for i in self.model.graph.node:
            if (i.op_type == 'Linear'):
                shape = self.find_shape(i.input[1])
                self.layer[str(i.output[0])] = autograd.Linear(shape[0], shape[1])
                self.layer[str(i.output[0])].set_params(W=tensor.to_numpy(modeldic[str(i.input[1])]))
                self.layer[str(i.output[0])].set_params(b=tensor.to_numpy(modeldic[str(i.input[2])]))


        for i in self.model.graph.node:
            if (i.op_type == 'Conv'):
                shape = self.find_shape(i.input[1])
                self.layer[str(i.output[0])] = autograd.Conv2d(shape[1], shape[0], shape[2],
                                                          padding=int(i.attribute[0].ints[0]))
                self.layer[str(i.output[0])].set_params(W=tensor.to_numpy(modeldic[str(i.input[1])].clone()))
                self.layer[str(i.output[0])].set_params(b=tensor.to_numpy(modeldic[str(i.input[2])].clone()))

        for i in self.model.graph.node:
            if (i.op_type == 'MaxPool'):
                k = (int(i.attribute[0].ints[0]), int(i.attribute[0].ints[0]))
                self.layer[str(i.output[0])] = autograd.MaxPool2d(k, int(i.attribute[2].ints[0]),
                                                             padding=int(i.attribute[1].ints[0]))
        for i in self.model.graph.node:
            if (i.op_type == 'AveragePool'):
                k = (int(i.attribute[0].ints[0]), int(i.attribute[0].ints[0]))
                self.layer[str(i.output[0])] = autograd.AvgPool2d(k, int(i.attribute[2].ints[0]),
                                                             padding=int(i.attribute[1].ints[0]))
        for i in self.model.graph.node:
            if (i.op_type == 'BatchNormalization'):
                shape = self.find_shape(i.input[1])
                self.layer[str(i.output[0])] = autograd.BatchNorm2d(shape[0])
                self.layer[str(i.output[0])].set_params(scale=tensor.to_numpy(modeldic[str(i.input[1])].clone()))
                self.layer[str(i.output[0])].set_params(bias=tensor.to_numpy(modeldic[str(i.input[2])].clone()))


    def find_add(self,output):
        '''
        #utils for combine operators to layers
        '''
        ans = []
        for idx, i in enumerate(self.model.graph.node):
            for j in i.input:
                if j == output and i.op_type == 'Add':
                    ans.append(idx)
                    return ans

    def find_shape(self,input):
        '''
        # find weight shape for layers
        '''
        for i in self.model.graph.node:
            if (i.op_type == 'Constant' and i.output[0] == input):
                return onnx.numpy_helper.to_array(i.attribute[0].t).shape

    def __init__(self,path):
        super(ONNXm, self).__init__()
        self.modeldic, self.model = self.onnx_model_init(path)
        self.combine_node(self,self.modeldic)

    def __call__(self,inputs):
        '''
            input: input for singa model
            load other nodes of onnx
            '''
        supportLayer = ['Linear','Conv','MaxPool','AveragePool','BatchNormalization']
        #supportLayer = ['Conv', 'MaxPool', 'AveragePool', 'BatchNormalization']
        layer, model,oper = self.layer, self.model,self.modeldic

        for counter,i in enumerate(model.graph.input):
            oper[i.name] = inputs[counter]
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
            elif (i.op_type == 'CrossEntropy'):
                oper[str(i.output[0])] = autograd.cross_entropy(oper[str(i.input[0])],oper[str(i.input[1])])
            elif (i.op_type == 'SoftMaxCrossEntropy'):
                oper[str(i.output[0])] = autograd.softmax_cross_entropy(oper[str(i.input[0])], oper[str(i.input[1])])
        out =[]
        for counter,i in enumerate(model.graph.output):
            out.append(self.modeldic[i.name])
        return out


    @staticmethod
    def to_onnx_model(y,inputs):

        '''
        get onnx model from singa computational graph
        Args:
            y: a Tensor instance, usually the loss
            Return:
            loss for onnx model
        '''
        X,Y = [],[]
        node = []
        dependencylist=[]
        for counter,i in enumerate(y):
            dependency = infer_dependency(i.creator)
            dependencylist.append(dependency)
            yi = i.creator
            yi.end = True
            ready = deque([yi])
            Y = [helper.make_tensor_value_info('Y'+str(counter), TensorProto.FLOAT, i.shape)]

        supportOp = set(['ReLU', 'SoftMax', 'Add', 'AddBias', 'Matmul', 'Flatten', '_Conv2d', 'Concat', 'ElemMatmul','Sigmoid','Tanh','_Pooling2d','_BatchNorm2d','CrossEntropy','SoftMaxCrossEntropy'])
        #singatoonnx = {'SoftMax':'Softmax','AddBias':'Add','Matmul':'MatMul','ReLU':'Relu','_Conv2d':'Conv','ElemMatmul':'Mul','_Pooling2d':'MaxPool','_BatchNorm2d':'BatchNormalization','CrossEntropy':'Or','SoftMaxCrossEntropy':'Xor'}
        singatoonnx = {'SoftMax': 'Softmax', 'AddBias': 'Add', 'Matmul': 'MatMul', 'ReLU': 'Relu', '_Conv2d': 'Conv',
                       'ElemMatmul': 'Mul', '_Pooling2d': 'MaxPool', '_BatchNorm2d': 'BatchNormalization'}
        lastop=0
        counterX = 0
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
                if not op.requires_grad:
                    name = "not_requires_grad"
                    print('not requres')
                else:name=''
                if (curop in singatoonnx): curop = singatoonnx[curop]
                if (hasattr(op, 'end')):
                    if(isinstance(op,SoftMaxCrossEntropy)):
                        # singa.autograd.SoftMaxCrossEntropy does not have dummy in op.src
                        X = [helper.make_tensor_value_info(str(op.t), TensorProto.FLOAT,
                                                           inputs[len(inputs) - 1 - counterX].shape)] + X
                        pre.append(str(op.t))
                        counterX += 1
                    node = [onnx.helper.make_node(curop, inputs=pre, outputs=['Y'+str(lastop)],name=name )] + node
                    lastop+=1
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
                num = 0
                while(True):
                    if (len(op.src) > num and isinstance(op.src[num][0],Dummy) and op.src[num][2] is not None):
                        dummy = to_numpy(op.src[num][2])
                        node = [onnx.helper.make_node('Constant', inputs=[], outputs=[pre[num]],
                                                      value=numpy_helper.from_array(dummy))] + node
                    elif (len(op.src) > num and isinstance(op.src[num][0], Dummy) and op.src[num][2] is None):
                        #X = [helper.make_tensor_value_info(pre[num], TensorProto.FLOAT,inputs[len(inputs)-1-counterX].shape)]+X
                        X = [helper.make_tensor_value_info(pre[num], TensorProto.FLOAT,
                                                           inputs[len(inputs) - 1 - counterX].shape)] + X
                        counterX+=1

                    num+=1
                    if(len(op.src) <= num):break
            if not op.requires_grad:continue
            for (src_op, x_id, y, y_stores_grad) in op.src:
                for i in range(len(dependencylist)):
                    if(src_op in dependencylist[i]):
                        dependency = dependencylist[i]
                        break
                dependency[src_op] -= 1
                if src_op.requires_grad is True:
                    if dependency[src_op] == 0:
                        if not isinstance(src_op, Dummy):ready.append((src_op))
        model_def = helper.make_model(helper.make_graph(node, "t", X, Y, ), producer_name='o')
        #onnx.checker.check_model(model_def)
        return model_def


def from_onnx_model(path):
    return ONNXm(path)





