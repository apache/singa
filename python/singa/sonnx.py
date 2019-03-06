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
from singa import tensor
from singa import autograd
from onnx import helper,checker
from onnx import AttributeProto, TensorProto, GraphProto
from onnx import numpy_helper
from  onnx.backend.base import BackendRep as backendRep
from  onnx.backend.base import Backend as backend
from collections import Counter, deque

from . import singa_wrap as singa
from autograd import *
from autograd import _Conv2d,_Pooling2d,_BatchNorm2d
#if not import, there will be an error
from singa.tensor import to_numpy


class BackendRep(backendRep):
    def __init__(self,model,device):
        self.model, self.modeldic = Backend.onnx_model_init(model,device)
        self.handledic={}
    def run(self,inputs):
        self.y,self.modeldic=Backend.run(self.model, self.modeldic,inputs,self.handledic)
        return self.y



class Backend(backend):

    @staticmethod
    def convhandle(name,handledic,x,model):
        if(name in handledic):return handledic
        i = Backend.find_name(model,name)

        shape = Backend.find_shape(model,i.input[1])
        cin,cout,k=shape[1], shape[0], (shape[2],shape[2])
        padding=(int(i.attribute[1].ints[0]),int(i.attribute[1].ints[0]))
        stride=(int(i.attribute[2].ints[0]),int(i.attribute[2].ints[0]))

        handledic[name] = singa.CudnnConvHandle(x.data, k, stride,padding, cin, cout, True)
        handledic[name].device_id = x.device.id()
        return handledic


    @staticmethod
    def MaxPool2dhandle(name,handledic,x,model):
        if(name in handledic):return handledic
        i = Backend.find_name(model,name)
        k = (int(i.attribute[0].ints[0]),int(i.attribute[0].ints[0]))
        padding=(int(i.attribute[1].ints[0]),int(i.attribute[1].ints[0]))
        stride=(int(i.attribute[2].ints[0]),int(i.attribute[2].ints[0]))

        handledic[name] = singa.CudnnPoolingHandle(x.data, k, stride, padding, True)
        handledic[name].device_id = x.device.id()
        return handledic

    @staticmethod
    def AveragePoolhandle(name,handledic,x,model):
        if(name in handledic):return handledic
        i = Backend.find_name(model,name)
        k = (int(i.attribute[0].ints[0]),int(i.attribute[0].ints[0]))
        padding=(int(i.attribute[1].ints[0]),int(i.attribute[1].ints[0]))
        stride=(int(i.attribute[2].ints[0]),int(i.attribute[2].ints[0]))

        handledic[name] = singa.CudnnPoolingHandle(x.data, k, stride, padding, False)
        handledic[name].device_id = x.device.id()
        return handledic

    @staticmethod
    def BatchNormalizationhandle(name,handledic,x,model):
        if(name in handledic):return handledic
        handledic[name] = singa.CudnnBatchNormHandle(0.9, x.data)
        handledic[name].device_id = x.device.id()
        return handledic






    @staticmethod
    def onnx_model_init(model,device):
        '''
        input model

        return: model and model dictionary
        '''

        modeldic = {}
        for i in model.graph.node:
            if (i.op_type == 'Constant'):
                modeldic[str(i.output[0])] = tensor.Tensor(device=device,data=numpy_helper.to_array(i.attribute[0].t),requires_grad=True, stores_grad=True)

        return model,modeldic


    @staticmethod
    def find_name(model,name):
        for i in model.graph.node:
            if (i.name == name):
                return i


    @staticmethod
    def find_shape(model,input):
        '''
        # find weight shape for layers
        '''
        for i in model.graph.node:
            if (i.op_type == 'Constant' and i.output[0] == input):
                return numpy_helper.to_array(i.attribute[0].t).shape


    @staticmethod
    def run_model(model,inputs,device):
        model, modeldic  = Backend.onnx_model_init(model,device)
        return Backend.run(model, modeldic,inputs)[0]

    @staticmethod
    def run(model, modeldic,inputs,handledic={}):
        '''
            input: input for singa model
            load other nodes of onnx
            '''
        supportLayer = ['Conv','MaxPool','AveragePool','BatchNormalization']
        oper=modeldic
        autograd.training = True
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
            elif (i.op_type == 'Conv'):
                handledic = Backend.convhandle(i.name,handledic,oper[str(i.input[0])],model)
                oper[str(i.output[0])] = autograd.conv2d(handledic[i.name],oper[str(i.input[0])].clone(),oper[str(i.input[1])].clone(),oper[str(i.input[2])].clone())
            elif (i.op_type == 'MaxPool'):
                handledic = Backend.MaxPool2dhandle(i.name,handledic,oper[str(i.input[0])],model)
                oper[str(i.output[0])] = autograd.pooling_2d(handledic[i.name],oper[str(i.input[0])])
            elif (i.op_type == 'AveragePool'):
                handledic = Backend.AveragePoolhandle(i.name,handledic,oper[str(i.input[0])],model)
                oper[str(i.output[0])] = autograd.pooling_2d(handledic[i.name],oper[str(i.input[0])])
            elif (i.op_type == 'BatchNormalization'):
                handledic = Backend.BatchNormalizationhandle(i.name,handledic,oper[str(i.input[0])],model)
                oper[str(i.output[0])] = autograd.batchnorm_2d(handledic[i.name],oper[str(i.input[0])],oper[str(i.input[1])],oper[str(i.input[2])],oper[str(i.input[3])],oper[str(i.input[4])])
        out =[]
        for counter,i in enumerate(model.graph.output):
            out.append(oper[i.name])
        return out,oper



def to_onnx_model(inputs,y):

    '''
    get onnx model from singa computational graph
    Args:
        y: a Tensor instance, usually the loss
        Return:
        loss for onnx model
    '''
    X,Y = [],[]
    node = []
    dependency = infer_dependency(y.creator)
    yi = y.creator
    yi.end = True
    ready = deque([yi])
    Y = [helper.make_tensor_value_info('Y'+str(0), TensorProto.FLOAT, y.shape)]

    supportOp = set(['ReLU', 'SoftMax', 'Add', 'AddBias', 'Matmul', 'Flatten', '_Conv2d', 'Concat', 'ElemMatmul','Sigmoid','Tanh','_Pooling2d','_BatchNorm2d'])
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
        layercnt=0
        if curop in supportOp:
            name=str(op)+str(layercnt)
            layercnt+=1
            if (curop in singatoonnx): curop = singatoonnx[curop]
            if (hasattr(op, 'end')):
                if(isinstance(op,SoftMaxCrossEntropy)):
                    # singa.autograd.SoftMaxCrossEntropy does not have dummy in op.src
                    X = [helper.make_tensor_value_info(str(op.t), TensorProto.FLOAT,
                                                       inputs[len(inputs) - 1 - counterX].shape)] + X
                    pre.append(str(op.t))
                    counterX += 1
                node = [helper.make_node(curop, inputs=pre, outputs=['Y'+str(lastop)],name=name )] + node
                lastop+=1
            else:
                if(isinstance(op,Concat)):
                    node = [helper.make_node(curop, inputs=pre, outputs=[cur], name=name,axis=int(op.axis))] + node
                elif(isinstance(op,_Conv2d)):
                    pads=[op.handle.padding_h,op.handle.padding_h,op.handle.padding_h,op.handle.padding_h]
                    stride=[op.handle.stride_h,op.handle.stride_w]
                    k = [op.handle.kernel_h, op.handle.kernel_w]
                    node = [helper.make_node(curop, inputs=pre, outputs=[cur], name=name, kernel_shape=k,pads=pads,strides=stride)] + node
                elif(isinstance(op,_Pooling2d)):
                    k = [op.handle.kernel_h, op.handle.kernel_w]
                    s = [op.handle.stride_h, op.handle.stride_w]
                    p = [op.handle.pad_h,op.handle.pad_h, op.handle.pad_w,op.handle.pad_w]
                    if (op.handle.is_max_pooling):
                        node = [helper.make_node(curop, inputs=pre, outputs=[cur], name=name,kernel_shape=k,pads=p,strides=s)] + node
                    else:
                        node = [helper.make_node('AveragePool', inputs=pre, outputs=[cur], name=name, kernel_shape=k,
                                                 pads=p, strides=s)] + node
                elif (isinstance(op, _BatchNorm2d)):
                    pre.append(cur + 'op.running_mean')
                    pre.append(cur + 'op.running_var')
                    dummy0 = to_numpy(tensor.Tensor(device=op.running_mean.device(), data=op.running_mean))
                    dummy1 = to_numpy(tensor.Tensor(device=op.running_var.device(), data=op.running_var))
                    node = [helper.make_node(curop, inputs=pre, outputs=[cur], name=name)] + node

                    node = [helper.make_node('Constant', inputs=[], outputs=[pre[3]],
                                             value=numpy_helper.from_array(dummy0))] + node
                    node = [helper.make_node('Constant', inputs=[], outputs=[pre[4]],
                                             value=numpy_helper.from_array(dummy1))] + node
                else:
                    node = [helper.make_node(curop, inputs=pre, outputs=[cur],name=name )] + node
            num = 0
            while(True):
                if (len(op.src) > num and isinstance(op.src[num][0],Dummy) and op.src[num][2] is not None):
                    dummy = to_numpy(op.src[num][2])
                    node = [helper.make_node('Constant', inputs=[], outputs=[pre[num]],
                                             value=numpy_helper.from_array(dummy))] + node
                elif (len(op.src) > num and isinstance(op.src[num][0], Dummy) and op.src[num][2] is None):
                    X = [helper.make_tensor_value_info(pre[num], TensorProto.FLOAT,
                                                       inputs[len(inputs) - 1 - counterX].shape)] + X
                    counterX+=1

                num+=1
                if(len(op.src) <= num):break
        for (src_op, x_id, y, y_stores_grad) in op.src:
            dependency[src_op] -= 1
            if src_op.requires_grad is True:
                if dependency[src_op] == 0:
                    if not isinstance(src_op, Dummy):ready.append((src_op))
    model_def = helper.make_model(helper.make_graph(node, "t", X, Y, ), producer_name='o')
    checker.check_model(model_def)
    return model_def




