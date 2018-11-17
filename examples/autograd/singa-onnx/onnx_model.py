import pickle
from singa import tensor
from singa.tensor import Tensor
from singa import autograd
from singa import optimizer
import numpy as np
import onnx
from onnx import numpy_helper

def load_onnx_model(name = 'singonnx.pkl'):

    with open(name, 'rb') as f:
        model = pickle.load(f)
    return model

def onnx_model_init(inputs,model):
    a = {}
    a['X'] = inputs
    for i in model.graph.node:
        if (i.op_type == 'Constant'):
            a[str(i.output[0])] = tensor.from_numpy(numpy_helper.to_array(i.attribute[0].t))
            a[str(i.output[0])].stores_grad = True
    return a

def onnx_loss(a,model,target):
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
