from singa import tensor
from singa.tensor import Tensor
from singa import autograd
from singa import optimizer
from singa import *

import numpy as np
#import caffe2.python.onnx.backend as backend
import pickle
autograd.training = True

# prepare training data in numpy array

# generate the boundary
f = lambda x: (5 * x + 1)
bd_x = np.linspace(-1., 1, 2)
bd_y = f(bd_x)
# generate the training data
#x = np.random.uniform(-1, 1, 4)
x = np.array([0,0.5,-0.5,0.1])
#print(x)
y = f(x)# + 2 * np.random.randn(len(x))
# convert training data to 2d space
label = np.asarray([5 * a + 1 > b for (a, b) in zip(x, y)])
data = np.array([[a, b] for (a, b) in zip(x, y)], dtype=np.float32)

def to_categorical(y, num_classes):
    '''
    Converts a class vector (integers) to binary class matrix.
    Args
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    Return
        A binary matrix representation of the input.
    '''
    y = np.array(y, dtype='int')
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical

label = to_categorical(label, 2).astype(np.float32)
print('train_data_shape:', data.shape)
print('train_label_shape:', label.shape)

inputs = Tensor(data=data)
target = Tensor(data=label)

w0 = Tensor(shape=(2, 2), requires_grad=True, stores_grad=True)
w0.gaussian(0.0, 0.1)
b0 = Tensor(shape=(1, 2), requires_grad=True, stores_grad=True)
b0.set_value(0.0)

w1 = Tensor(shape=(2, 2), requires_grad=True, stores_grad=True)
w1.gaussian(0.0, 0.1)
b1 = Tensor(shape=(1, 2), requires_grad=True, stores_grad=True)
b1.set_value(0.0)

w2 = Tensor(shape=(2, 2), requires_grad=True, stores_grad=True)
w2.gaussian(0.0, 0.1)
b2 = Tensor(shape=(1, 2), requires_grad=True, stores_grad=True)
b2.set_value(0.0)



sgd = optimizer.SGD(0.00)

# training process
for i in range(1):
    #print('auto grad x', tensor.to_numpy(Tensor(data=inputs.data, device=inputs.data.device)))
    x = autograd.matmul(inputs, w0)
    x = autograd.add_bias(x, b0)
    # x = autograd.relu(x)
    x2 = autograd.matmul(x, w2)
    x2 = autograd.add_bias(x2, b2)
    x1 = autograd.matmul(x, w1)
    x1 = autograd.add_bias(x1, b1)
    x3 = autograd.add(x1, x2)
    #print('auto grad x',tensor.to_numpy(x))
    #print('auto grad x',x)
    #print('auto grad x', tensor.to_numpy(Tensor(data=x.data,device=x.data.device)))
    #print('---auto end---')
    #x = autograd.matmul(x, w1)
    #x = autograd.add_bias(x, b1)
    x3 = autograd.softmax(x3)
    loss = autograd.cross_entropy(x3, target)
    #print(autograd.backward(loss))
    gradient = autograd.backward(loss)
    for p, gp in gradient:
        #print(p.shape)
        #print(gp.shape)
        gp.reshape(p.shape)
        #print()
        #gp = gp.reshape(p.shape)
        #print(gp.shape)
        sgd.apply(0, gp, p, '')
    if (i % 100 == 0):
        print('training loss = ', tensor.to_numpy(loss)[0])


model=autograd.get_onnx_model(loss)

with open('singonnx.pkl', 'wb') as output:
    pickle.dump(model,output)
