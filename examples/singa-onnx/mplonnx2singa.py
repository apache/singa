from singa import tensor
from singa.tensor import Tensor
from singa import autograd
from singa import optimizer
import numpy as np
from singa import sonnx
autograd.training = True

# prepare training data in numpy array

# generate the boundary
f = lambda x: (5 * x + 1)
bd_x = np.linspace(-1., 1, 2)
bd_y = f(bd_x)
# generate the training data
#x = np.random.uniform(-1, 1, 4)
x = np.array([0,0.5,-0.5,0.1])
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

model =sonnx.load_onnx_model('singonnx.pkl')
a = sonnx.onnx_model_init(inputs,model)


sgd = optimizer.SGD(0.00)

# training process
for epoch in range(1):
    loss = sonnx.onnx_loss(a,model,target)
    gradient = autograd.backward(loss)
    for p, gp in gradient:
        gp.reshape(p.shape)
        sgd.apply(0, gp, p, '')
    if (epoch % 100 == 0):
        print('training loss = ', tensor.to_numpy(loss)[0])

