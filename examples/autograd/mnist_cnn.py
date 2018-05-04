import numpy as np
from singa import convolution_operation as layer_ops
from singa import tensor
from singa import autograd
from singa import optimizer


def load_data(path):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

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
    categorical=categorical.astype(np.float32)
    return categorical

def preprocess(data):
    data=data.astype(np.float32)
    data /= 255
    data=np.expand_dims(data, axis=1)
    return data

def accuracy(pred,target):
    y = np.argmax(pred, axis=1)
    t = np.argmax(target, axis=1)
    a = y == t
    return np.array(a, 'int').sum() / float(len(t))


if __name__ == '__main__':

    batch_number=600
    num_classes = 10
    epochs = 1

    sgd = optimizer.SGD(0.05)
    #opt = optimizer.SGD(momentum=0.9, weight_decay=1e-4)

    train,test=load_data('/Users/wanqixue/Downloads/mnist.npz')
    x_train=preprocess(train[0])
    y_train = to_categorical(train[1], num_classes)

    x_test=preprocess(test[0])
    y_test=to_categorical(test[1],num_classes)
    print 'the shape of training data is',x_train.shape
    print 'the shape of training label is',y_train.shape
    print 'the shape of testing data is', x_test.shape
    print 'the shape of testing label is', y_test.shape


    conv1=layer_ops.Convolution2D('conv1',32,3,1,border_mode='same')
    conv2=layer_ops.Convolution2D('conv2',32,3,1,border_mode='same')

    #operations can create when call
    relu1=layer_ops.Activation('relu1')
    relu2 = layer_ops.Activation('relu2')
    pooling= layer_ops.MaxPooling2D('pooling',3,1,border_mode='same')
    flatten=layer_ops.Flatten('flatten')
    matmul=tensor.Matmul()
    add_bias=tensor.AddBias()
    softmax=tensor.SoftMax()
    cross_entropy=tensor.CrossEntropy()
    #avoid repeat create operations

    w = tensor.Tensor(shape=(25088, 10), requires_grad=True, stores_grad=True) #package a dense layer to calculate the shape automatically
    w.gaussian(0.0, 0.1)

    b = tensor.Tensor(shape=(1, 10), requires_grad=True, stores_grad=True)
    b.set_value(0.0)

    def forward(x,t):
        y=conv1(x)[0]
        y=relu1(y)[0]
        y=conv2(y)[0]
        y=relu2(y)[0]
        y=pooling(y)[0]
        y=flatten(y)[0]
        y=matmul(y,w)[0]
        y=add_bias(y,b)[0]
        y=softmax(y)[0]
        loss=cross_entropy(y,t)[0]
        return loss, y

    for epoch in range(epochs):
        #for i in range(batch_number):
        for i in range(50):
            inputs = tensor.Tensor(data=x_train[i * 100:(1 + i) * 100, :], requires_grad=False, stores_grad=False)
            targets = tensor.Tensor(data=y_train[i * 100:(1 + i) * 100, :], requires_grad=False, stores_grad=False)
            loss, y = forward(inputs,targets)

            accuracy_rate = accuracy(tensor.ctensor2numpy(y.data),tensor.ctensor2numpy(targets.data))
            if (i % 5 == 0):
                print 'accuracy is:', accuracy_rate,'loss is:', tensor.ctensor2numpy(loss.data)[0]

            in_grads = autograd.backward(loss)

            for param in in_grads:
                sgd.apply(0, in_grads[param], param, '')

