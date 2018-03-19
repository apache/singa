
from singa import tensor
from singa import engine
from singa import singa_wrap as singa
import numpy as np

def print_singa_tensor(x):
    np_array = x.GetFloatValue(int(x.Size()))
    print(np_array.reshape(x.shape()))
    return

if __name__ =='__main__':

    #prepare numpy training data
    # generate the boundary
    f = lambda x: (5 * x + 1)
    bd_x = np.linspace(-1., 1, 200)
    bd_y = f(bd_x)
    # generate the training data
    x = np.random.uniform(-1, 1, 400)
    y = f(x) + 2 * np.random.randn(len(x))
    # convert training data to 2d space
    label = np.asarray([5 * a + 1 > b for (a, b) in zip(x, y)])
    data = np.array([[a, b] for (a, b) in zip(x, y)], dtype=np.float32)

    def to_categorical(y, num_classes=None):
        """Converts a class vector (integers) to binary class matrix.

        E.g. for use with categorical_crossentropy.

        # Arguments
            y: class vector to be converted into a matrix
                (integers from 0 to num_classes).
            num_classes: total number of classes.

        # Returns
            A binary matrix representation of the input.
        """
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, num_classes))
        categorical[np.arange(n), y] = 1
        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)
        return categorical

    label=to_categorical(label,2).astype(np.float32)
    print 'train_data_shape:',data.shape,'train_label_shape:',label.shape

    # send numpy data to singa_tensor
    tr_data=singa.Tensor((400,2))
    tr_data.CopyFloatDataFromHostPtr(data.flatten())

    tr_label=singa.Tensor((400,2))
    tr_label.CopyFloatDataFromHostPtr(label.flatten())

    w_0=singa.Tensor((2,3))
    singa.Gaussian(float(0), float(0.1), w_0)
    b_0=singa.Tensor((1,3))
    b_0.SetFloatValue(float(0))

    w_1=singa.Tensor((3,2))
    singa.Gaussian(float(0), float(0.1), w_1)
    b_1=singa.Tensor((1,2))
    b_1.SetFloatValue(float(0))


    # initialize Tensor using singa_tensor
    inputs=tensor.Tensor(data=tr_data,requires_grad=False,grad_outlet=False)
    target=tensor.Tensor(data=tr_label,requires_grad=False,grad_outlet=False)

    weight_0=tensor.Tensor(data=w_0,requires_grad=True,grad_outlet=True)
    bias_0=tensor.Tensor(data=b_0,requires_grad=True,grad_outlet=True)

    weight_1=tensor.Tensor(data=w_1,requires_grad=True,grad_outlet=True)
    bias_1=tensor.Tensor(data=b_1,requires_grad=True,grad_outlet=True)

    def update(lr,param,grad): #param:Tensor grad:singa_tensor
        grad *= float(lr)
        assert param.singa_tensor.shape() == grad.shape()
        param.singa_tensor = singa.__sub__(param.singa_tensor,grad)
        return

    lr=0.05
    for i in range(1001):
        outputs=tensor.dot(inputs,weight_0)
        outputs=tensor.add_bias(bias_0,outputs)
        outputs=tensor.relu(outputs)
        outputs = tensor.dot(outputs, weight_1)
        outputs = tensor.add_bias(bias_1, outputs)
        outputs=tensor.softmax(outputs)

        loss=tensor.cross_entropy(outputs,target)

        grads=float(1)
        in_grads = engine.gradients(loss, grads)

        for param in in_grads:
            update(lr,param,in_grads[param])

        if (i % 100 == 0):
            print 'training loss = ' ,float(tensor.To_numpy(loss.singa_tensor))