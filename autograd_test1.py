import numpy as np
from collections import Counter
import random
import copy
from keras.datasets import mnist
import keras

def affine_forward(x, w, b):
    N = x.shape[0]
    x_rs = np.reshape(x, (N, -1))
    out = x_rs.dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    x, w, b = cache
    N = x.shape[0]
    x_rs = np.reshape(x, (N, -1))
    db = dout.sum(axis=0)
    dw = x_rs.T.dot(dout)
    dx = dout.dot(w.T)
    dx = dx.reshape(x.shape)
    return dx, dw, db

def do_zero_padding(x, pad):
    new_x = np.zeros((x.shape[0], x.shape[1], x.shape[2] + pad, x.shape[3] + pad))
    for i in range(x.shape[0]):
        for j in range(x[i].shape[0]):
            new_x[i, j] = np.pad(x[i, j], int(pad / 2), 'constant')
    return new_x

def conv_forward(x, w, b, conv_param):
    new_x = do_zero_padding(x, conv_param['pad'])

    w_h = w.shape[2]
    w_w = w.shape[3]
    s = conv_param['stride']
    h_num = 1 + int((new_x[0, 0].shape[0] - w_h) / s)
    w_num = 1 + int((new_x[0, 0].shape[1] - w_w) / s)
    X_new = np.ndarray(shape=(new_x.shape[0], w_h * w_w * new_x.shape[1], h_num * w_num))
    for i in range(new_x.shape[0]):
        X = []
        for k in range(h_num):
            for l in range(w_num):
                x_vec = new_x[i, 0:new_x.shape[1], k*s:k*s + w_h, l*s:l*s + w_w].reshape(-1)
                X.append(x_vec)
        X_new[i] = np.asarray(X).T

    W = np.array([])
    for i in range(w.shape[0]):
        w_vec = w[i].reshape(-1)
        if i == 0:
            W = w_vec
        else:
            W = np.vstack((W, w_vec))
    out = np.ndarray(shape=(x.shape[0], w.shape[0], h_num, w_num))
    B = b
    for i in range(X_new.shape[2] - 1):
        B = np.vstack((B, b))
    B = B.T
    for i in range(X_new.shape[0]):
        out[i] = (W.dot(X_new[i]) + B).reshape(w.shape[0], h_num, w_num)

    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward(dout, cache):
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    H_out = 1 + int((H + pad - HH) / stride)
    W_out = 1 + int((W + pad - WW) / stride)
    t_pad = int(pad / 2)


    x_pad = np.pad(x, ((0,), (0,), (t_pad,), (t_pad,)), mode='constant', constant_values=0)
    dx = np.zeros(x.shape)
    dx_pad = np.zeros(x_pad.shape)
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)

    db = np.sum(dout, axis=(0, 2, 3))
    for i in range(H_out):
        for j in range(W_out):
            x_pad_masked = x_pad[:, :, i * stride:i * stride + HH, j * stride:j * stride + WW]
            for k in range(F):  # compute dw
                dw[k, :, :, :] += np.sum(x_pad_masked * (dout[:, k, i, j])[:, None, None, None], axis=0)
            for n in range(N):  # compute dx_pad
                dx_pad[n, :, i * stride:i * stride + HH, j * stride:j * stride + WW] += np.sum((w[:, :, :, :] * (dout[n, :, i, j])[:, None, None, None]), axis=0)
    dx = dx_pad[:, :, t_pad:-t_pad, t_pad:-t_pad]
    return dx, dw, db



class variable(object):
    def __init__(self, data, creator=None, requires_grad=True, grad_outlet=False):
        self.data = data
        if creator is None:
            creator = Leaf(self, requires_grad)
        self.creator = creator #function
        self.grad_outlet=grad_outlet

    def add(self, other):
            return add()(self, other)[0]

    def sub(self, other):
        return sub()(self, other)[0]

    def mul(self, other):
        if isinstance(other, variable):
            return dot()(self, other)[0]

    def dot(self,other):
        return dot()(self,other)

    def __add__(self, other):
        return self.add(other)

    __radd__ = __add__

    def __sub__(self, other):
        return self.sub(other)

    def __mul__(self, other):
        return self.mul(other)

    __rmul__ = __mul__

class operation(object):

    def __init__(self,**operation_params):
        pass

    def __call__(self, *input):
        return self._do_forward(*input)  #input: variables, output:tuple of varibales

    def _do_forward(self, *input):
        unpacked_input = tuple(arg.data for arg in input) #input is Variables
        raw_output = self.forward(*unpacked_input)#data之间的运算
        if not isinstance(raw_output, tuple):
            raw_output = (raw_output,)
            #convert output to tuple type
        self.needs_input_grad = tuple(arg.creator.requires_grad for arg in input) #type:tuple
        self.requires_grad = any(self.needs_input_grad) #bool
        output = tuple(variable(tensor, self) for tensor in raw_output) #packaging: tensor:data self:create/function,many properties like requires_grad are recorded in function
        self.previous_functions = [(arg.creator, id(arg)) for arg in input] #list:[(creator_funciton class, input variables addresses)]
        self.output_ids = {id(var): i for i, var in enumerate(output)}#dict:{variables_addresses: index}
        return output
        #Variables上只记录data与creator(function,leaf)，其余更多forward信息记录在creator上。
    def _do_backward(self, grad_output):
        grad_input = self.backward(grad_output) #numpy array list
        if not isinstance(grad_input, tuple):
            grad_input = (grad_input,)#packaging
        return grad_input #type:tuple

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *grad_output):
        raise NotImplementedError


class Leaf(operation):

    def __init__(self, variable, requires_grad):
        self.variable = variable
        #print(variable.data)
        self.output_ids = {id(variable): 0}
        self.previous_functions = []
        self.requires_grad = requires_grad
        self.init=np.zeros(shape=self.variable.data.shape)
        self.grads=copy.deepcopy(self.init)

    def _do_forward(self, *input):
        raise NotImplementedError

    def _do_backward(self, *grad_output):
        assert len(grad_output) == 1
        #print(type(self.grads),'aaa')
        self.grads += grad_output[0]
        '''if self.grads == None
            self.grads=grad_output[0]
        else:
            self.grads += grad_output[0]'''
        return tuple()

class add(operation):

    def forward(self, a, b):
        self.input=(a,b)
        return a+b

    def backward(self, grad_output):
        if self.input[0].shape==self.input[1].shape:
            return grad_output, grad_output
        else:
            #return grad_output, np.dot(grad_output.T,np.ones(100))
            return grad_output, np.sum(grad_output, axis=0)

class sub(operation):

    def forward(self, a, b):
        if a.shape == b.shape:
            self.broadcast=False
        else:
            self.broadcast=True
        return a-b

    def backward(self, grad_output):
        if self.broadcast == False:
            return grad_output, grad_output.__neg__()
        else:
            return grad_output,None

class dot(operation):

    def forward(self, a, b):
        self.input = (a, b)
        return np.dot(a,b)

    def backward(self, grad_output):
        return np.dot(grad_output,self.input[1].T), np.dot(self.input[0].T,grad_output)

class square(operation):

    def forward(self, a):
        self.input = (a,)
        return np.square(a)

    def backward(self, grad_output):
        return np.dot(grad_output,np.diag(self.input[0]))

class average(operation):
    def forward(self, a):
        self.input = (a,)
        return np.average(a)

    def backward(self, grad_output):
        n=len(self.input[0])
        out=np.ones(n)/n
        return grad_output*out

class relu(operation):
    def forward(self, input):
        self.input=(input,)
        return np.maximum(input, 0.0)

    def backward(self, grad_output):
        a=1. * (self.input[0] > 0)
        b=grad_output*a
        #print(b.shape,'1')
        return grad_output*a

class softmax(operation):
    def forward(self, z):
        #e_x = np.exp(x)
        #v = (e_x.T / e_x.sum(axis=1)).T
        v=np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
        self.output=(v,)
        return v

    def backward(self, grad_output):
        '''medium=np.zeros(shape=(100,10,10))
        for q in range(self.output[0].shape[0]):
            a=np.diag(self.output[0][q])
            medium[q]+=a'''
        out1 = np.einsum('ki,ki->ki',grad_output,self.output[0])
        medium = np.einsum('ki,kj->kij',self.output[0],self.output[0])
        out2=np.einsum('kij,kj->ki', medium , grad_output)
        return out1-out2

class accuracy(operation):
    def forward(self, pred, target):
        y = np.argmax(pred, axis=1)
        t = np.argmax(target, axis=1)
        a = y == t
        return np.array(a, 'int').sum() / len(t)
    def backward(self, *grad_output):
        pass
def accuracy_np(y,t):
    y = np.argmax(y, axis=1)
    t = np.argmax(t, axis=1)
    a = y == t
    return np.array(a, 'int').sum() / len(t)

class cross_entropy(operation):
    def forward(self, pred,target):
        out = - np.multiply(target, np.log(pred)).sum() / pred.shape[0]
        self.input=(pred,target)
        #self.output=(out,)
        return out
    def backward(self, grad_output):
        #print(self.input[1]/self.input[0], '223123')
        #a=-np.ones(self.input[0].shape)*(self.input[1]/self.input[0])/(self.input[0].shape[0])
        a = -(self.input[1] / self.input[0]) / (self.input[0].shape[0])

        #print(grad_output*a)
        return grad_output*a

class output(operation):
    def forward(self, z,T):
        Y=np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
        loss= - np.multiply(T, np.log(Y)).sum() / Y.shape[0]
        accu=accuracy_np(Y,T)
        print('accuracy:',accu)
        self.input=(Y,T)
        return loss
    def backward(self,grad_output):
        return grad_output*(self.input[0] - self.input[1]) / self.input[0].shape[0]

class ExecutionEngine(object):
    def __init__(self):
        pass

    def _compute_dependencies(self, function):
        dependencies = {}
        seen = {function}
        queue = [function]
        while len(queue) > 0:
            fn = queue.pop()
            for prev_fn, arg_id in fn.previous_functions:

                if prev_fn not in dependencies:
                    dependencies[prev_fn] = [Counter() for _ in prev_fn.output_ids]
                    #print(dependencies[prev_fn])
                #output_idx=0
                output_idx = prev_fn.output_ids[arg_id]
                dependencies[prev_fn][output_idx][fn] += 1 #dependencies[prev_fn]:list of [Counter(),... ], output_idx:index,take out an Counter dict
                if prev_fn not in seen:
                    queue.append(prev_fn)
                    seen.add(prev_fn)
        return dependencies,seen  #记录了 某一个函数通过其第几个输出依赖于其后一个函数

    def _free_backward_dependency(self, dependencies, prev_fn, fn, arg_id):
        deps = dependencies[prev_fn]
        output_idx = prev_fn.output_ids[arg_id]
        output_deps = deps[output_idx] #Counter()
        output_deps[fn] -= 1
        if output_deps[fn] == 0:
            del output_deps[fn]
        return output_idx#返回prev_fn的为arg_id输出的下标


    def _is_ready_for_backward(self, dependencies, function):
        for deps in dependencies[function]: #list
            if len(deps) > 0:
                return False
        return True

    def run_backward(self, variable, grad):
        ready = [(variable.creator, (grad,))]
        not_ready = {}

        dependencies,seen = self._compute_dependencies(variable.creator)  #dict

        while len(ready) > 0:
            fn, grad = ready.pop()
            grad_input = fn._do_backward(*grad)
            for (prev_fn, arg_id), d_prev_fn in zip(fn.previous_functions, grad_input):
                if not prev_fn.requires_grad:
                    continue
                output_nr = self._free_backward_dependency(dependencies, prev_fn, fn, arg_id)
                is_ready = self._is_ready_for_backward(dependencies, prev_fn)
                if is_ready:
                    if prev_fn in not_ready:
                        prev_grad = not_ready[prev_fn] #take out a list, whose length is equal to the number of outputs of 'prev_fn'
                        if not prev_grad[output_nr]:
                            prev_grad[output_nr] = d_prev_fn
                        else:
                            prev_grad[output_nr].add_(d_prev_fn)
                        del not_ready[prev_fn]
                    else:
                        assert output_nr == 0
                        prev_grad = (d_prev_fn,)
                    ready.append((prev_fn, prev_grad))    #add tuple(function,[out_grads])
                else:
                    if prev_fn in not_ready:
                        prev_grad = not_ready[prev_fn]
                    else:
                        prev_grad = [None for _ in prev_fn.output_ids]

                    if not prev_grad[output_nr]:
                        prev_grad[output_nr] = d_prev_fn
                    else:
                        prev_grad[output_nr].add_(d_prev_fn)

                    not_ready[prev_fn] = prev_grad

        in_grads={}
        for func in seen:
            if isinstance(func,Leaf):
                if func.variable.grad_outlet==True:
                    #func.grads = copy.deepcopy(func.init)
                    in_grads[func.variable]=func.grads
                    func.grads=copy.deepcopy(func.init)

        return in_grads


def gradients(function,gradient):
    _execution_engine = ExecutionEngine()
    return _execution_engine.run_backward(function, gradient)


class convolution(operation):
    def __init__(self,**operation_params):
        self.params= operation_params

    def forward(self,x,w,b):
        y,cache=conv_forward(x,w,b,self.params)
        self.cache=cache
        return y

    def backward(self, grad_output):
        dx,dw,db=conv_backward(grad_output,self.cache)
        return dx, dw, db
# call: convolution[params][x,w,b]
#convolution(stride=1,pad=0)(x,w0,b0)[0]

class affine(operation):
    def forward(self, x,w,b):
        y,cache=affine_forward(x,w,b)
        self.cache=cache
        return y
    def backward(self, grad_output):
        dx, dw, db = affine_backward(grad_output, self.cache)
        return dx, dw, db
#affine()(x,w1,b1)



#--------------------------------------------------------------------------------------

if __name__ =='__main__':
    num_classes = 10

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255


    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    #print(x_train.shape, y_train.shape)

    w_0=np.random.randn(784,200)*0.1 #(784,20)
    b_0=np.zeros(w_0.shape[1]) #(20,)
    w_1=np.random.randn(200,40)*0.1 #(20,10)
    b_1=np.zeros(w_1.shape[1]) #(10,)
    w_2 = np.random.randn(40,10)*0.1  # (20,10)
    b_2 = np.zeros(w_2.shape[1])  # (10,)

    weight_0=variable(w_0,requires_grad=True,grad_outlet=True)
    print('weight:',weight_0)

    bias_0=variable(b_0,requires_grad=True,grad_outlet=True)
    print('bias:',bias_0)

    weight_1=variable(w_1,requires_grad=True,grad_outlet=True)
    print('weight:',weight_1)

    bias_1=variable(b_1,requires_grad=True,grad_outlet=True)
    print('bias:',bias_1)

    weight_2 = variable(w_2, requires_grad=True, grad_outlet=True)
    print('weight:', weight_2)

    bias_2 = variable(b_2, requires_grad=True, grad_outlet=True)
    print('bias:', bias_2)

    learning_rate=0.01
    batch_size=100
    batch_number=600
    grads=np.array([1])
    #print(y_train)
    def network(input):
        y=dot()(input,weight_0)[0]
        #print(y.data)
        y=add()(y,bias_0)[0]
        #print(y.data)
        y=relu()(y)[0]
        #print(y.data)
        y=dot()(y,weight_1)[0]
        #print(y.data)
        y=add()(y,bias_1)[0]
        y = relu()(y)[0]
        # print(y.data)
        y = dot()(y, weight_2)[0]
        # print(y.data)
        y = add()(y, bias_2)[0]
        y=softmax()(y)[0]
        #y=output()(y,target)[0]
        #print(y.data)
        return y

    w=variable(np.reshape(np.random.randn(9)*0.1,(1,1,3,3)),requires_grad=True,grad_outlet=True)
    b=variable(np.zeros(1),requires_grad=True,grad_outlet=True)
    weight_c_0=variable(np.random.randn(676,200)*0.1,requires_grad=True,grad_outlet=True)
    def conv_test(input):
        y=convolution(stride=1,pad=0)(input,w,b)[0]
        y=affine()(y, weight_c_0, bias_0)[0]
        y=relu()(y)[0]
        y=dot()(y,weight_1)[0]
        y=add()(y,bias_1)[0]
        y = relu()(y)[0]
        y = dot()(y, weight_2)[0]
        y = add()(y, bias_2)[0]
        y=softmax()(y)[0]
        return y

    def shuffle_input(x,y):
        a=np.hstack((x,y))
        np.random.shuffle(a)
        x_out=a[:,0:784]
        y_out=a[:,784:794]
        print(x_out.shape)
        return x_out,y_out

    '''for epoch in range(5):
        idx=[i for i in range(batch_number)]
        random.shuffle(idx)
        #x_train,y_train=shuffle_input(x_train,y_train)

        pred_labels = network(variable(x_train))
        loss = cross_entropy()(pred_labels, variable(y_train))[0]
        accu = accuracy()(pred_labels, variable(y_train))[0]
        print('for train set before training,', 'epoch:', epoch, 'the loss:', loss.data, 'the accuracy:', accu.data)

        for i in idx:
        #for i in range(1):
            inputs=variable(x_train[i*100:(1+i)*100,:] , requires_grad=False , grad_outlet=False)
            targets=variable(y_train[i*100:(1+i)*100,:] , requires_grad=False , grad_outlet=False)
            pred_labels=network(inputs)
            loss=cross_entropy()(pred_labels,targets)[0]
            #loss=network(inputs)
            #print(loss.data)
            accu = accuracy()(pred_labels, targets)[0]
            #print(loss.data)
            in_grads = gradients(loss, grads)
            #print(np.abs(in_grads[weight_1]).sum())
            #print(np.abs(w_0).sum())

            #print(in_grads)
            for param in in_grads:
                #print(param.data.shape)
                #print(param.data.shape,in_grads[param].shape,'11')
                param.data -= learning_rate*in_grads[param]
                #param.data -= learning_rate * (in_grads[param]+2*param.data)
                #print(param,np.abs(in_grads[param]).sum())
            #print(np.abs(weight_2.data).sum())
            #print(np.abs(weight_0.data).sum())
            #print('     ')
        pred_labels = network(variable(x_train))
        loss = cross_entropy()(pred_labels, variable(y_train))[0]
        accu = accuracy()(pred_labels, variable(y_train))[0]
        print('for train set,','epoch:',epoch,'the loss:',loss.data,'the accuracy:',accu.data)
        
        pred_labels = network(variable(x_test))
        loss = cross_entropy()(pred_labels, variable(y_test))[0]
        accu = accuracy()(pred_labels, variable(y_test))[0]
        print('for test set,','epoch:', epoch, 'the loss:', loss.data, 'the accuracy:', accu.data)'''

    for epoch in range(5):
        x_train=np.reshape(x_train,(-1,1,28,28))
        print(x_train.shape)
        x_test=np.reshape(x_test,(-1,1,28,28))
        idx=[i for i in range(batch_number)]
        random.shuffle(idx)

        for i in idx:
        #for i in range(1):
            inputs=variable(x_train[i*100:(1+i)*100,:] , requires_grad=False , grad_outlet=False)
            targets=variable(y_train[i*100:(1+i)*100,:] , requires_grad=False , grad_outlet=False)
            pred_labels=conv_test(inputs)
            loss=cross_entropy()(pred_labels,targets)[0]
            print(loss.data)
            in_grads = gradients(loss, grads)


            for param in in_grads:
                param.data -= learning_rate*in_grads[param]








