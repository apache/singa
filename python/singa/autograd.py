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

from collections import Counter, deque
import numpy as np
import math

from singa import tensor
from .tensor import Tensor
from . import singa_wrap as singa


CTensor = singa.Tensor
training = False


def infer_dependency(op):
    """
    Infer the dependency of all operations with the
    given op as the last operation.
    Operation A is depending on B if A uses the output(s) of B.

    Args:
        op: an Operation instance, e.g. the loss operation.

    Return:
        a Counter instance with the operation as the key,
        and the number of operations that are depending on it as the value;
        and a Counter instance with the id of the output tensor as the key, and 
        the number of operations that are depending on it as the value.
    """

    # current op is not inserted into the dependency_count
    # if the current op is not a terminal op, then this function may just
    # count dependency of a branch.
    op_count = Counter()
    tensor_count = Counter()
    queue = deque([op])
    while len(queue) > 0:
        cur_op = queue.pop()
        for src_op, xid, _, _ in cur_op.src:
            if src_op not in op_count:
                op_count[src_op] = 1
                queue.append(src_op)
            else:
                op_count[src_op] += 1
            tensor_count[xid] += 1
    return op_count, tensor_count


def gradients(y, dy=None):
    """
    Compute the gradients of the output w.r.t the parameters

    Args:
        y: the output tensor, e.g., the loss
        dy: gradient of the target w.r.t y; None indicates the gradient is 1.0;
            it can be used to rescale the loss.

    Return:
        a dictionary storing the gradient tensors of all tensors
            whose stores_grad is true (e.g. parameter tensors)
    """
    grads = {}  # mapping: x->dx if x.stores_grad
    for p, dp in backward(y, dy):
        grads[p] = dp
    return grads


def backward(y, dy=None):
    """
    Run the backward propagation starting at y.
    Args:
        y: a Tensor instance, usually the loss
        dy: a number or a Tensor instance, for the gradient of the
            objective/loss w.r.t y, usually None, i.e., 1.0
    Return:
        yeild the parameter (tensor with stores_grad true) and the
            gradient tensors.
    """
    assert isinstance(y, Tensor), "wrong input type."
    op_dep, tensor_dep = infer_dependency(y.creator)
    assert y.size() == 1, (
        "y must be a Tensor with a single value;" "size of y is % d" % y.size()
    )

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

    if y.stores_grad:
        # gradients[y] = dy
        if isinstance(dy, float):
            g = np.array(dy)
        else:
            g = dy
        tg = Tensor(device=g.device(), data=g)
        yield (y, tg)

    while len(ready) > 0:
        op, dys = ready.pop()
        if not op.requires_grad or isinstance(op, Dummy):
            continue
        # if not isinstance(op, tensor.Dummy):
        dxs = op._do_backward(*dys)
        # TODO src and dx must match

        assert len(op.src) == len(dxs), (
            "the number of src ops (=%d) and dx (=%d) not match"
            % (len(op.src), len(dxs))
        )
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

            if isinstance(src_op, Dummy) and (not src_op.stores_grad):
                continue

            y_idx = src_op.y_id2idx[x_id]
            if src_op not in not_ready:
                # src_op may have mulitple outputs
                not_ready[src_op] = [None for _ in src_op.y_id2idx]
                not_ready[src_op][y_idx] = dx
            else:
                dxs_ = not_ready[src_op]
                if dxs_[y_idx] is None:
                    dxs_[y_idx] = dx
                else:
                    # add the gradient from another children operation that
                    # uses y_idx'th output of src_op as input arg
                    dxs_[y_idx] += dx

            op_dep[src_op] -= 1
            tensor_dep[x_id] -= 1
            if y_stores_grad and tensor_dep[x_id] == 0:
                # store the gradient for final return, e.g. for parameters.
                # it may cause a delay to yield. Only after src_op's all
                # output tensors have recieved the gradients, then output
                g = not_ready[src_op][y_idx]
                tg = Tensor(
                    device=g.device(), data=g, name=src_op.grad_name(y_idx)
                )
                yield (y, tg)

            if op_dep[src_op] == 0:
                if src_op.requires_grad is True:
                    assert not isinstance(
                        src_op, Dummy
                    ), "Dummy op does not do backward()"
                    ready.append((src_op, not_ready[src_op]))
                del not_ready[src_op]
        del op  # delete the operation to free all tensors from this op


class Operation(object):
    """
    An operation includes the forward and backward function of
    tensor calculation.
    Steps to add a specific operation Xxxx:
    1. create a subclass of Operation, name it as Xxxx
    2. override the forward() and backward(); The arguments of forward()
       and backward() should only include CTensor;
    """

    op_count = 0

    def __init__(self, name=None):
        if name is None:
            self.name = "{}#{}".format(
                self.__class__.__name__, Operation.op_count
            )
            Operation.op_count += 1
        else:
            self.name = name

    def __call__(self, *xs):
        return self._do_forward(*xs)

    def output_name(self, idx):
        """
        Args:
            idx: index of the output among all outputs

        Return:
            the name of the output tensor
        """
        return "{}:{}".format(self.name, idx)

    def grad_name(self, idx):
        """
        Args:
            idx: index of the output among all outputs

        Return:
            the name of the gradient of the output tensor
        """
        return "{}_g".format(self.output_name(idx))

    def _do_forward(self, *xs):
        """
        Do not call this function from user code. It is called by __call__().
        Args:
            xs, Tensor instance(s)
        Returns:
            Tensor instance(s)
        """
        # TODO add the pre hook
        assert all(
            [isinstance(x, Tensor) for x in xs]
        ), "xs should include only Tensor instances"

        # need to do backward if any of its input arg needs gradient
        self.requires_grad = any([x.requires_grad for x in xs])

        self.src = []
        for x in xs:
            if x.stores_grad:
                # store the tensor whose gradient needs be returned in
                # backward(), e.g. if x is parameter
                self.src.append((x.creator, id(x), x, x.stores_grad))
            else:
                # for intermediate tensors, they will be released soon;
                # no need to store them --> use None
                self.src.append((x.creator, id(x), None, x.stores_grad))

        # get the CTensor (data) if the input arg is Tensor
        xs = tuple(x.data for x in xs)
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        # create Tensor based on CTensor(data);
        # assume outputs are all Tensor instances
        ys = tuple(
            Tensor(
                device=y.device(),
                data=y,
                requires_grad=self.requires_grad,
                creator=self,
                name=self.output_name(idx),
            )
            for idx, y in enumerate(ys)
        )
        # map from python id to output index
        self.y_id2idx = {id(y): i for i, y in enumerate(ys)}
        # TODO add the post hook
        return ys

    def _do_backward(self, *dys):
        dxs = self.backward(*dys)
        if not isinstance(dxs, tuple):
            dxs = (dxs,)
        return dxs

    def forward(self, *xs):
        """Forward propagation.
        Args:
            xs: input args consisting of only CTensors.
        Returns:
            CTensor instance(s)
        """
        raise NotImplementedError

    def backward(self, *dys):
        """ Backward propagation.
        Args:
            dys: input args consisting of only CTensors.
        Returns:
            CTensor instance(s)
        """
        raise NotImplementedError

    def get_params(self):
        return []


class Dummy(Operation):
    """Dummy operation whice serves as a placehoder for autograd
    Args:
        name(string): set it for debug
    """

    def __init__(self, tensor, name=None):
        super(Dummy, self).__init__(name)
        self.src = []
        self.y_id2idx = {id(tensor): 0}
        self.tensor = tensor
        self.requires_grad = False

    def output_name(self, idx):
        return self.name

    def grad_name(self, idx):
        return "{}_g".format(self.name)


    def __getattr__(self, name):
        return self.tensor.__getattribute__(name)

class Mean(Operation):
    def __init__(self):
        super(Mean, self).__init__()

    def forward(self, *l):
        """
        Args:
            l: a list of CTensor
            element-wise mean operator
        Returns:
            a new CTensor
        """
        if training:
            self.l = len(l)
        assert(len(l)>0);
        x = singa.Tensor(list(l[0].shape()),l[0].device())
        x.SetFloatValue(0.0)
        for i in range(len(l)):
            x+=l[i]
        return singa.MultFloat(x,1/len(l))

    def backward(self, dy):
        """
        Args:
            dy(CTensor): dL / dy
        Returns:
            a list of dx(CTensor)
        """
        return [singa.MultFloat(dy,1/self.l)]*self.l

def mean(*l):
    return Mean()(*l)[0]


class ReLU(Operation):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        """
        Args:
            x(CTensor): input tensor
        Returns:
            a new CTensor whose element y = x if x >= 0; otherwise 0;
        """
        if training:
            self.input = x
        return singa.ReLU(x)

    def backward(self, dy):
        """
        Args:
            dy(CTensor): dL / dy
        Returns:
            dx(CTensor): dL / dx = dy if x >= 0; otherwise 0;
        """
        dx = singa.GTFloat(self.input, 0.0)
        dx *= dy
        return dx


def relu(x):
    return ReLU()(x)[0]


class Less(Operation):
    def __init__(self):
        super(Less, self).__init__()

    def forward(self, x,y):
        """Do forward propgation.
        Store the [x<y] if requires gradient.
        Args:
            x (CTensor): matrix
            y (CTensor): matrix
        Returns:
            a CTensor for the result
        """
        cur = singa.LTFloat(singa.__sub__(x,y),0)
        if training:
            self.cache = cur
        return cur

    def backward(self, dy):
        """
        Args:
            dy (CTensor): data for the dL / dy, L is the loss
        """
        assert False,('no backward function for less')

def less(x,y):
    return Less()(x,y)[0]


class Clip(Operation):
    def __init__(self,min,max):
        super(Clip, self).__init__()
        self.max=max
        self.min=min
    def forward(self, x):
        """
        Args:
            x(CTensor): input tensor
        Returns:
            np.clip(x,min,max)
        """
        mask0 = singa.LTFloat(x, self.min)
        mask1 = singa.GTFloat(x, self.max)
        mask00 = singa.MultFloat(mask0,self.min)
        mask11 = singa.MultFloat(mask1,self.max)
        mask2 = singa.LEFloat(x, self.max)
        mask3 = singa.GEFloat(x, self.min)
        maskm = singa.__mul__(mask2,mask3)
        if training:
            self.mask = maskm
        return singa.__add__(singa.__add__(singa.__mul__(maskm,x),mask00),mask11)

    def backward(self, dy):
        return singa.__mul__(dy, self.mask)

def clip(x,min,max):
    return Clip(min,max)(x)[0]

class Identity(Operation):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self,x):
        return x

    def backward(self, dy):
        """
        Args:
            dy(CTensor): dL / dy
        Returns:
            dx(CTensor): dL / dx = dy;
        """
        return dy


def identity(x):
    return Identity()(x)[0]

class Matmul(Operation):
    """For matrix multiplication"""

    def __init__(self):
        super(Matmul, self).__init__()

    def forward(self, x, w):
        """Do forward propgation.
        Store the x(or w) if w(or x) requires gradient.
        Args:
            x (CTensor): matrix
            w (CTensor): matrix
        Returns:
            a CTensor for the result
        """
        if training:
            self.input = (x, w)
        return singa.Mult(x, w)

    def backward(self, dy):
        """
        Args:
            dy (CTensor): data for the dL / dy, L is the loss
        Returns:
            a tuple for (dx, dw)
        """
        return (
            singa.Mult(dy, singa.DefaultTranspose(self.input[1])),
            singa.Mult(singa.DefaultTranspose(self.input[0]), dy),
        )


def matmul(x, w):
    return Matmul()(x, w)[0]

class Greater(Operation):
    def __init__(self):
        super(Greater, self).__init__()

    def forward(self, x,y):
        """Do forward propgation.
        Store the [x>y] if requires gradient.
        Args:
            x (CTensor): matrix
            y (CTensor): matrix
        Returns:
            a CTensor for the result
        """
        cur = singa.GTFloat(singa.__sub__(x,y),0)
        if training:
            self.cache = cur
        return cur

    def backward(self, dy):
        """
        Args:
            dy (CTensor): data for the dL / dy, L is the loss
        """
        assert False,('no backward function for greater')

def greater(x,y):
    return Greater()(x,y)[0]

class AddBias(Operation):
    """
    Add Bias to each row / column of the Tensor, depending on the axis arg.
    """

    def __init__(self, axis=0):
        """
        To indicate the calculation axis, 0 for row, 1 for column.
        Args:
            axis: 0 or 1, default is 0.
        """
        super(AddBias, self).__init__()
        self.axis = axis

    def forward(self, x, b):
        """
        Args:
            x: matrix.
            b: bias to be added.
        Return:
            the result Tensor
        """
        if self.axis == 0:
            singa.AddRow(b, x)
        elif self.axis == 1:
            singa.AddColumn(b, x)
        return x

    def backward(self, dy):
        """
        Args:
            dy (CTensor): data for the dL / dy, L is the loss.
        Return:
            a tuple for (db, dx), db is data for dL / db, dx is data
            for dL / dx.
        """
        if self.axis == 0:
            return dy, singa.Sum(dy, 0)
        elif self.axis == 1:
            return dy, singa.Sum(dy, 0)


def add_bias(x, b, axis=0):
    return AddBias(axis)(x, b)[0]


class Reshape(Operation):
    def __init__(self,shape):
        super(Reshape, self).__init__()
        self.shape=list(shape)

    def forward(self, x):
        self.cache=x.shape()
        return singa.Reshape(x, self.shape)

    def backward(self, dy):
        return singa.Reshape(dy, self.cache)


def reshape(a,shape):
    return Reshape(shape)(a)[0]

class PRelu(Operation):

    def __init__(self):
        super(PRelu, self).__init__()

    def forward(self, x, slope):
        mask0 = singa.LTFloat(x, 0.0)
        if training:
            self.input = x
            self.slope = slope
            self.mask0 = mask0
        x1 = singa.__mul__(x, mask0)
        x1 *= slope
        x2 = singa.ReLU(x)
        x1 += x2
        return x1

    def backward(self, dy):
        dx1mask = singa.GEFloat(self.input, 0.0)
        dx2 = singa.__mul__(self.mask0, self.slope)
        dx = singa.__add__(dx1mask, dx2)
        return singa.__mul__(dy, dx), singa.__mul__(dy,
                                                    singa.__mul__(
                                                        self.mask0, self.input))


def prelu(x, slope):
    return PRelu()(x, slope)[0]

class Add(Operation):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, a, b):
        #up till now, the dimensions of tensor a and b should less than 3
        self.shape0=list(a.shape())
        self.shape1=list(b.shape())

        # fix for convolution, tensor has 4 dims
        assert( (len(self.shape0) <= 2 and len(self.shape1) <= 2) or (self.shape0 == self.shape1) ),"up till now, the dimensions of tensor a and b should less than 3"
        
        return singa.__add__(a, b)

    def backward(self, dy):
        if(type(dy)==float):
            assert self.shape0==self.shape1,('should have same shape')
            return dy,dy
        db=CTensor(list(dy.shape()), dy.device())
        db.CopyData(dy)
        for i in range(len(self.shape0)-len(self.shape1)):
            db=singa.Sum(db, 0)
        return dy, db


def add(a, b):
    return Add()(a, b)[0]

class Elu(Operation):
    def __init__(self,alpha=1):
        super(Elu, self).__init__()
        self.alpha=alpha

    def forward(self, x):
        """Do forward propgation.
        Store the x if requires gradient.
        Args:
            x (CTensor): matrix
        Returns:
            a CTensor for the result
        """
        #f(x) = alpha * (exp(x) - 1.) for x < 0, f(x) = x for x >= 0
        if training:
            self.input = x
        x1 = singa.LTFloat(x, 0.0)
        x1 *= x
        x1 = singa.MultFloat(singa.SubFloat(singa.Exp(x1),1.0),self.alpha)
        x2 = singa.ReLU(x)
        x1 += x2
        return x1

    def backward(self, dy):
        """
        Args:
            dy (CTensor): data for the dL / dy, L is the loss
        Returns:
            a tuple for dx
        """
        dx1mask = singa.LTFloat(self.input, 0.0)
        dx = singa.MultFloat(singa.Exp(self.input), self.alpha)
        dx *= dx1mask

        dx2mask = singa.GEFloat(self.input, 0.0)

        dx += dx2mask
        dx *= dy
        return dx

def elu(x,alpha=1):
    return Elu(alpha)(x)[0]


class Equal(Operation):
    def __init__(self):
        super(Equal, self).__init__()

    def forward(self, x,y):
        """Do forward propgation.
       Store the x if requires gradient.
       Args:
           x (CTensor): matrix
       Returns:
           a CTensor for the result
       """
        m = singa.__sub__(x,y)
        cur = singa.__mul__(singa.GEFloat(m,0),singa.LEFloat(m,0))
        return cur

    def backward(self, dy):
        """
        Args:
            dy (CTensor): data for the dL / dy, L is the loss
        """
        assert False,('no backward function for equal')

def equal(x,y):
    return Equal()(x,y)[0]


class SeLU(Operation):
    def __init__(self,alpha=1.67326,gamma=1.0507):
        super(SeLU, self).__init__()
        self.alpha=alpha
        self.gamma=gamma

    def forward(self, x):
        """Do forward propgation.
        Store the x if x requires gradient.
        Args:
            x (CTensor): matrix
        Returns:
            a CTensor for the result
        """
        #y = gamma * (alpha * e^x - alpha) for x <= 0, y = gamma * x for x > 0
        if training:
            self.input = x
        x1 = singa.LEFloat(x, 0.0)
        x1 *= x
        x1 = singa.MultFloat(singa.SubFloat(singa.Exp(x1), 1.0), self.alpha * self.gamma)
        x2 = singa.ReLU(x)
        x2 = singa.MultFloat(x2,self.gamma)
        x1 += x2
        return x1

    def backward(self, dy):
        """
        Args:
            dy (CTensor): data for the dL / dy, L is the loss
        Returns:
            dx
        """
        dx1mask = singa.LEFloat(self.input, 0.0)
        dx1 = singa.MultFloat(singa.Exp(self.input), self.gamma*self.alpha)
        dx1 = singa.__mul__(dx1mask, dx1)

        dx2mask = singa.GTFloat(self.input, 0.0)
        dx2 = singa.MultFloat(dx2mask, self.gamma)

        dx = singa.__add__(dx1, dx2)
        dx *= dy
        return dx

def selu(x,alpha=1.67326,gamma=1.0507):
    return SeLU(alpha,gamma)(x)[0]



class SoftMax(Operation):
    """
    Apply SoftMax for each row of the Tensor or each column of the Tensor
    according to the parameter axis.
    """

    def __init__(self, axis=0):
        super(SoftMax, self).__init__()
        self.axis = axis

    def forward(self, x):
        """
        Args:
            x(data): the input 1d or 2d tensor
        Returns:
            the result Tensor
        """
        if self.axis == 1:
            x = singa.DefaultTranspose(x)
        self.output = singa.SoftMax(x)
        if self.axis == 0:
            return self.output
        elif self.axis == 1:
            return singa.DefaultTranspose(self.output)

    def backward(self, dy):
        """
        Args:
            dy (CTensor): data for the dL / dy, L is the loss
        Returns:
            dx (Ctensor): data for the dL / dx, L is the loss,
            x is the input of current Opertion
        """
        # calculations are made on numpy array
        if self.axis == 1:
            dy = singa.DefaultTranspose(dy)
        grad = ctensor2numpy(dy)
        output = ctensor2numpy(self.output)
        out_1 = np.einsum("ki,ki->ki", grad, output)
        medium_out = np.einsum("ki,kj->kij", output, output)
        out_2 = np.einsum("kij,kj->ki", medium_out, grad)
        out = out_1 - out_2
        dx = CTensor(out_1.shape)
        dx.CopyFloatDataFromHostPtr(out.flatten())
        """grad = Tensor(data=dy)
        output = Tensor(data=self.output)
        out_1 = einsum('ki,ki->ki', grad, output)
        medium_out = einsum('ki,kj->kij', output, output)
        out_2 = einsum('kij,kj->ki', medium_out, grad)
        out = out_1 - out_2
        dx = CTensor(out_1.data.shape)
        dx.CopyFloatDataFromHostPtr(out.data.flatten())"""
        if self.axis == 0:
            return dx
        elif self.axis == 1:
            return singa.DefaultTranspose(dx)


def softmax(x, axis=0):
    return SoftMax(axis)(x)[0]

class Sum(Operation):
    def __init__(self):
        super(Sum, self).__init__()

    def forward(self, *l):
        if training:
            self.l = len(l)
        assert(len(l)>0);
        x = singa.Tensor(list(l[0].shape()),l[0].device())
        x.SetFloatValue(0.0)
        for i in range(len(l)):
            x+=l[i]
        return x

    def backward(self, dy):
        return [dy]*self.l


def sum(*l):
    return Sum()(*l)[0]

class CrossEntropy(Operation):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    """
    Calculte negative log likelihood loss for a batch of training data.
    """

    def forward(self, x, t):
        """
        Args:
            x (CTensor): 1d or 2d tensor, the prediction data(output)
                         of current network.
            t (CTensor): 1d or 2d tensor, the target data for training.
        Returns:
            loss (CTensor): scalar.
        """
        loss = CTensor((1,))
        loss_data = -singa.SumAsFloat(singa.__mul__(t, singa.Log(x)))
        loss.SetFloatValue(loss_data / x.shape()[0])
        self.x = x
        self.t = t
        self.input = (x, t)
        return loss

    def backward(self, dy=1.0):
        """
        Args:
            dy (float or CTensor): scalar, accumulate gradient from outside
                                of current network, usually equal to 1.0
        Returns:
            dx (CTensor): data for the dL /dx, L is the loss, x is the output
                          of current network. note that this is true for
                          dy = 1.0
        """
        dx = singa.__div__(self.t, self.x)
        dx *= float(-1 / self.x.shape()[0])
        if isinstance(dy, float):
            # dtype of dy: float
            dx *= dy
            return dx, None
        elif isinstance(dy, CTensor):
            pass  # TODO, broadcast elementwise multiply seems not support


def cross_entropy(y, t):
    return CrossEntropy()(y, t)[0]


class SoftMaxCrossEntropy(Operation):
    def __init__(self, t):
        super(SoftMaxCrossEntropy, self).__init__()
        self.t = t.data

    def forward(self, x):
        self.p = singa.SoftMax(x)
        loss = CTensor((1,), self.p.device())
        ret = singa.CrossEntropyFwd(self.p, self.t)
        loss.SetFloatValue(singa.SumAsFloat(ret) / x.shape()[0])
        return loss

    def backward(self, dy=1.0):
        dx = singa.SoftmaxCrossEntropyBwd(self.p, self.t)
        return singa.DivFloat(dx, float(self.p.shape()[0]))


def softmax_cross_entropy(x, t):
    # x is the logits and t is the ground truth; both are 2D.
    return SoftMaxCrossEntropy(t)(x)[0]


class MeanSquareError(Operation):
    def __init__(self):
        super(MeanSquareError, self).__init__()

    def forward(self, x, t):
        self.err = singa.__sub__(x, t)
        sqr = singa.Square(self.err)
        loss = CTensor((1,), x.device())
        loss.SetFloatValue(singa.SumAsFloat(sqr) / x.shape()[0] / 2)
        return loss

    def backward(self, dy=1.0):
        dx = self.err
        dx *= float(1 / self.err.shape()[0])
        if isinstance(dy, float):
            # dtype of dy: float
            dx *= dy
            return dx, None
        elif isinstance(dy, CTensor):
            pass  # TODO, broadcast elementwise multiply seems not support


def mse_loss(x, t):
    return MeanSquareError()(x, t)[0]


def ctensor2numpy(x):
    """
    To be used in SoftMax Operation.
    Convert a singa_tensor to numpy_tensor.
    """
    np_array = x.GetFloatValue(int(x.Size()))
    return np_array.reshape(x.shape())


class Flatten(Operation):
    def __init__(self, start_axis=1):
        super(Flatten, self).__init__()
        # flatten all axis after (inclusive) start_axis
        self.start_axis = start_axis
        assert start_axis == 1, "must flatten into 2d array not"

    def forward(self, x):
        # TODO Do flatten start from axis != 1
        self.shape = list(x.shape())
        y = singa.Reshape(x, (x.shape()[0], x.Size() // x.shape()[0]))
        return y

    def backward(self, dy):
        dx = singa.Reshape(dy, self.shape)
        return dx


def flatten(x):
    return Flatten()(x)[0]


class Layer(object):
    def __init__(self):
        pass

    def device_check(self, *inputs):
        x_device = inputs[0].device
        x_dev_id = x_device.id()
        for var in inputs:
            if var.device.id() != x_dev_id:
                var.to_device(x_device)

    def find_sublayers(self):
        # return a list whose elements are in form of (attribute_name,
        # sublayer)
        sublayers = []
        for attr in self.__dict__:
            if isinstance(self.__dict__[attr], Layer):
                sublayers.append((attr, self.__dict__[attr]))
        return sublayers

    def get_params(self):
        sublayers = self.find_sublayers()
        params = dict()
        for sublayer_name, sublayer in sublayers:
            params[sublayer_name] = sublayer.get_params()
        return params

    def set_params(self, **parameters):
        # set parameters for Layer
        # input should be either a PyTensor or numpy ndarray.
        # examples: Layer.set_params(W=np.ones((in, out), dtype=np.float32)),
        # Layer.set_params(**{'block1':{'linear1':{'W':np.ones((in, out),
        # dtype=np.float32)}}})
        for (parameter_name, parameter_value) in parameters.items():
            # assert isinstance(self.__dict__[parameter_name], Layer)
            assert (
                parameter_name in self.__dict__
            ), "please input correct parameters."
            if isinstance(self.__dict__[parameter_name], Layer):
                self.__dict__[parameter_name].set_params(
                    **parameters[parameter_name]
                )
            elif isinstance(self.__dict__[parameter_name], Tensor):
                self.set_one_param(parameter_name, parameter_value)
            else:
                raise ValueError("please input correct parameters.")

    def set_one_param(self, parameter_name, parameter_value):
        assert (
            parameter_name in self.allow_params
        ), "please input allowed parameters."
        assert (
            parameter_value.shape == self.__dict__[parameter_name].shape
        ), "Shape dismatched."
        if isinstance(parameter_value, Tensor):
            self.__dict__[parameter_name].reset_like(parameter_value)
        elif isinstance(parameter_value, np.ndarray):
            self.__dict__[parameter_name].copy_from_numpy(parameter_value)
        else:
            raise ValueError("parameters should be Tensor or Numpy array.")


class Linear(Layer):
    def __init__(self, in_features, out_features, bias=True):
        w_shape = (in_features, out_features)
        b_shape = (out_features,)
        self.bias = bias

        self.W = Tensor(shape=w_shape, requires_grad=True, stores_grad=True)
        std = math.sqrt(2.0 / (in_features + out_features))
        self.W.gaussian(0.0, std)

        if self.bias:
            self.b = Tensor(shape=b_shape, requires_grad=True, stores_grad=True)
            self.b.set_value(0.0)

    def __call__(self, x):
        if self.bias:
            self.device_check(x, self.W, self.b)
        else:
            self.device_check(x, self.W)
        y = matmul(x, self.W)
        if self.bias:
            y = add_bias(y, self.b, axis=0)
        return y

    def get_params(self):
        if self.bias:
            return {"W": self.W, "b": self.b}
        else:
            return {"W": self.W}

    def set_params(self, **parameters):
        # TODO(wangwei) remove this funciton as Opeation's set_params() enough
        # set parameters for Linear Layer
        # input should be either a PyTensor or numpy ndarray.
        # examples: Linear.set_params(W=np.ones((in, out), dtype=np.float32)),
        # Linear.set_params(**{'W':np.ones((in, out), dtype=np.float32)})
        self.allow_params = ["W", "b"]
        super(Linear, self).set_params(**parameters)
        for parameter_name in parameters:
            if parameter_name is "b":
                self.bias = True


class Concat(Operation):
    def __init__(self, axis=0):
        super(Concat, self).__init__()
        self.axis = axis

    def forward(self, *xs):
        if training:
            offset = 0
            self.slice_point = []
            for t in xs:
                offset += t.shape()[self.axis]
                self.slice_point.append(offset)
        x = singa.VecTensor(list(xs))
        return singa.ConcatOn(x, self.axis)

    def backward(self, dy):
        assert hasattr(
            self, "slice_point"
        ), "Please set training as True before do BP. "
        assert self.slice_point[-1] == dy.shape()[self.axis], "Shape mismatch."
        dxs = []
        last_offset = 0
        for p in self.slice_point:
            dxs.append(singa.SliceOn(dy, last_offset, p, self.axis))
            last_offset = p
        return tuple(dxs)


def cat(xs, axis=0):
    # xs is a tuple of multiple Tensors
    return Concat(axis)(*xs)[0]


class _Conv2d(Operation):
    def __init__(self, handle):
        super(_Conv2d, self).__init__()
        self.handle = handle

    def forward(self, x, W, b=None):
        assert x.nDim() == 4, "The dimensions of input should be 4D."

        if training:
            if self.handle.bias_term:
                self.inputs = (x, W, b)
            else:
                self.inputs = (x, W)

        if not self.handle.bias_term:
            # create empty bias tensor for Cpp API
            b = CTensor((self.handle.num_filters,), x.device())
            b.SetFloatValue(0.0)

        if isinstance(self.handle, singa.CudnnConvHandle):
            return singa.GpuConvForward(x, W, b, self.handle)
        else:
            return singa.CpuConvForward(x, W, b, self.handle)

    def backward(self, dy):
        assert training is True and hasattr(
            self, "inputs"
        ), "Please set training as True before do BP. "
        
        if isinstance(self.handle, singa.CudnnConvHandle):
            dx = singa.GpuConvBackwardx(
                dy, self.inputs[1], self.inputs[0], self.handle
            )
            dW = singa.GpuConvBackwardW(
                dy, self.inputs[0], self.inputs[1], self.handle
            )
            if self.handle.bias_term:
                db = singa.GpuConvBackwardb(dy, self.inputs[2], self.handle)
                return dx, dW, db
            else:
                return dx, dW
        else:
            dx = singa.CpuConvBackwardx(
                dy, self.inputs[1], self.inputs[0], self.handle
            )
            dW = singa.CpuConvBackwardW(
                dy, self.inputs[0], self.inputs[1], self.handle
            )
            if self.handle.bias_term:
                db = singa.CpuConvBackwardb(dy, self.inputs[2], self.handle)
                return dx, dW, db
            else:
                return dx, dW

def conv2d(handle, x, W, b=None):
    if b is None:
        return _Conv2d(handle)(x, W)[0]
    else:
        return _Conv2d(handle)(x, W, b)[0]


class Conv2d(Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        group=1,
        bias=True,
        **kwargs
    ):

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.group = group

        assert (
            self.group >= 1 and self.in_channels % self.group == 0
        ), "please set reasonable group."

        assert (
            self.out_channels >= self.group
            and self.out_channels % self.group == 0
        ), "out_channels and group dismatched."

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple):
            self.kernel_size = kernel_size
        else:
            raise TypeError("Wrong kernel_size type.")

        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif isinstance(stride, tuple):
            self.stride = stride
        else:
            raise TypeError("Wrong stride type.")

        if isinstance(padding, int):
            self.padding = (padding, padding)
        elif isinstance(padding, tuple):
            self.padding = padding
        else:
            raise TypeError("Wrong padding type.")

        if dilation != 1:
            raise ValueError("Not implemented yet")

        self.bias = bias

        self.inner_params = {
            "cudnn_prefer": "fastest",
            "workspace_MB_limit": 1024,
        }
        # TODO valid value of inner_params check

        for kwarg in kwargs:
            if kwarg not in self.inner_params:
                raise TypeError("Keyword argument not understood:", kwarg)
            else:
                self.inner_params[kwarg] = kwargs[kwarg]

        w_shape = (
            self.out_channels,
            int(self.in_channels / self.group),
            self.kernel_size[0],
            self.kernel_size[1],
        )

        self.W = Tensor(shape=w_shape, requires_grad=True, stores_grad=True)
        # std = math.sqrt(
        # 2.0 / (self.in_channels * self.kernel_size[0] * self.kernel_size[1] +
        # self.out_channels))
        std = math.sqrt(
            2.0
            / (
                w_shape[1] * self.kernel_size[0] * self.kernel_size[1]
                + self.out_channels
            )
        )
        self.W.gaussian(0.0, std)

        if self.bias:
            b_shape = (self.out_channels,)
            self.b = Tensor(shape=b_shape, requires_grad=True, stores_grad=True)
            self.b.set_value(0.0)
        else:
            # to keep consistency when to do forward.
            self.b = None
            # Tensor(data=CTensor([]), requires_grad=False, stores_grad=False)

    def __call__(self, x):

        assert x.shape[1] == self.in_channels, "in_channels mismatched"

        if self.bias:
            self.device_check(x, self.W, self.b)
        else:
            self.device_check(x, self.W)

        if x.device.id() == -1:
            if self.group != 1:
                raise ValueError("Not implemented yet")
            else:
                if (not hasattr(self, "handle")) or (
                    x.shape[0] != self.handle.batchsize
                ):
                    self.handle = singa.ConvHandle(
                        x.data,
                        self.kernel_size,
                        self.stride,
                        self.padding,
                        self.in_channels,
                        self.out_channels,
                        self.bias,
                        self.group,
                    )
        else:
            if (not hasattr(self, "handle")) or (
                x.shape[0] != self.handle.batchsize
            ):
                self.handle = singa.CudnnConvHandle(
                    x.data,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.in_channels,
                    self.out_channels,
                    self.bias,
                    self.group,
                )

        y = conv2d(self.handle, x, self.W, self.b)
        return y

    def get_params(self):
        if self.bias:
            return {"W": self.W, "b": self.b}
        else:
            return {"W": self.W}

    def set_params(self, **parameters):
        # TODO(wangwei) remove it as Operation's set_params() is enough
        # input should be either a PyTensor or numpy ndarray.
        # Conv2d.set_params(W=np.ones((n, c, h, w), dtype=np.float32)),
        # Conv2d.set_params(**{'W':np.ones((n, c, h, w), dtype=np.float32)})
        self.allow_params = ["W", "b"]
        super(Conv2d, self).set_params(**parameters)
        for parameter_name in parameters:
            if parameter_name is "b":
                self.bias = True


class SeparableConv2d(Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=False,
    ):
        self.depthwise_conv = Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            group=in_channels,
            bias=bias,
        )

        self.point_conv = Conv2d(in_channels, out_channels, 1, bias=bias)

    def __call__(self, x):
        y = self.depthwise_conv(x)
        y = self.point_conv(y)
        return y


class BatchNorm2d(Layer):
    def __init__(self, num_features, momentum=0.9):
        self.channels = num_features
        self.momentum = momentum

        param_shape = (self.channels,)

        self.scale = Tensor(
            shape=param_shape, requires_grad=True, stores_grad=True
        )
        self.scale.set_value(1.0)

        self.bias = Tensor(
            shape=param_shape, requires_grad=True, stores_grad=True
        )
        self.bias.set_value(0.0)

        self.running_mean = Tensor(
            shape=param_shape, requires_grad=False, stores_grad=False
        )
        self.running_mean.set_value(0.0)

        self.running_var = Tensor(
            shape=param_shape, requires_grad=False, stores_grad=False
        )
        self.running_var.set_value(1.0)

    def __call__(self, x):
        assert x.shape[1] == self.channels, (
            "number of channels dismatched. %d vs %d"
            % (x.shape[1], self.channels)
        )

        self.device_check(
            x, self.scale, self.bias, self.running_mean, self.running_var
        )

        if x.device.id() == -1:
            if not hasattr(self, "handle"):
                self.handle = singa.BatchNormHandle(self.momentum, x.data)
            elif x.shape[0] != self.handle.batchsize:
                self.handle = singa.BatchNormHandle(self.momentum, x.data)
        else:
            if not hasattr(self, "handle"):
                self.handle = singa.CudnnBatchNormHandle(self.momentum, x.data)
            elif x.shape[0] != self.handle.batchsize:
                self.handle = singa.CudnnBatchNormHandle(self.momentum, x.data)

        y = batchnorm_2d(
            self.handle,
            x,
            self.scale,
            self.bias,
            self.running_mean,
            self.running_var,
        )
        return y

    def get_params(self):
        return {"scale": self.scale, "bias": self.bias}

    def set_params(self, **parameters):
        # set parameters for BatchNorm2d Layer
        # input should be either a PyTensor or numpy ndarray.
        # examples:
        #   Batchnorm2d.set_params(scale=np.ones((1,), dtype=np.float32)),
        #   Batchnorm2d.set_params(**{'bias':np.ones((1), dtype=np.float32)})
        self.allow_params = ["scale", "bias"]
        super(BatchNorm2d, self).set_params(**parameters)


class _BatchNorm2d(Operation):
    def __init__(self, handle, running_mean, running_var, name=None):
        super(_BatchNorm2d, self).__init__(name)
        self.handle = handle
        self.running_mean = running_mean.data
        self.running_var = running_var.data

    def forward(self, x, scale, bias):
        if training:
            if (type(self.handle) == singa.BatchNormHandle):
                y, mean, var = singa.CpuBatchNormForwardTraining(
                    self.handle, x, scale, bias, self.running_mean, self.running_var
                )

                self.cache = (x, scale, mean, var, y, bias)
            else:
                y, mean, var = singa.GpuBatchNormForwardTraining(
                    self.handle, x, scale, bias, self.running_mean, self.running_var
                )

                self.cache = (x, scale, mean, var)

        else:

            if (type(self.handle) == singa.BatchNormHandle):
                y = singa.CpuBatchNormForwardInference(
                    self.handle,
                    x,
                    scale,
                    bias,
                    self.running_mean,
                    self.running_var,
                )
            else:
                y = singa.GpuBatchNormForwardInference(
                    self.handle,
                    x,
                    scale,
                    bias,
                    self.running_mean,
                    self.running_var,
                )
        return y

    def backward(self, dy):
        assert training is True and hasattr(
            self, "cache"
        ), "Please set training as True before do BP. "



        if (type(self.handle) == singa.BatchNormHandle):
            x, scale, mean, var, y, bias = self.cache
            dx, ds, db = singa.CpuBatchNormBackwardx(
                self.handle, y, dy, x, scale, bias, mean, var
            )
        else:
            x, scale, mean, var = self.cache
            dx, ds, db = singa.GpuBatchNormBackward(
                self.handle, dy, x, scale, mean, var
            )
            
        return dx, ds, db


def batchnorm_2d(handle, x, scale, bias, running_mean, running_var):
    return _BatchNorm2d(handle, running_mean, running_var)(x, scale, bias)[0]


class _Pooling2d(Operation):
    def __init__(self, handle):
        super(_Pooling2d, self).__init__()
        self.handle = handle

    def forward(self, x):
        if isinstance(self.handle, singa.CudnnPoolingHandle):
            y = singa.GpuPoolingForward(self.handle, x)
        else:
            y = singa.CpuPoolingForward(self.handle, x)

        if training:
            self.cache = (x, y)

        return y

    def backward(self, dy):
        if isinstance(self.handle, singa.CudnnPoolingHandle):
            dx = singa.GpuPoolingBackward(
                self.handle, dy, self.cache[0], self.cache[1]
            )
        else:
            dx = singa.CpuPoolingBackward(
                self.handle, dy, self.cache[0], self.cache[1]
            )
            
        return dx


def pooling_2d(handle, x):
    return _Pooling2d(handle)(x)[0]


class Pooling2d(Layer):
    def __init__(self, kernel_size, stride=None, padding=0, is_max=True):
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple):
            self.kernel_size = kernel_size
        else:
            raise TypeError("Wrong kernel_size type.")

        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        elif isinstance(stride, tuple):
            self.stride = stride
            assert stride[0] > 0 or (kernel_size[0] == 1 and padding[0] == 0), (
                "stride[0]=0, but kernel_size[0]=%d, padding[0]=%d"
                % (kernel_size[0], padding[0])
            )
        else:
            raise TypeError("Wrong stride type.")

        if isinstance(padding, int):
            self.padding = (padding, padding)
        elif isinstance(padding, tuple):
            self.padding = padding
        else:
            raise TypeError("Wrong padding type.")

        self.is_max = is_max

    def __call__(self, x):

        out_shape_h = (
            int(
                (x.shape[2] + 2 * self.padding[0] - self.kernel_size[0])
                // self.stride[0]
            )
            + 1
        )
        out_shape_w = (
            int(
                (x.shape[3] + 2 * self.padding[1] - self.kernel_size[1])
                // self.stride[1]
            )
            + 1
        )
        if x.device.id() == -1:
            if not hasattr(self, "handle"):
                self.handle = singa.PoolingHandle(
                    x.data,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.is_max,
                )
            elif (
                x.shape[0] != self.handle.batchsize
                or out_shape_h != self.handle.pooled_height
                or out_shape_w != self.handle.pooled_width
            ):
                self.handle = singa.PoolingHandle(
                    x.data,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.is_max,
                )
        else:
            if not hasattr(self, "handle"):
                self.handle = singa.CudnnPoolingHandle(
                    x.data,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.is_max,
                )
            elif (
                x.shape[0] != self.handle.batchsize
                or out_shape_h != self.handle.pooled_height
                or out_shape_w != self.handle.pooled_width
            ):
                self.handle = singa.CudnnPoolingHandle(
                    x.data,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.is_max,
                )

        y = pooling_2d(self.handle, x)
        return y


class MaxPool2d(Pooling2d):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(MaxPool2d, self).__init__(kernel_size, stride, padding, True)


class AvgPool2d(Pooling2d):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(AvgPool2d, self).__init__(kernel_size, stride, padding, False)


class MaxPool1d(Pooling2d):
    def __init__(self, kernel_size, stride=None, padding=0):
        if stride is None:
            stride = kernel_size
        super(MaxPool1d, self).__init__(
            (1, kernel_size), (0, stride), (0, padding), True
        )


class AvgPool1d(Pooling2d):
    def __init__(self, kernel_size, stride=None, padding=0):
        if stride is None:
            stride = kernel_size
        super(AvgPool1d, self).__init__(
            (1, kernel_size), (0, stride), (0, padding), False
        )


class Tanh(Operation):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        out = singa.Tanh(x)
        if training:
            self.cache = (out,)
        return out

    def backward(self, dy):
        dx = singa.__mul__(self.cache[0], self.cache[0])
        dx = singa.MultFloat(dx, -1.0)
        dx = singa.AddFloat(dx, 1.0)
        dx *= dy
        return dx


def tanh(x):
    return Tanh()(x)[0]

class Cos(Operation):
    def __init__(self):
        super(Cos, self).__init__()

    def forward(self, x):
        if training:
            self.input = x
        return singa.Cos(x)

    def backward(self, dy):
        dx = singa.Sin(self.input)
        dx = singa.MultFloat(dx, -1.0)
        dx *= dy
        return dx

def cos(x):
    return Cos()(x)[0]

class Cosh(Operation):
    def __init__(self):
        super(Cosh, self).__init__()

    def forward(self, x):
        if training:
            self.input = x
        return singa.Cosh(x)

    def backward(self, dy):
        dx = singa.Sinh(self.input)
        dx *= dy
        return dx

def cosh(x):
    return Cosh()(x)[0]

class Acos(Operation):
    def __init__(self):
        super(Acos, self).__init__()

    def forward(self, x):
        if training:
            self.input = x
        return singa.Acos(x)

    def backward(self, dy):
        dx = singa.Square(self.input)
        dx = singa.MultFloat(dx, -1.0)         
        dx = singa.AddFloat(dx, 1.0)
        dx = singa.PowFloat(dx, -0.5)
        dx = singa.MultFloat(dx, -1.0)
        dx *= dy
        return dx

def acos(x):
    return Acos()(x)[0]

class Acosh(Operation):
    def __init__(self):
        super(Acosh, self).__init__()

    def forward(self, x):
        if training:
            self.input = x
        return singa.Acosh(x)

    def backward(self, dy):
        dx = singa.SubFloat(self.input, 1.0)
        dx = singa.Sqrt(dx)
        temp = singa.AddFloat(self.input, 1.0)
        temp = singa.Sqrt(temp)
        dx = singa.__mul__(dx, temp)
        dx = singa.PowFloat(dx, -1.0)
        dx *= dy
        return dx

def acosh(x):
    return Acosh()(x)[0]

class Sin(Operation):
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        if training:
            self.input = x
        return singa.Sin(x)

    def backward(self, dy):
        dx = singa.Cos(self.input)
        dx *= dy
        return dx

def sin(x):
    return Sin()(x)[0]

class Sinh(Operation):
    def __init__(self):
        super(Sinh, self).__init__()

    def forward(self, x):
        if training:
            self.input = x
        return singa.Sinh(x)

    def backward(self, dy):
        dx = singa.Cosh(self.input)
        dx *= dy
        return dx

def sinh(x):
    return Sinh()(x)[0]

class Asin(Operation):
    def __init__(self):
        super(Asin, self).__init__()

    def forward(self, x):
        if training:
            self.input = x
        return singa.Asin(x)

    def backward(self, dy):
        dx = singa.Square(self.input)
        dx = singa.MultFloat(dx, -1.0)         
        dx = singa.AddFloat(dx, 1.0)
        dx = singa.PowFloat(dx, -0.5)
        dx *= dy
        return dx

def asin(x):
    return Asin()(x)[0]

class Asinh(Operation):
    def __init__(self):
        super(Asinh, self).__init__()

    def forward(self, x):
        if training:
            self.input = x
        return singa.Asinh(x)

    def backward(self, dy):
        dx = singa.Square(self.input)
        dx = singa.AddFloat(dx, 1.0)
        dx = singa.PowFloat(dx, -0.5)
        dx *= dy
        return dx

def asinh(x):
    return Asinh()(x)[0]

class Tan(Operation):
    def __init__(self):
        super(Tan, self).__init__()

    def forward(self, x):
        if training:
            self.input = x
        return singa.Tan(x)

    def backward(self, dy):
        dx = singa.Cos(self.input)
        dx = singa.Square(dx)
        dx = singa.PowFloat(dx, -1.0)
        dx *= dy
        return dx

def tan(x):
    return Tan()(x)[0]

class Atan(Operation):
    def __init__(self):
        super(Atan, self).__init__()

    def forward(self, x):
        if training:
            self.input = x
        return singa.Atan(x)

    def backward(self, dy):
        dx = singa.Square(self.input)
        dx = singa.AddFloat(dx, 1.0)
        dx = singa.PowFloat(dx, -1.0)
        dx *= dy
        return dx

def atan(x):
    return Atan()(x)[0]

class Atanh(Operation):
    def __init__(self):
        super(Atanh, self).__init__()

    def forward(self, x):
        if training:
            self.input = x
        return singa.Atanh(x)

    def backward(self, dy):
        dx = singa.Square(self.input)
        dx = singa.MultFloat(dx, -1.0)         
        dx = singa.AddFloat(dx, 1.0)
        dx = singa.PowFloat(dx, -1.0)
        dx *= dy
        return dx

def atanh(x):
    return Atanh()(x)[0]

class Sigmoid(Operation):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        out = singa.Sigmoid(x)
        if training:
            self.cache = (out,)
        return out

    def backward(self, dy):
        dx = singa.MultFloat(self.cache[0], -1.0)
        dx = singa.AddFloat(dx, 1.0)
        dx = singa.__mul__(self.cache[0], dx)
        dx *= dy
        return dx


def sigmoid(x):
    return Sigmoid()(x)[0]


class Mul(Operation):
    def __init__(self):
        super(Mul, self).__init__()

    def forward(self, x1, x2):
        if training:
            self.cache = (x1, x2)
        return singa.__mul__(x1, x2)

    def backward(self, dy):
        dx1 = singa.__mul__(dy, self.cache[1])
        dx2 = singa.__mul__(dy, self.cache[0])
        return dx1, dx2


class Unsqueeze(Operation):
    def __init__(self,axis):
        super(Unsqueeze, self).__init__()
        if(type(axis) is int):
            self.axis=list(axis)
        else:
            self.axis=axis

    def forward(self, x):
        self.cache=x.shape()
        cur = list(self.cache)
        for i in self.axis:
            cur.insert(i,1)
        return singa.Reshape(x, cur)

    def backward(self, dy):
        return singa.Reshape(dy, self.cache)


def unsqueeze(x,axis=-1):
    return Unsqueeze(axis)(x)[0]


def mul(x, y):
    # do pointwise multiplication
    return Mul()(x, y)[0]

class Transpose(Operation):
    def __init__(self,perm):
        super(Transpose, self).__init__()
        self.perm=list(perm)

    def forward(self, x):
        return singa.Transpose(x, self.perm)

    def backward(self, dy):
        cur=[]
        for i in range(len(self.perm)):
            cur+=[self.perm.index(i)]
        return singa.Transpose(dy, cur)


def transpose(x,shape):
    return Transpose(shape)(x)[0]


def add_all(*xs):
    assert len(xs) > 2
    y = add(xs[0], xs[1])
    for x in xs[2:]:
        y = add(y, x)
    return


class RNN_Base(Layer):
    def __init__(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError

    def step_forward(self, x=None, h=None, c=None, Wx=None, Wh=None, Bx=None, Bh=None, b=None):
        raise NotImplementedError


class RNN(RNN_Base):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        nonlinearity="tanh",
        bias=True,
        batch_first=False,
        dropout=0,
        bidirectional=False,
    ):
        self.nonlinearity = nonlinearity

        Wx_shape = (input_size, hidden_size)
        self.Wx = Tensor(shape=Wx_shape, requires_grad=True, stores_grad=True)
        self.Wx.gaussian(0.0, 1.0)

        Wh_shape = (hidden_size, hidden_size)
        self.Wh = Tensor(shape=Wh_shape, requires_grad=True, stores_grad=True)
        self.Wh.gaussian(0.0, 1.0)

        B_shape = (hidden_size,)
        self.b = Tensor(shape=B_shape, requires_grad=True, stores_grad=True)
        self.b.set_value(0.0)

        self.params = (self.Wx, self.Wh, self.b)

    def __call__(self, xs, h0):
        # xs: a tuple or list of input tensors
        if not isinstance(xs, tuple):
            xs = tuple(xs)
        inputs = xs + (h0,)
        self.device_check(*inputs)
        # self.device_check(inputs[0], *self.params)
        self.device_check(inputs[0], self.Wx, self.Wh, self.b)
        batchsize = xs[0].shape[0]
        out = []
        h = self.step_forward(xs[0], h0, self.Wx, self.Wh, self.b)
        out.append(h)
        for x in xs[1:]:
            assert x.shape[0] == batchsize
            h = self.step_forward(x, h, self.Wx, self.Wh, self.b)
            out.append(h)
        return out, h

    def step_forward(self, x, h, Wx, Wh, b):
        y2 = matmul(h, Wh)
        y1 = matmul(x, Wx)
        y = add(y2, y1)
        y = add_bias(y, b, axis=0)
        if self.nonlinearity == "tanh":
            y = tanh(y)
        elif self.nonlinearity == "relu":
            y = relu(y)
        else:
            raise ValueError
        return y


class LSTM(RNN_Base):
    def __init__(
        self,
        input_size,
        hidden_size,
        nonlinearity="tanh",
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0,
        bidirectional=False,
    ):
        self.nonlinearity = nonlinearity

        Wx_shape = (input_size, hidden_size)
        self.Wx = []
        for i in range(4):
            w = Tensor(shape=Wx_shape, requires_grad=True, stores_grad=True)
            w.gaussian(0.0, 1.0)
            self.Wx.append(w)

        Wh_shape = (hidden_size, hidden_size)
        self.Wh = []
        for i in range(4):
            w = Tensor(shape=Wh_shape, requires_grad=True, stores_grad=True)
            w.gaussian(0.0, 1.0)
            self.Wh.append(w)

        Bx_shape = (hidden_size,)
        self.Bx = []
        for i in range(4):
            b = Tensor(shape=Bx_shape, requires_grad=True, stores_grad=True)
            b.set_value(0.0)
            self.Bx.append(b)

        self.Bh = []
        for i in range(4):
            b = Tensor(shape=Bx_shape, requires_grad=True, stores_grad=True)
            b.set_value(0.0)
            self.Bh.append(b)

        self.params = self.Wx + self.Wh + self.Bx + self.Bh

    def __call__(self, xs, h0_c0):
        # xs: a tuple or list of input tensors
        # h0_c0: a tuple of (h0, c0)
        h0, c0 = h0_c0
        if not isinstance(xs, list):
            xs = list(xs)
        inputs = xs + list((h0, c0))
        self.device_check(*inputs)
        # self.device_check(inputs[0], *self.params)
        self.device_check(inputs[0], *(self.Wx + self.Wh + self.Bx + self.Bh))
        batchsize = xs[0].shape[0]
        out = []
        h, c = self.step_forward(
            xs[0], h0, c0, self.Wx, self.Wh, self.Bx, self.Bh
        )
        out.append(h)
        for x in xs[1:]:
            assert x.shape[0] == batchsize
            h, c = self.step_forward(
                x, h, c, self.Wx, self.Wh, self.Bx, self.Bh
            )
            out.append(h)
        return out, h, c

    def step_forward(self, x, h, c, Wx, Wh, Bx, Bh):
        y1 = matmul(x, Wx[0])
        y1 = add_bias(y1, Bx[0], axis=0)
        y2 = matmul(h, Wh[0])
        y2 = add_bias(y2, Bh[0], axis=0)
        i = add(y1, y2)
        i = sigmoid(i)

        y1 = matmul(x, Wx[1])
        y1 = add_bias(y1, Bx[1], axis=0)
        y2 = matmul(h, Wh[1])
        y2 = add_bias(y2, Bh[1], axis=0)
        f = add(y1, y2)
        f = sigmoid(f)

        y1 = matmul(x, Wx[2])
        y1 = add_bias(y1, Bx[2], axis=0)
        y2 = matmul(h, Wh[2])
        y2 = add_bias(y2, Bh[2], axis=0)
        o = add(y1, y2)
        o = sigmoid(o)

        y1 = matmul(x, Wx[3])
        y1 = add_bias(y1, Bx[3], axis=0)
        y2 = matmul(h, Wh[3])
        y2 = add_bias(y2, Bh[3], axis=0)
        g = add(y1, y2)
        g = tanh(g)

        cout1 = mul(f, c)
        cout2 = mul(i, g)
        cout = add(cout1, cout2)

        hout = tanh(cout)
        hout = mul(o, hout)
        return hout, cout


class Abs(Operation):
    def forward(self, a):
        if training:
            self.input = a
        return singa.Abs(a)

    def backward(self, dy):
        dx = singa.Sign(self.input)
        dx *= dy
        return dx


def abs(a):
    return Abs()(a)[0]


class Exp(Operation):
    def forward(self, a):
        if training:
            self.input = a
        return singa.Exp(a)

    def backward(self, dy):
        dx = singa.Exp(self.input)
        dx *= dy
        return dx


def exp(a):
    return Exp()(a)[0]


class LeakyRelu(Operation):
    def __init__(self, a):
        super().__init__(self)
        self.a = a

    def forward(self, x):
        if training:
            self.input = x
        x1 = singa.LTFloat(x, 0.0)
        x1 = singa.__mul__(x, x1)
        x1 = singa.MultFloat(x1, self.a)
        x2 = singa.ReLU(x)
        x1 = singa.__add__(x1, x2)
        return x1

    def backward(self, dy):
        # TODO(wangwei) check the correctness
        dx1 = singa.GTFloat(self.input, 0.0)
        dx2 = singa.LTFloat(self.input, 0.0)
        dx2 = singa.MultFloat(dx2, self.a)
        dx = singa.__add__(dx1, dx2)
        dx *= dy
        return dx


def leakyrelu(x, a=0.01):
    return LeakyRelu(a)(x)[0]


class Sign(Operation):
    def __init__(self):
        super(Sign, self).__init__()

    def forward(self, a):
        if training:
            self.input = a
        return singa.Sign(a)

    def backward(self, dy):
        dx = singa.MultFloat(dy, 0.0)
        return dx


def sign(a):
    return Sign()(a)[0]


class Pow(Operation):
    def __init__(self):
        super(Pow, self).__init__()

    def forward(self, a, b):
        if training:
            self.input = (a, b)
        return singa.Pow(a, b)

    def backward(self, dy):
        da1=singa.__mul__(self.input[1], singa.Pow(self.input[0], singa.SubFloat(self.input[1],1.0)))
        da=singa.__mul__(da1, dy)

        db1=singa.__mul__(singa.Pow(self.input[0],self.input[1]), singa.Log(self.input[0]))
        db=singa.__mul__(db1, dy)

        return da, db

def pow(a, b):
    return Pow()(a,b)[0]


class SoftSign(Operation):
    def __init__(self):
        super(SoftSign, self).__init__()  
    
    def forward(self, x):
    # y = x / (1 + np.abs(x))
        if training:
            self.input = x
        x1 = singa.AddFloat(singa.Abs(x),1.0)
        y = singa.__div__(x,x1)
        
        return y
      
    def backward(self, dy):
        dx = singa.AddFloat(singa.Abs(self.input),1.0)
        dx = singa.PowFloat(singa.Square(dx),-1.0)
        dx = singa.__mul__(dy, dx)
        return dx
      
def softsign(x):
    return SoftSign()(x)[0]


class Sqrt(Operation):
    def __init__(self):
        super(Sqrt, self).__init__()  
    
    def forward(self, x):
        if training:
            self.input = x
        return singa.Sqrt(x)
      
    def backward(self, dy):
        dx = singa.PowFloat(self.input,-0.5)
        dx = singa.MultFloat(dx,0.5)
        dx = singa.__mul__(dy, dx)
        return dx

def sqrt(x):
    return Sqrt()(x)[0]
  

class SoftPlus(Operation):
    def __init__(self):
        super(SoftPlus, self).__init__()  
    
    def forward(self, x):
    #f(x) = ln(exp(x) + 1)
        if training:
            self.input = x
        x1 = singa.AddFloat(singa.Exp(x),1.0)
        y = singa.Log(x1)    
        return y

    def backward(self, dy):
        dx = singa.Exp(singa.MultFloat(self.input, -1.0))
        dx = singa.PowFloat(singa.AddFloat(dx,1.0),-1.0)
        dx = singa.__mul__(dy, dx)
        return dx

      
def softplus(x):
    return SoftPlus()(x)[0]


class Sub(Operation):
    def __init__(self):
        super(Sub, self).__init__()    
    
    def forward(self, a, b):    
        if training:
            self.input = (a, b)
            return singa.__sub__(a, b)

    def backward(self, dy):
        return dy, singa.MultFloat(dy, -1.0)


def sub(a, b):
    return Sub()(a,b)[0]


class Min(Operation):
    def __init__(self):
        super(Min, self).__init__()    
    
    def forward(self, a, b):
        m = singa.__sub__(a,b)
        mask0 = singa.LTFloat(m,0) 
        mask00 = singa.__mul__(singa.LEFloat(m,0),a)
        mask1 = singa.GTFloat(m,0)
        mask11=singa.__mul__(mask1,b)
        mask = singa.__add__(mask00,mask11)
        
        if training:
            self.mask0 = mask0
            self.mask1 = mask1
        
        return mask

    def backward(self, dy):
        return (self.mask0,self.mask1)

        
def min(a,b):
    return Min()(a,b)[0]

class Log(Operation):
    def __init__(self):
        super(Log, self).__init__()

    def forward(self, x):
        if training:
            self.input = x
        return singa.Log(x)
    def backward(self, dy):
        dx = singa.PowFloat(self.input,-1)
        dx = singa.__mul__(dy, dx)
        return dx

def log(x):
    return Log()(x)[0]


class HardSigmoid(Operation):
    def __init__(self,alpha=0.2,gamma=0.5):
        super(HardSigmoid, self).__init__()
        self.alpha=alpha
        self.gamma=gamma

    def forward(self, x):
        """Do forward propgation.
        #y = max(0, min(1, alpha * x + gamma))
        Args:
            x (CTensor): matrix
        Returns:
            a CTensor for the result
        """
        x = singa.AddFloat(singa.MultFloat(x,self.alpha),self.gamma)
        if training:
            self.cache = x

        x = singa.ReLU(x)
        mask1 = singa.LTFloat(x, 1.0)
        mask2 = singa.GEFloat(x, 1.0)

        ans = singa.__add__(singa.__mul__(x, mask1),mask2)
        return singa.ReLU(ans)

    def backward(self, dy):
        mask0 = singa.GTFloat(self.cache, 0.0)
        mask1 = singa.LTFloat(self.cache, 1.0)
        mask = singa.__mul__(mask0,mask1)
        return singa.__mul__(singa.MultFloat(mask, self.alpha),dy)

def hardsigmoid(x,alpha=0.2,gamma=0.5):
    return HardSigmoid(alpha,gamma)(x)[0]


class Squeeze(Operation):
    def __init__(self,axis=[]):
        super(Squeeze, self).__init__()
        self.axis=axis

    def forward(self, x):
        self.cache=x.shape()
        newshape = []
        if(self.axis==[]):
            newshape=list(filter(lambda i: i != 1, self.cache))
        else:
            for i in self.axis:
                assert i < len(self.cache)
                assert self.cache[i] == 1, "the length of axis {} is {}, which should be 1".format(i, self.cache[i])
            for ind,v in enumerate(self.cache):
                if ind not in self.axis:
                    newshape.append(v)
        return singa.Reshape(x, newshape)

    def backward(self, dy):
        return singa.Reshape(dy, self.cache)


def squeeze(x,axis=[]):
    return Squeeze(axis)(x)[0]


class Div(Operation):
    def __init__(self):
        super(Div, self).__init__()

    def forward(self, a, b):
        if training:
            self.input = (a, b)
        return singa.__div__(a, b)

    def backward(self, dy):
        #dy/dx_0 = b^(-1)
        #dy/dx_1 = (-a)*b^(-2)
        da = singa.__mul__(dy, singa.PowFloat(self.input[1],-1.0))

        db1 = singa.PowFloat(self.input[1], -2.0)
        db1 = singa.__mul__(db1, singa.MultFloat(self.input[0], -1.0))
        db = singa.__mul__(dy, db1)

        return da,db


def div(a, b):
    return Div()(a,b)[0]



class Shape(Operation):
    def __init__(self):
        super(Shape, self).__init__()

    def forward(self, x):
        cur=list(x.shape())
        cur=tensor.from_numpy(np.array(cur))
        cur.to_device(x.device())
        return cur.data

    def backward(self, dy):
        return list(dy.shape())

        
def shape(x):
    return Shape()(x)[0]


class Max(Operation):
    def __init__(self):
        super(Max, self).__init__()

    def forward(self, a, b):
        m = singa.__sub__(a,b)
        mask0 = singa.GTFloat(m,0)
        mask00 = singa.__mul__(singa.GEFloat(m,0),a)
        mask1 = singa.LTFloat(m,0)
        mask11=singa.__mul__(mask1,b)
        mask = singa.__add__(mask00,mask11)

        if training:
            self.mask0 = mask0
            self.mask1 = mask1

        return mask

    def backward(self, dy):
        return (self.mask0, self.mask1)


def max(a,b):
    return Max()(a,b)[0]


class And(Operation):
    def __init__(self):
        super(And, self).__init__()

    def forward(self, a, b):
        m = singa.__mul__(a, b)
        cur = singa.PowFloat(singa.Sign(m), 2)

        return cur

    def backward(self, dy):
        assert False,('no gradient for backward function')


def _and(a,b):
    return And()(a,b)[0]


class Or(Operation):
    def __init__(self):
        super(Or, self).__init__()

    def forward(self, a, b):
        m = singa.__add__(singa.PowFloat(singa.Sign(a), 2.0), singa.PowFloat(singa.Sign(b), 2.0))
        cur = singa.Sign(m) 

        return cur

    def backward(self, dy):
        assert False,('no gradient for backward function')


def _or(a,b):
    return Or()(a,b)[0]


class Not(Operation):
    def __init__(self):
        super(Not, self).__init__()

    def forward(self, x):
        mask0 = singa.GEFloat(x,0)
        mask1 = singa.LEFloat(x,0)
        cur = singa.__mul__(mask0,mask1)

        return cur

    def backward(self, dy):
        assert False,('no gradient for backward function')


def _not(x):
    return Not()(x)[0]


class Xor(Operation):
    def __init__(self):
        super(Xor, self).__init__()

    def forward(self, a, b):
        m = singa.__sub__(singa.PowFloat(singa.Sign(a), 2.0), singa.PowFloat(singa.Sign(b), 2.0))
        cur = singa.PowFloat(singa.Sign(m), 2.0)    

        return cur

    def backward(self, dy):
        assert False,('no gradient for backward function')


def _xor(a,b):
    return Xor()(a,b)[0]


class Negative(Operation):
    def __init__(self):
        super(Negative, self).__init__()

    def forward(self, x):
        #y=-x
        return singa.MultFloat(x, -1)

    def backward(self, dy):
        return singa.MultFloat(dy, -1)


def negative(x):
    return Negative()(x)[0]


class Reciprocal(Operation):
    def __init__(self):
        super(Reciprocal, self).__init__()

    def forward(self, x):
        #y=1/x elementwise
        if training:
            self.input = x

        return singa.PowFloat(x, -1)

    def backward(self, dy):
        #dy/dx = -1/x**2
        dx = singa.MultFloat(singa.PowFloat(self.input, -2), -1)
        return singa.__mul__(dy, dx)


def reciprocal(x):
    return Reciprocal()(x)[0]
