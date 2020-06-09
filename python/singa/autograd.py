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

from singa import tensor
from singa import utils
from .tensor import Tensor
from . import singa_wrap as singa

CTensor = singa.Tensor
training = False


def axis_helper(y_shape, x_shape):
    """
    check which axes the x has been broadcasted
    Args:
        y_shape: the shape of result
        x_shape: the shape of x
    Return:
        a tuple refering the axes
    """
    res = []
    j = len(x_shape) - 1
    for i in range(len(y_shape) - 1, -1, -1):
        if j < 0 or x_shape[j] != y_shape[i]:
            res.append(i)
        j -= 1
    return tuple(res[::-1])


def back_broadcast(y_shape, x_shape, x):
    """
    for a brodcasted tensor, restore its shape of x from y_shape to x_shape
    Args:
        y_shape: the shape of result
        x_shape: the shape of x
        x: the input
    Return:
        a tensor
    """
    if y_shape != x_shape:
        x = tensor.from_raw_tensor(x)
        axis = axis_helper(y_shape, x_shape)
        x = tensor.sum(x, axis)
        x = tensor.reshape(x, x_shape)
        x = x.data
    return x


def infer_dependency(op):
    """
    Infer the dependency of all operations with the
    given op as the last operation.
    Operator A is depending on B if A uses the output(s) of B.

    Args:
        op: an Operator instance, e.g. the loss operation.

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
    assert y.size() == 1, ("y must be a Tensor with a single value;"
                           "size of y is % d" % y.size())

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
            "the number of src ops (=%d) and dx (=%d) not match" %
            (len(op.src), len(dxs)))
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
                tg = Tensor(device=g.device(),
                            data=g,
                            name=src_op.grad_name(y_idx))
                yield (y, tg)

            if op_dep[src_op] == 0:
                if src_op.requires_grad is True:
                    assert not isinstance(
                        src_op, Dummy), "Dummy op does not do backward()"
                    ready.append((src_op, not_ready[src_op]))
                del not_ready[src_op]
        del op  # delete the operation to free all tensors from this op


class Operator(object):
    """
    An operation includes the forward and backward function of
    tensor calculation.
    Steps to add a specific operation Xxxx:
    1. create a subclass of Operator, name it as Xxxx
    2. override the forward() and backward(); The arguments of forward()
       and backward() should only include CTensor;
    """

    op_count = 0

    def __init__(self, name=None):
        if name is None:
            self.name = "{}#{}".format(self.__class__.__name__,
                                       Operator.op_count)
            Operator.op_count += 1
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
        assert all([isinstance(x, Tensor) for x in xs
                   ]), "xs should include only Tensor instances"

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
            ) for idx, y in enumerate(ys))
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


class Dummy(Operator):
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


class Mean(Operator):
    """
    Element-wise mean of each of the input CTensors.
    """

    def __init__(self):
        super(Mean, self).__init__()

    def forward(self, *l):
        """
        Args:
            l (a list of CTensor): a list of CTensor for element-wise mean.
        Returns:
            a new CTensor.
        """
        if training:
            self.l = len(l)
        assert (len(l) > 0)
        x = singa.Tensor(list(l[0].shape()), l[0].device())
        x.SetFloatValue(0.0)
        for i in range(len(l)):
            x += l[i]
        return singa.MultFloat(x, 1 / len(l))

    def backward(self, dy):
        """
        Args:
            dy (CTensor): dL / dy.
        Returns:
            a list of dx (CTensor).
        """
        return [singa.MultFloat(dy, 1 / self.l)] * self.l


def mean(*l):
    """
    Element-wise mean of each of the input tensors.
    Args:
        l (a list of Tensor): element-wise mean operator.
    Returns:
        a new Tensor.
    """
    return Mean()(*l)[0]


class ReLU(Operator):
    """
    Relu means rectified linear function, i.e, y = max(0, x) is applied to the
    CTensor elementwise.
    """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        """
        Args:
            x (CTensor): input tensor.
        Returns:
            a new CTensor whose element y = x if x >= 0; otherwise 0.
        """
        if training:
            self.input = x
        return singa.ReLU(x)

    def backward(self, dy):
        """
        Args:
            dy (CTensor): dL / dy.
        Returns:
            dx (CTensor): dL / dx = dy if x >= 0; otherwise 0.
        """
        return singa.ReLUBackward(dy, self.input)


def relu(x):
    """
    Relu means rectified linear function, i.e, y = max(0, x) is applied to the
    CTensors elementwise.
    Args:
        x (Tensor): input tensor.
    Returns:
        a new Tensor whose element y = x if x >= 0; otherwise 0.
    """
    return ReLU()(x)[0]


class Less(Operator):
    """
    Returns the tensor resulted from performing the less logical operation
    elementwise on the input CTensors x and y.
    """

    def __init__(self):
        super(Less, self).__init__()

    def forward(self, x, y):
        """
        Return a<b, where a and b are CTensor.
        """
        cur = singa.LTFloat(singa.__sub__(x, y), 0)
        if training:
            self.cache = cur
        return cur

    def backward(self, dy):
        """
        Args:
            dy (CTensor): data for the dL / dy, L is the loss.
        Raises:
            AssertionError: no backward function for this operator.
        """
        assert False, ('no backward function for less')


def less(x, y):
    """
    Return a<b, where a and b are CTensor.
    """
    return Less()(x, y)[0]


class Clip(Operator):
    """
    Clip operator limits the given input within an interval. The interval
    is specified by the inputs 'min' and 'max'.
    """

    def __init__(self, min, max):
        """
        Args:
            min (float): min value, under which element is replaced by min.
            max (float): max value, above which element is replaced by max.
        """
        super(Clip, self).__init__()
        self.max = max
        self.min = min

    def forward(self, x):
        """
        Args:
            x (CTensor): input tensor
        Returns:
            a new CTensor with np.clip(x,min,max)
        """
        self.mask = singa.Tensor(list(x.shape()), x.device())
        self.mask.SetFloatValue(1.0)

        if self.min is not None:
            self.min = float(self.min)
            mask0 = singa.LTFloat(x, self.min)
            mask1 = singa.GEFloat(x, self.min)
            self.mask = singa.__mul__(mask1, self.mask)
            x = singa.__add__(singa.MultFloat(mask0, self.min),
                              singa.__mul__(mask1, x))

        if self.max is not None:
            self.max = float(self.max)
            mask0 = singa.GTFloat(x, self.max)
            mask1 = singa.LEFloat(x, self.max)
            self.mask = singa.__mul__(mask1, self.mask)
            x = singa.__add__(singa.MultFloat(mask0, self.max),
                              singa.__mul__(mask1, x))

        return x

    def backward(self, dy):
        """
        Args:
            dy (CTensor): dL / dy
        Returns:
            dx (CTensor): dL / dx
        """
        return singa.__mul__(dy, self.mask)


def clip(x, min=None, max=None):
    """
    Clip operator limits the given input within an interval. The interval
    is specified by the inputs 'min' and 'max'.
    Args:
        x (Tensor): input tensor
        min (float): Minimum value, under which element is replaced by min.
        max (float): Maximum value, above which element is replaced by max.
    Returns:
        a new Tensor with np.clip(x,min,max).
    """
    return Clip(min, max)(x)[0]


class Identity(Operator):
    """
    Init a identity operator
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """
        Args:
            x (CTensor): input tensor.
        Returns:
            the same CTensor x.
        """
        return x

    def backward(self, dy):
        """
        Args:
            dy (CTensor): dL / dy.
        Returns:
            dx (CTensor): dL / dx.
        """
        return dy


def identity(x):
    """
    Init a identity operator.
    Args:
        x (Tensor): input tensor.
    Returns:
        the same Tensor with x.
    """
    return Identity()(x)[0]


class Matmul(Operator):
    """
    Init matrix multiplication operator.
    """

    def __init__(self):
        super(Matmul, self).__init__()

    def forward(self, x, w):
        """
        Return `np.matmul(x,w)`, where x and w are CTensor.
        """
        if training:
            self.input = (x, w)
        return singa.Mult(x, w)

    def backward(self, dy):
        """
        Args:
            dy (CTensor): data for the dL / dy, L is the loss.
        Returns:
            a tuple for (dx, dw).
        """
        return (
            singa.Mult(dy, singa.DefaultTranspose(self.input[1])),
            singa.Mult(singa.DefaultTranspose(self.input[0]), dy),
        )


def matmul(x, w):
    """
    Return `np.matmul(x,w)`, where x and w are Tensor.
    """
    return Matmul()(x, w)[0]


class Greater(Operator):
    """
    Returns the tensor resulted from performing the greater logical
    operation elementwise on the input tensors A and B.
    """

    def __init__(self):
        super(Greater, self).__init__()

    def forward(self, x, y):
        """
        Return a>b, where a and b are CTensor.
        """
        cur = singa.GTFloat(singa.__sub__(x, y), 0)
        if training:
            self.cache = cur
        return cur

    def backward(self, dy):
        """
        Args:
            dy (CTensor): data for the dL / dy, L is the loss.
        Raises:
            AssertionError: no backward function for this operator.
        """
        assert False, ('no backward function for greater')


def greater(x, y):
    """
    Return a>b, where a and b are Tensor.
    """
    return Greater()(x, y)[0]


class AddBias(Operator):
    """
    Add Bias to each row / column of the Tensor, depending on the axis arg.
    """

    def __init__(self, axis=0):
        """
        To indicate the calculation axis, 0 for row, 1 for column.
        Args:
            axis (int): 0 or 1, default is 0.
        """
        super(AddBias, self).__init__()
        self.axis = axis

    def forward(self, x, b):
        """
        Args:
            x (CTensor): matrix.
            b (CTensor): bias to be added.
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
    """
    Add Bias to each row / column of the Tensor, depending on the axis arg.
    Args:
        x (Tensor): matrix.
        b (Tensor): bias to be added.
        axis (int): 0 or 1, default is 0.
    Return:
        the result Tensor
    """
    return AddBias(axis)(x, b)[0]


class Reshape(Operator):
    """
    Reshape the input tensor similar to np.reshape.
    """

    def __init__(self, shape):
        """
        Args:
            shape (list of int): Specified shape for output. At most one
                dimension of the new shape can be -1. In this case, the
                value is inferred from the size of the tensor and the
                remaining dimensions. A dimension could also be 0,
                in which case the actual dimension value is unchanged
                (i.e. taken from the input tensor).
        """
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        """
        Args:
            x (CTensor): matrix.
        Return:
            the result CTensor
        """
        self._shape = x.shape()
        shape = list(self.shape)
        # handle the shape with 0
        shape = [
            self._shape[i]
            if i < len(self._shape) and shape[i] == 0 else shape[i]
            for i in range(len(shape))
        ]
        # handle the shape with -1
        hidden_shape = int(np.prod(self._shape) // np.abs(np.prod(shape)))
        self.cache = [int(s) if s != -1 else hidden_shape for s in shape]
        return singa.Reshape(x, self.cache)

    def backward(self, dy):
        """
        Args:
            dy (CTensor): dL / dy
        Returns:
            dx (CTensor): dL / dx
        """
        return singa.Reshape(dy, self._shape)


def reshape(x, shape):
    """
    Reshape the input tensor similar to mp.reshape.
    Args:
        x (Tensor): matrix.
        shape (list of int): Specified shape for output. At most one
            dimension of the new shape can be -1. In this case, the
            value is inferred from the size of the tensor and the
            remaining dimensions. A dimension could also be 0,
            in which case the actual dimension value is unchanged
            (i.e. taken from the input tensor).
    Return:
        the result Tensor
    """
    return Reshape(shape)(x)[0]


class PRelu(Operator):
    """
    PRelu applies the function `f(x) = slope * x` for x < 0,
    `f(x) = x` for x >= 0 to the data tensor elementwise.
    """

    def __init__(self):
        super(PRelu, self).__init__()

    def forward(self, x, slope):
        """
        Args:
            x (CTensor): matrix.
        Return:
            the result CTensor
        """
        mask0 = singa.LTFloat(x, 0.0)
        res = singa.__mul__(x, mask0)
        res = singa.__mul__(res, slope)
        res += singa.ReLU(x)
        if training:
            self.input = x
            self.slope = slope
            self.mask0 = mask0
            self.shape0 = list(x.shape())
            self.shape1 = list(slope.shape())
            self.shape3 = list(res.shape())
        return res

    def backward(self, dy):
        """
        Args:
            dy (CTensor): dL / dy
        Returns:
            dx (CTensor): dL / dx
        """
        dx1mask = singa.GEFloat(self.input, 0.0)
        dx2 = singa.__mul__(self.mask0, self.slope)
        dx = singa.__add__(dx1mask, dx2)
        dx = singa.__mul__(dy, dx)
        dslope = singa.__mul__(dy, singa.__mul__(self.mask0, self.input))
        if (type(dy) == float) or self.shape0 == self.shape1:
            assert self.shape0 == self.shape1, ('should have same shape')
            return dx, dslope
        # handle broadcast
        dx = back_broadcast(self.shape3, self.shape0, dx)
        dslope = back_broadcast(self.shape3, self.shape1, dslope)
        return dx, dslope


def prelu(x, slope):
    """
    PRelu applies the function `f(x) = slope * x` for x < 0,
    `f(x) = x` for x >= 0 to the data tensor elementwise.
    Args:
        x (Tensor): matrix.
    Return:
        the result Tensor
    """
    return PRelu()(x, slope)[0]


class Add(Operator):
    """
    Performs element-wise binary addition.
    """

    def __init__(self):
        super(Add, self).__init__()

    def forward(self, a, b):
        """
        Return `a+b`, where a and b are CTensor.
        """
        res = singa.__add__(a, b)
        if training:
            self.shape0 = list(a.shape())
            self.shape1 = list(b.shape())
            self.shape3 = list(res.shape())
        return res

    def backward(self, dy):
        """
        Args:
            dy(CTensor): dL / dy
        Return:
            a tuple for (dx0, dx1), dx0 is data for dL / da, dx1 is data
            for dL / db.
        """
        dx0, dx1 = dy, dy
        if (type(dy) == float) or self.shape0 == self.shape1:
            assert self.shape0 == self.shape1, ('should have same shape')
            return dx0, dx1
        # handle broadcast
        dx0 = back_broadcast(self.shape3, self.shape0, dx0)
        dx1 = back_broadcast(self.shape3, self.shape1, dx1)
        return dx0, dx1


def add(a, b):
    """
    Return `a+b`, where a and b are Tensor.
    """
    return Add()(a, b)[0]


class Elu(Operator):
    """
    `f(x) = alpha * (exp(x) - 1.)` for x < 0, `f(x) = x` for x >= 0., is applied to
    the tensor elementwise.
    """

    def __init__(self, alpha=1.):
        """
        Args:
            alpha (float): Coefficient of ELU, default is 1.0
        """
        super(Elu, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        """
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
        x1 = singa.MultFloat(singa.SubFloat(singa.Exp(x1), 1.0), self.alpha)
        x2 = singa.ReLU(x)
        x1 += x2
        return x1

    def backward(self, dy):
        """
        Args:
            dy (CTensor): dL / dy
        Returns:
            dx (CTensor): dL / dx
        """
        dx1mask = singa.LTFloat(self.input, 0.0)
        dx = singa.MultFloat(singa.Exp(self.input), self.alpha)
        dx *= dx1mask

        dx2mask = singa.GEFloat(self.input, 0.0)

        dx += dx2mask
        dx *= dy
        return dx


def elu(x, alpha=1):
    """
    `f(x) = alpha * (exp(x) - 1.)` for x < 0, `f(x) = x` for x >= 0., is applied to
    the tensor elementwise.
    Args:
        x (Tensor): matrix
        alpha (float): Coefficient of ELU, default is 1.0
    Returns:
        a Tensor for the result
    """
    return Elu(alpha)(x)[0]


class Equal(Operator):
    """
    Returns the tensor resulted from performing the equal logical operation
    elementwise on the input tensors x and y.
    """

    def __init__(self):
        super(Equal, self).__init__()

    def forward(self, x, y):
        """
        Return `a=b`, where a and b are CTensor.
        """
        m = singa.__sub__(x, y)
        cur = singa.__mul__(singa.GEFloat(m, 0), singa.LEFloat(m, 0))
        return cur

    def backward(self, dy):
        """
        Args:
            dy (CTensor): data for the dL / dy, L is the loss
        Raises:
            AssertionError: no backward function for this operator
        """
        assert False, ('no backward function for equal')


def equal(x, y):
    """
    Return `a=b`, where a and b are Tensor.
    """
    return Equal()(x, y)[0]


class SeLU(Operator):
    """
    `y = gamma * (alpha * e^x - alpha)` for x <= 0, `y = gamma * x` for x > 0
    is applied to the tensor elementwise.
    """

    def __init__(self, alpha=1.67326, gamma=1.0507):
        """
        Args:
            alpha (float): Coefficient of SELU default to 1.67326
            gamma (float): Coefficient of SELU default to 1.0507
        """
        super(SeLU, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x):
        """
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
        x1 = singa.MultFloat(singa.SubFloat(singa.Exp(x1), 1.0),
                             self.alpha * self.gamma)
        x2 = singa.ReLU(x)
        x2 = singa.MultFloat(x2, self.gamma)
        x1 += x2
        return x1

    def backward(self, dy):
        """
        Args:
            dy (CTensor): dL / dy
        Returns:
            dx (CTensor): dL / dx
        """
        dx1mask = singa.LEFloat(self.input, 0.0)
        dx1 = singa.MultFloat(singa.Exp(self.input), self.gamma * self.alpha)
        dx1 = singa.__mul__(dx1mask, dx1)

        dx2mask = singa.GTFloat(self.input, 0.0)
        dx2 = singa.MultFloat(dx2mask, self.gamma)

        dx = singa.__add__(dx1, dx2)
        dx *= dy
        return dx


def selu(x, alpha=1.67326, gamma=1.0507):
    """
    `y = gamma * (alpha * e^x - alpha)` for x <= 0, `y = gamma * x` for x > 0
    is applied to the tensor elementwise.
    Args:
        x (Tensor): matrix
        alpha (float): Coefficient of SELU default to 1.67326
        gamma (float): Coefficient of SELU default to 1.0507
    Returns:
        a Tensor for the result
    """
    return SeLU(alpha, gamma)(x)[0]


class SoftMax(Operator):
    """
    Apply SoftMax for each row of the Tensor or each column of the Tensor
    according to the parameter axis.
    """

    def __init__(self, axis=1):
        """
        Args:
            axis (int): axis of softmax, default to 1
        """
        super(SoftMax, self).__init__()
        self.axis = axis

    def forward(self, x):
        """
        Args:
            x (CTensor): the input 1d or 2d tensor
        Returns:
            the result CTensor
        """
        self.output = singa.SoftMax(x, self.axis)
        return self.output

    def backward(self, dy):
        """
        Args:
            dy (CTensor): dL / dy
        Returns:
            dx (CTensor): dL / dx
        """
        return singa.SoftMaxBackward(dy, self.axis, self.output)


def softmax(x, axis=1):
    """
    Apply SoftMax for each row of the Tensor or each column of the Tensor
    according to the parameter axis.
    Args:
        x (Tensor): the input 1d or 2d tensor
        axis (int): axis of softmax, default to 1
    Returns:
        the result Tensor
    """
    return SoftMax(axis)(x)[0]


class Sum(Operator):
    """
    Element-wise sum of each of the input tensors
    """

    def __init__(self):
        super(Sum, self).__init__()

    def forward(self, *l):
        """
        Args:
            l (a list of CTensor): element-wise sum operator
        Returns:
            a CTensor for the result
        """
        if training:
            self.l = len(l)
        assert (len(l) > 0)
        x = singa.Tensor(list(l[0].shape()), l[0].device())
        x.SetFloatValue(0.0)
        for i in range(len(l)):
            x += l[i]
        return x

    def backward(self, dy):
        """
        Args:
            dy (CTensor): dL / dy
        Returns:
            dx (CTensor): dL / dx
        """
        return [dy] * self.l


def sum(*l):
    """
    Element-wise sum of each of the input tensors
    Args:
        l (a list of Tensor): element-wise sum operator
    Returns:
        a Tensor for the result
    """
    return Sum()(*l)[0]


class CrossEntropy(Operator):

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
        loss = singa.SumAll(singa.__mul__(t, singa.Log(x)))
        loss /= -x.shape()[0]
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
        dx *= float(-1.0 / self.x.shape()[0])
        if isinstance(dy, float):
            # dtype of dy: float
            dx *= dy
            return dx, None
        elif isinstance(dy, CTensor):
            pass  # TODO, broadcast elementwise multiply seems not support


def cross_entropy(y, t):
    return CrossEntropy()(y, t)[0]


class QALSTMLoss(Operator):

    def __init__(self, M=0.2):
        super(QALSTMLoss, self).__init__()
        self.M = M

    def forward(self, pos, neg):
        # L = max{0, M - cosine(q, a+) + cosine(q, a-)}
        zero = singa.Tensor(list(pos.shape()), pos.device())
        zero.SetFloatValue(0.0)
        val = singa.AddFloat(singa.__sub__(neg, pos), self.M)
        gt_zero = singa.__gt__(val, zero)
        self.inputs = (gt_zero, ) # (BS,)
        all_loss = singa.__mul__(gt_zero, val)
        loss = singa.SumAll(all_loss)
        loss /= (pos.shape()[0])
        # assert loss.shape(0) == 1
        return loss

    def backward(self, dy=1.0):
        # dpos = -1 if M-pos+neg > 0 else 0
        # dneg =  1 if M-pos+neg > 0 else 0
        gt_zero = self.inputs[0]
        dpos_factor = singa.Tensor(list(gt_zero.shape()), gt_zero.device())
        dpos_factor.SetFloatValue(-1.0)
        dneg_factor = singa.Tensor(list(gt_zero.shape()), gt_zero.device())
        dneg_factor.SetFloatValue(1.0)
        dpos = singa.__mul__(gt_zero, dpos_factor)
        dneg = singa.__mul__(gt_zero, dneg_factor)
        return dpos, dneg

def qa_lstm_loss(pos, neg, M=0.2):
    return QALSTMLoss(M)(pos, neg)[0]


class SoftMaxCrossEntropy(Operator):

    def __init__(self, t):
        super(SoftMaxCrossEntropy, self).__init__()
        self.t = t.data

    def forward(self, x):
        self.p = singa.SoftMax(x)
        ret = singa.CrossEntropyFwd(self.p, self.t)
        loss = singa.SumAll(ret)
        loss /= x.shape()[0]
        return loss

    def backward(self, dy=1.0):
        dx = singa.SoftmaxCrossEntropyBwd(self.p, self.t)
        dx /= float(self.p.shape()[0])
        return dx


def softmax_cross_entropy(x, t):
    # x is the logits and t is the ground truth; both are 2D.
    return SoftMaxCrossEntropy(t)(x)[0]


class MeanSquareError(Operator):

    def __init__(self):
        super(MeanSquareError, self).__init__()

    def forward(self, x, t):
        self.err = singa.__sub__(x, t)
        sqr = singa.Square(self.err)
        loss = singa.SumAll(sqr)
        loss /= (x.shape()[0] * 2)
        return loss

    def backward(self, dy=1.0):
        dx = self.err
        dx *= float(1 / self.err.shape()[0])
        dx *= dy
        return dx, None


def mse_loss(x, t):
    return MeanSquareError()(x, t)[0]


def ctensor2numpy(x):
    """
    To be used in SoftMax Operator.
    Convert a singa_tensor to numpy_tensor.
    """
    np_array = x.GetFloatValue(int(x.Size()))
    return np_array.reshape(x.shape())


class Flatten(Operator):
    """
    Flattens the input tensor into a 2D matrix. If input tensor has shape
    `(d_0, d_1, ... d_n)` then the output will have shape `(d_0 X d_1 ...
    d_(axis-1), d_axis X d_(axis+1) ... X dn)`.
    """

    def __init__(self, axis=1):
        """
        Args:
            axis (int): Indicate up to which input dimensions (exclusive)
                should be flattened to the outer dimension of the output. The
                value for axis must be in the range [-r, r], where r is the
                rank of the input tensor. Negative value means counting
                dimensions from the back. When axis = 0, the shape of the
                output tensor is `(1, (d_0 X d_1 ... d_n)`, where the shape
                of the input tensor is `(d_0, d_1, ... d_n)`.
        Returns:
            the result CTensor
        """
        super(Flatten, self).__init__()
        self.axis = axis

    def forward(self, x):
        """
        Args:
            x (CTensor): the input tensor
        Returns:
            the result CTensor
        """
        self.shape = list(x.shape())
        shape, axis = self.shape, self.axis
        # the axis must be within this range (0, r-1)
        assert axis <= len(
            shape) - 1 or axis >= 0, "the axis must be within (0, %d-1)" % len(
                shape)
        # calculate the new shape
        new_shape = (1, int(np.prod(shape))) if axis == 0 else (
            int(np.prod(shape[0:axis]).astype(int)),
            int(np.prod(shape[axis:]).astype(int)))
        y = singa.Reshape(x, new_shape)
        return y

    def backward(self, dy):
        """
        Args:
            dy (CTensor): data for the dL / dy, L is the loss
        Returns:
            dx (CTensor): data for the dL / dx, L is the loss,
        """
        dx = singa.Reshape(dy, self.shape)
        return dx


def flatten(x, axis=1):
    """
    Flattens the input tensor into a 2D matrix. If input tensor has shape
    `(d_0, d_1, ... d_n)` then the output will have shape `(d_0 X d_1 ...
    d_(axis-1), d_axis X d_(axis+1) ... X dn)`.
    Args:
        x (Tensor): the input tensor
        axis (int): Indicate up to which input dimensions (exclusive)
            should be flattened to the outer dimension of the output. The
            value for axis must be in the range [-r, r], where r is the
            rank of the input tensor. Negative value means counting
            dimensions from the back. When axis = 0, the shape of the
            output tensor is `(1, (d_0 X d_1 ... d_n)`, where the shape
            of the input tensor is `(d_0, d_1, ... d_n)`.
    Returns:
        the result Tensor
    """
    return Flatten(axis)(x)[0]


class Concat(Operator):
    """
    Concatenate a list of tensors into a single tensor. All input tensors must
    have the same shape, except for the dimension size of the axis to
    concatenate on.
    """

    def __init__(self, axis=0):
        """
        Args:
            axis (int): Which axis to concat on. A negative value means
                counting dimensions from the back. Accepted range is [-r, r-1]
                where r = rank(inputs).
        Returns:
            the result CTensor
        """
        super(Concat, self).__init__()
        self.axis = axis

    def forward(self, *xs):
        """
        Args:
            xs (a list of CTensor): List of tensors for concatenation
        Returns:
            a CTensor for the result
        """
        if self.axis < 0:
            self.axis = self.axis % len(xs[0].shape())
        if training:
            offset = 0
            self.slice_point = []
            for t in xs:
                offset += t.shape()[self.axis]
                self.slice_point.append(offset)
        x = singa.VecTensor(list(xs))
        return singa.ConcatOn(x, self.axis)

    def backward(self, dy):
        """
        Args:
            dy (CTensor): data for the dL / dy, L is the loss
        Returns:
            dxs (a tuple of CTensor): data for the dL / dxs, L is the loss,
        """
        assert hasattr(
            self, "slice_point"), "Please set training as True before do BP. "
        assert self.slice_point[-1] == dy.shape()[self.axis], "Shape mismatch."
        dxs = []
        last_offset = 0
        for p in self.slice_point:
            dxs.append(singa.SliceOn(dy, last_offset, p, self.axis))
            last_offset = p
        return tuple(dxs)


def cat(xs, axis=0):
    """
    Concatenate a list of tensors into a single tensor. All input tensors must
    have the same shape, except for the dimension size of the axis to
    concatenate on.
    Args:
        xs (a list of Tensor): List of tensors for concatenation
        axis (int): Which axis to concat on. A negative value means
            counting dimensions from the back. Accepted range is [-r, r-1]
            where r = rank(inputs).
    Returns:
        a Tensor for the result
    """
    return Concat(axis)(*xs)[0]


class _Conv2d(Operator):
    """
    Init a conv 2d operator
    """

    def __init__(self, handle, odd_padding=(0, 0, 0, 0)):
        """
        Args:
            handle (object): ConvHandle for cpu or CudnnConvHandle for gpu
            odd_padding (tuple of four ints):, the odd paddding is the value
                that cannot be handled by the tuple padding (w, h) mode so
                we need to firstly handle the input, then use the nomal padding
                method.
        """
        super(_Conv2d, self).__init__()
        self.handle = handle
        self.odd_padding = odd_padding

    def forward(self, x, W, b=None):
        """
        Args:
            x (CTensor): input
            W (CTensor): weight
            b (CTensor): bias
        Returns:
            CTensor
        """
        assert x.nDim() == 4, "The dimensions of input should be 4D."
        if self.odd_padding != (0, 0, 0, 0):
            x = utils.handle_odd_pad_fwd(x, self.odd_padding)

        if training:
            if self.handle.bias_term:
                self.inputs = (x, W, b)
            else:
                self.inputs = (x, W)

        if not self.handle.bias_term:
            # create empty bias tensor for Cpp API
            b = CTensor((self.handle.num_filters,), x.device())
            b.SetFloatValue(0.0)

        if (type(self.handle) != singa.ConvHandle):
            return singa.GpuConvForward(x, W, b, self.handle)
        else:
            return singa.CpuConvForward(x, W, b, self.handle)

    def backward(self, dy):
        """
        Args:
            dy (CTensor): dL / dy
        Returns:
            dx (CTensor): dL / dx
        """
        assert training is True and hasattr(
            self, "inputs"), "Please set training as True before do BP. "

        if (type(self.handle) != singa.ConvHandle):
            dx = singa.GpuConvBackwardx(dy, self.inputs[1], self.inputs[0],
                                        self.handle)
            dW = singa.GpuConvBackwardW(dy, self.inputs[0], self.inputs[1],
                                        self.handle)
            db = singa.GpuConvBackwardb(
                dy, self.inputs[2],
                self.handle) if self.handle.bias_term else None
        else:
            dx = singa.CpuConvBackwardx(dy, self.inputs[1], self.inputs[0],
                                        self.handle)
            dW = singa.CpuConvBackwardW(dy, self.inputs[0], self.inputs[1],
                                        self.handle)
            db = singa.CpuConvBackwardb(
                dy, self.inputs[2],
                self.handle) if self.handle.bias_term else None
        if self.odd_padding != (0, 0, 0, 0):
            dx = utils.handle_odd_pad_bwd(dx, self.odd_padding)

        if db:
            return dx, dW, db

        else:
            return dx, dW


def conv2d(handle, x, W, b=None, odd_padding=(0, 0, 0, 0)):
    """
    Conv 2d operator
    Args:
        handle (object): ConvHandle for cpu or CudnnConvHandle for gpu
        x (Tensor): input
        W (Tensor): weight
        b (Tensor): bias
        odd_padding (tuple of four ints):, the odd paddding is the value
            that cannot be handled by the tuple padding (w, h) mode so
            we need to firstly handle the input, then use the nomal padding
            method.
    """
    if b is None:
        return _Conv2d(handle, odd_padding)(x, W)[0]
    else:
        return _Conv2d(handle, odd_padding)(x, W, b)[0]


class _BatchNorm2d(Operator):
    """
    Carries out batch normalization as described in the paper
    https://arxiv.org/abs/1502.03167.
    """

    def __init__(self, handle, running_mean, running_var, name=None):
        """
        Args:
            handle (object): BatchNormHandle for cpu and CudnnBatchNormHandle
                for gpu
            running_mean (float): the running_mean
            running_var (float): the running_var
            name (string): the name assigned to this operator
        """
        super(_BatchNorm2d, self).__init__(name)
        self.handle = handle
        self.running_mean = running_mean.data
        self.running_var = running_var.data

    def forward(self, x, scale, bias):
        """
        Args:
            x (CTensor): the input tensor
            scale (CTensor): the bias tensor
            bias (CTensor): the bias tensor
        Returns:
            the result CTensor
        """
        if training:
            if (type(self.handle) == singa.BatchNormHandle):
                y, mean, var = singa.CpuBatchNormForwardTraining(
                    self.handle, x, scale, bias, self.running_mean,
                    self.running_var)

                self.cache = (x, scale, mean, var, y, bias)
            else:
                y, mean, var = singa.GpuBatchNormForwardTraining(
                    self.handle, x, scale, bias, self.running_mean,
                    self.running_var)

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
        """
        Args:
            dy (CTensor): data for the dL / dy, L is the loss
        Returns:
            dx (CTensor): data for the dL / dx, L is the loss
            ds (CTensor): data for the dL / ds, L is the loss
            db (CTensor): data for the dL / db, L is the loss
        """
        assert training is True and hasattr(
            self, "cache"), "Please set training as True before do BP. "

        if (type(self.handle) == singa.BatchNormHandle):
            x, scale, mean, var, y, bias = self.cache
            dx, ds, db = singa.CpuBatchNormBackwardx(self.handle, y, dy, x,
                                                     scale, bias, mean, var)
        else:
            x, scale, mean, var = self.cache
            dx, ds, db = singa.GpuBatchNormBackward(self.handle, dy, x, scale,
                                                    mean, var)

        return dx, ds, db


def batchnorm_2d(handle, x, scale, bias, running_mean, running_var):
    """
    Carries out batch normalization as described in the paper
    https://arxiv.org/abs/1502.03167.
    Args:
        handle (object): BatchNormHandle for cpu and CudnnBatchNormHandle
            for gpu
        x (Tensor): the input tensor
        scale (Tensor): the bias tensor
        bias (Tensor): the bias tensor
        running_mean (float): the running_mean
        running_var (float): the running_var
    Returns:
        the result Tensor
    """
    return _BatchNorm2d(handle, running_mean, running_var)(x, scale, bias)[0]


class _Pooling2d(Operator):
    """
    Init a pool 2d operator
    """

    def __init__(self, handle, odd_padding=(0, 0, 0, 0)):
        """
        Args:
            handle (object): PoolingHandle for cpu or CudnnPoolingHandle for
                gpu
            odd_padding (tuple of four int): the odd paddding is the value
                that cannot be handled by the tuple padding (w, h) mode so
                it needs to firstly handle the input, then use the normal
                padding method.
        """
        super(_Pooling2d, self).__init__()
        self.handle = handle
        self.odd_padding = odd_padding

    def forward(self, x):
        """
        Args:
            x (CTensor): the input tensor
        Returns:
            the result CTensor
        """
        assert x.nDim() == 4, "The dimensions of input should be 4D."
        if self.odd_padding != (0, 0, 0, 0):
            x = utils.handle_odd_pad_fwd(x, self.odd_padding)

        if (type(self.handle) != singa.PoolingHandle):
            y = singa.GpuPoolingForward(self.handle, x)
        else:
            y = singa.CpuPoolingForward(self.handle, x)
        if training:
            self.cache = (x, y)
        return y

    def backward(self, dy):
        """
        Args:
            dy (CTensor): data for the dL / dy, L is the loss
        Returns:
            dx (CTensor): data for the dL / dx, L is the loss,
        """
        if (type(self.handle) != singa.PoolingHandle):
            dx = singa.GpuPoolingBackward(self.handle, dy, self.cache[0],
                                          self.cache[1])
        else:
            dx = singa.CpuPoolingBackward(self.handle, dy, self.cache[0],
                                          self.cache[1])
        if self.odd_padding != (0, 0, 0, 0):
            dx = utils.handle_odd_pad_bwd(dx, self.odd_padding)

        return dx


def pooling_2d(handle, x, odd_padding=(0, 0, 0, 0)):
    """
    Pooling 2d operator
    Args:
        handle (object): PoolingHandle for cpu or CudnnPoolingHandle for
            gpu
        x (Tensor): input
        odd_padding (tuple of four int): the odd paddding is the value
            that cannot be handled by the tuple padding (w, h) mode so
            it needs to firstly handle the input, then use the normal
            padding method.
    Returns:
        the result Tensor
    """
    return _Pooling2d(handle, odd_padding)(x)[0]


class Tanh(Operator):
    """
    Calculates the hyperbolic tangent of the given input tensor element-wise.
    """

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        """
        Args:
            x (CTensor): Input tensor
        Returns:
            CTensor, the output
        """
        out = singa.Tanh(x)
        if training:
            self.cache = (out,)
        return out

    def backward(self, dy):
        """
        Args:
            dy (CTensor): the gradient tensor from upper operations
        Returns:
            CTensor, the gradient over input
        """
        dx = singa.__mul__(self.cache[0], self.cache[0])
        dx = singa.MultFloat(dx, -1.0)
        dx = singa.AddFloat(dx, 1.0)
        dx *= dy
        return dx


def tanh(x):
    """
    Calculates the hyperbolic tangent of the given input tensor element-wise.
    Args:
        x (Tensor): Input tensor
    Returns:
        Tensor, the output
    """
    return Tanh()(x)[0]


class Cos(Operator):
    """
    Calculates the cosine of the given input tensor, element-wise.
    """

    def __init__(self):
        super(Cos, self).__init__()

    def forward(self, x):
        """
        Args:
            x (CTensor): Input tensor
        Returns:
            CTensor, the output
        """
        if training:
            self.input = x
        return singa.Cos(x)

    def backward(self, dy):
        """
        Args:
            dy (CTensor): the gradient tensor from upper operations
        Returns:
            CTensor, the gradient over input
        """
        dx = singa.Sin(self.input)
        dx = singa.MultFloat(dx, -1.0)
        dx *= dy
        return dx


def cos(x):
    """
    Calculates the cosine of the given input tensor, element-wise.
    Args:
        x (Tensor): Input tensor
    Returns:
        Tensor, the output
    """

    return Cos()(x)[0]


class Cosh(Operator):
    """
    Calculates the hyperbolic cosine of the given input tensor element-wise.
    """

    def __init__(self):
        super(Cosh, self).__init__()

    def forward(self, x):
        """
        Args:
            x (CTensor): Input tensor
        Returns:
            CTensor, the output
        """
        if training:
            self.input = x
        return singa.Cosh(x)

    def backward(self, dy):
        """
        Args:
            dy (CTensor): the gradient tensor from upper operations
        Returns:
            CTensor, the gradient over input
        """
        dx = singa.Sinh(self.input)
        dx *= dy
        return dx


def cosh(x):
    """
    Calculates the hyperbolic cosine of the given input tensor element-wise.
    Args:
        x (Tensor): Input tensor
    Returns:
        Tensor, the output
    """
    return Cosh()(x)[0]


class Acos(Operator):
    """
    Calculates the arccosine (inverse of cosine) of the given input tensor,
    element-wise.
    """

    def __init__(self):
        super(Acos, self).__init__()

    def forward(self, x):
        """
        Args:
            x (CTensor): Input tensor
        Returns:
            CTensor, the output
        """
        if training:
            self.input = x
        return singa.Acos(x)

    def backward(self, dy):
        """
        Args:
            dy (CTensor): the gradient tensor from upper operations
        Returns:
            CTensor, the gradient over input
        """
        dx = singa.Square(self.input)
        dx = singa.MultFloat(dx, -1.0)
        dx = singa.AddFloat(dx, 1.0)
        dx = singa.PowFloat(dx, -0.5)
        dx = singa.MultFloat(dx, -1.0)
        dx *= dy
        return dx


def acos(x):
    """
    Calculates the arccosine (inverse of cosine) of the given input tensor,
    element-wise.
    Args:
        x (Tensor): Input tensor
    Returns:
        Tensor, the output
    """
    return Acos()(x)[0]


class Acosh(Operator):
    """
    Calculates the hyperbolic arccosine of the given input tensor element-wise.
    """

    def __init__(self):
        super(Acosh, self).__init__()

    def forward(self, x):
        """
        Args:
            x (CTensor): Input tensor
        Returns:
            CTensor, the output
        """
        if training:
            self.input = x
        return singa.Acosh(x)

    def backward(self, dy):
        """
        Args:
            dy (CTensor): the gradient tensor from upper operations
        Returns:
            CTensor, the gradient over input
        """
        dx = singa.SubFloat(self.input, 1.0)
        dx = singa.Sqrt(dx)
        temp = singa.AddFloat(self.input, 1.0)
        temp = singa.Sqrt(temp)
        dx = singa.__mul__(dx, temp)
        dx = singa.PowFloat(dx, -1.0)
        dx *= dy
        return dx


def acosh(x):
    """
    Calculates the hyperbolic arccosine of the given input tensor element-wise.
    Args:
        x (Tensor): Input tensor
    Returns:
        Tensor, the output
    """
    return Acosh()(x)[0]


class Sin(Operator):
    """
    Calculates the sine of the given input tensor, element-wise.
    """

    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        """
        Args:
            x (CTensor): Input tensor
        Returns:
            CTensor, the output
        """
        if training:
            self.input = x
        return singa.Sin(x)

    def backward(self, dy):
        """
        Args:
            dy (CTensor): the gradient tensor from upper operations
        Returns:
            CTensor, the gradient over input
        """
        dx = singa.Cos(self.input)
        dx *= dy
        return dx


def sin(x):
    """
    Calculates the sine of the given input tensor, element-wise.
    Args:
        x (Tensor): Input tensor
    Returns:
        Tensor, the output
    """
    return Sin()(x)[0]


class Sinh(Operator):
    """
    Calculates the hyperbolic sine of the given input tensor element-wise.
    """

    def __init__(self):
        super(Sinh, self).__init__()

    def forward(self, x):
        """
        Args:
            x (CTensor): Input tensor
        Returns:
            CTensor, the output
        """
        if training:
            self.input = x
        return singa.Sinh(x)

    def backward(self, dy):
        """
        Args:
            dy (CTensor): the gradient tensor from upper operations
        Returns:
            CTensor, the gradient over input
        """
        dx = singa.Cosh(self.input)
        dx *= dy
        return dx


def sinh(x):
    """
    Calculates the hyperbolic sine of the given input tensor element-wise.
    Args:
        x (Tensor): Input tensor
    Returns:
        Tensor, the output
    """
    return Sinh()(x)[0]


class Asin(Operator):
    """
    Calculates the arcsine (inverse of sine) of the given input tensor, element-wise.
    """

    def __init__(self):
        super(Asin, self).__init__()

    def forward(self, x):
        """
        Args:
            x (CTensor): Input tensor
        Returns:
            CTensor, the output
        """
        if training:
            self.input = x
        return singa.Asin(x)

    def backward(self, dy):
        """
        Args:
            dy (CTensor): the gradient tensor from upper operations
        Returns:
            CTensor, the gradient over input
        """
        dx = singa.Square(self.input)
        dx = singa.MultFloat(dx, -1.0)
        dx = singa.AddFloat(dx, 1.0)
        dx = singa.PowFloat(dx, -0.5)
        dx *= dy
        return dx


def asin(x):
    """
    Calculates the arcsine (inverse of sine) of the given input tensor, element-wise.
    Args:
        x (Tensor): Input tensor
    Returns:
        Tensor, the output
    """

    return Asin()(x)[0]


class Asinh(Operator):
    """
    Calculates the hyperbolic arcsine of the given input tensor element-wise.
    """

    def __init__(self):
        super(Asinh, self).__init__()

    def forward(self, x):
        """
        Args:
            x (CTensor): Input tensor
        Returns:
            CTensor, the output
        """
        if training:
            self.input = x
        return singa.Asinh(x)

    def backward(self, dy):
        """
        Args:
            dy (CTensor): the gradient tensor from upper operations
        Returns:
            CTensor, the gradient over input
        """
        dx = singa.Square(self.input)
        dx = singa.AddFloat(dx, 1.0)
        dx = singa.PowFloat(dx, -0.5)
        dx *= dy
        return dx


def asinh(x):
    """
    Calculates the hyperbolic arcsine of the given input tensor element-wise.
    Args:
        x (Tensor): Input tensor
    Returns:
        Tensor, the output
    """
    return Asinh()(x)[0]


class Tan(Operator):
    """
    Insert single-dimensional entries to the shape of an input tensor (data).
    """

    def __init__(self):
        super(Tan, self).__init__()

    def forward(self, x):
        """
        Args:
            x (CTensor): Input tensor
        Returns:
            CTensor, the output
        """
        if training:
            self.input = x
        return singa.Tan(x)

    def backward(self, dy):
        """
        Args:
            dy (CTensor): the gradient tensor from upper operations
        Returns:
            CTensor, the gradient over input
        """
        dx = singa.Cos(self.input)
        dx = singa.Square(dx)
        dx = singa.PowFloat(dx, -1.0)
        dx *= dy
        return dx


def tan(x):
    """
    Calculates the tangent of the given input tensor, element-wise.
    Args:
        x (Tensor): Input tensor
    Returns:
        Tensor, the output
    """
    return Tan()(x)[0]


class Atan(Operator):
    """
    Calculates the arctangent (inverse of tangent) of the given input tensor, element-wise.
    """

    def __init__(self):
        super(Atan, self).__init__()

    def forward(self, x):
        """
        Args:
            x (CTensor): Input tensor
        Returns:
            CTensor, the output
        """
        if training:
            self.input = x
        return singa.Atan(x)

    def backward(self, dy):
        """
        Args:
            dy (CTensor): the gradient tensor from upper operations
        Returns:
            CTensor, the gradient over input
        """
        dx = singa.Square(self.input)
        dx = singa.AddFloat(dx, 1.0)
        dx = singa.PowFloat(dx, -1.0)
        dx *= dy
        return dx


def atan(x):
    """
    Calculates the arctangent (inverse of tangent) of the given input tensor, element-wise.
    Args:
        x (Tensor): Input tensor
    Returns:
        Tensor, the output
    """
    return Atan()(x)[0]


class Atanh(Operator):
    """
    Calculates the hyperbolic arctangent of the given input tensor element-wise.
    """

    def __init__(self):
        super(Atanh, self).__init__()

    def forward(self, x):
        """
        Args:
            x (CTensor): Input tensor
        Returns:
            CTensor, the output
        """
        if training:
            self.input = x
        return singa.Atanh(x)

    def backward(self, dy):
        """
        Args:
            dy (CTensor): the gradient tensor from upper operations
        Returns:
            CTensor, the gradient over input
        """
        dx = singa.Square(self.input)
        dx = singa.MultFloat(dx, -1.0)
        dx = singa.AddFloat(dx, 1.0)
        dx = singa.PowFloat(dx, -1.0)
        dx *= dy
        return dx


def atanh(x):
    """
    Calculates the hyperbolic arctangent of the given input tensor element-wise.
    Args:
        x (Tensor): Input tensor
    Returns:
        Tensor, the output
    """
    return Atanh()(x)[0]


class Sigmoid(Operator):
    """
    `y = 1 / (1 + exp(-x))`, is applied to the tensor elementwise.
    """

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        """
        Args:
            x (CTensor): Input tensor
        Returns:
            CTensor, the output
        """
        out = singa.Sigmoid(x)
        if training:
            self.cache = (out,)
        return out

    def backward(self, dy):
        """
        Args:
            dy (CTensor): the gradient tensor from upper operations
        Returns:
            CTensor, the gradient over input
        """
        dx = singa.MultFloat(self.cache[0], -1.0)
        dx = singa.AddFloat(dx, 1.0)
        dx = singa.__mul__(self.cache[0], dx)
        dx *= dy
        return dx


def sigmoid(x):
    """
    `y = 1 / (1 + exp(-x))`, is applied to the tensor elementwise.
    Args:
        x (Tensor): Input tensor
    Returns:
        Tensor, the output
    """
    return Sigmoid()(x)[0]


class Mul(Operator):
    """
    Performs element-wise binary multiplication (with Numpy-style broadcasting
    support).
    """

    def __init__(self):
        super(Mul, self).__init__()

    def forward(self, a, b):
        """
        Return `np.multiply(a,b)`, where a and b are CTensor.
        """
        # todo we cannot support mul op for int tensors
        _a, _b = a, b
        dtype0 = _a.data_type()
        dtype1 = _b.data_type()
        if dtype0 == singa.kInt or dtype1 == singa.kInt:
            _a = a.AsType(singa.kFloat32)
            _b = b.AsType(singa.kFloat32)
            res = singa.__mul__(_a, _b)
            res = res.AsType(singa.kInt)
        else:
            res = singa.__mul__(_a, _b)
        if training:
            self.input = (_a, _b)
            self.shape0 = list(_a.shape())
            self.shape1 = list(_b.shape())
            self.shape3 = list(res.shape())
        return res

    def backward(self, dy):
        """
        Args:
            dy (CTensor): the gradient tensor from upper operations
        Returns:
            a tuple for (da, db), da is data for dL / da, db is data
                for dL / db.
        """
        dx0 = singa.__mul__(dy, self.input[1])
        dx1 = singa.__mul__(dy, self.input[0])
        if (type(dy) == float) or self.shape0 == self.shape1:
            assert self.shape0 == self.shape1, ('should have same shape')
            return dx0, dx1
        # handle broadcast
        dx0 = back_broadcast(self.shape3, self.shape0, dx0)
        dx1 = back_broadcast(self.shape3, self.shape1, dx1)
        return dx0, dx1


def mul(x, y):
    """
    Return `np.multiply(x,y)`, where a and b are Tensor.
    """
    return Mul()(x, y)[0]


class Unsqueeze(Operator):
    """
    Insert single-dimensional entries to the shape of an input tensor (data).
    """

    def __init__(self, axis):
        """
        Args:
            axis (list of int): the dimensions to be inserted.
        """
        super(Unsqueeze, self).__init__()
        if (type(axis) is int):
            self.axis = list(axis)
        else:
            self.axis = axis

    def forward(self, x):
        """
        Args:
            x (CTensor): Input tensor
        Returns:
            CTensor, the output
        """
        self.cache = x.shape()
        cur = list(self.cache)
        # todo, need optimize after we have scalar tensor
        if len(self.cache) == 1 and self.axis == [0]:
            return x
        for i in self.axis:
            cur.insert(i, 1)
        return singa.Reshape(x, cur)

    def backward(self, dy):
        """
        Args:
            dy (CTensor): the gradient tensor from upper operations
        Returns:
            CTensor, the gradient over input
        """
        return singa.Reshape(dy, self.cache)


def unsqueeze(x, axis=-1):
    """
    Insert single-dimensional entries to the shape of an input tensor (data).
    Args:
        x (Tensor): Input tensor
        axis (list of int): the dimensions to be inserted.
    Returns:
        Tensor, the output
    """
    return Unsqueeze(axis)(x)[0]


class Transpose(Operator):
    """
    Transpose the input tensor similar to numpy.transpose.
    """

    def __init__(self, perm):
        """
        Args:
            perm (list of ints): A list of integers. By default, reverse the
                dimensions, otherwise permute the axes according to the values given.
        """
        super(Transpose, self).__init__()
        self.perm = list(perm)

    def forward(self, x):
        """
        Args:
            x (CTensor): Input tensor
        Returns:
            CTensor, the output
        """
        return singa.Transpose(x, self.perm)

    def backward(self, dy):
        """
        Args:
            dy (CTensor): the gradient tensor from upper operations
        Returns:
            CTensor, the gradient over input
        """
        cur = []
        for i in range(len(self.perm)):
            cur += [self.perm.index(i)]
        return singa.Transpose(dy, cur)


def transpose(x, shape):
    """
    Transpose the input tensor similar to numpy.transpose.
    Args:
        x (Tensor): Input tensor
        perm (list of ints): A list of integers. By default, reverse the
            dimensions, otherwise permute the axes according to the values given.
    Returns:
        Tensor, the output
    """
    return Transpose(shape)(x)[0]


def add_all(*xs):
    assert len(xs) > 2
    y = add(xs[0], xs[1])
    for x in xs[2:]:
        y = add(y, x)
    return


class Abs(Operator):
    """
    `y = abs(x)`, is applied to the tensor elementwise.
    """

    def forward(self, a):
        """
        Return `abs(a)`, where a is CTensor.
        """
        if training:
            self.input = a
        return singa.Abs(a)

    def backward(self, dy):
        """
        Args:
            dy (CTensor): the gradient tensor from upper operations
        Returns:
            CTensor, the gradient over input
        """
        dx = singa.Sign(self.input)
        dx *= dy
        return dx


def abs(a):
    """
    Return abs(a), where a is Tensor.
    """
    return Abs()(a)[0]


class Exp(Operator):
    """
    `y = exp(x)`, is applied to the tensor elementwise.
    """

    def forward(self, a):
        """
        Return `exp(a)`, where a is Tensor.
        """
        if training:
            self.input = a
        return singa.Exp(a)

    def backward(self, dy):
        """
        Args:
            dy (CTensor): the gradient tensor from upper operations
        Returns:
            CTensor, the gradient over input
        """
        dx = singa.Exp(self.input)
        dx *= dy
        return dx


def exp(a):
    """
    Return `exp(a)`, where a is Tensor.
    """
    return Exp()(a)[0]


class LeakyRelu(Operator):
    """
    `f(x) = alpha * x` for x < 0, `f(x) = x` for x >= 0, is applied to the tensor elementwise.
    """

    def __init__(self, a):
        """
        Args:
            a (float): Coefficient of leakage.
        """
        super(LeakyRelu, self).__init__()
        self.a = a

    def forward(self, x):
        """
        Args:
            x (CTensor): Input tensor
        Returns:
            CTensor, the output
        """
        if training:
            self.input = x
        x1 = singa.LTFloat(x, 0.0)
        x1 = singa.__mul__(x, x1)
        x1 = singa.MultFloat(x1, self.a)
        x2 = singa.ReLU(x)
        x1 = singa.__add__(x1, x2)
        return x1

    def backward(self, dy):
        """
        Args:
            dy (CTensor): the gradient tensor from upper operations
        Returns:
            CTensor, the gradient over input
        """
        # TODO(wangwei) check the correctness
        dx1 = singa.GTFloat(self.input, 0.0)
        dx2 = singa.LTFloat(self.input, 0.0)
        dx2 = singa.MultFloat(dx2, self.a)
        dx = singa.__add__(dx1, dx2)
        dx *= dy
        return dx


def leakyrelu(x, a=0.01):
    """
    `f(x) = alpha * x` for x < 0, `f(x) = x` for x >= 0 is applied to the tensor
    elementwise.
    Args:
        x (Tensor): Input tensor
        a (float): Coefficient of leakage, default to 0.01.
    Returns:
        Tensor, the output
    """
    return LeakyRelu(a)(x)[0]


class Sign(Operator):
    """
    Calculate the sign of the given input tensor element-wise. If input > 0,
    output 1. if input < 0, output -1. if input == 0, output 0.
    """

    def __init__(self):
        super(Sign, self).__init__()

    def forward(self, a):
        """
        Args:
            a (CTensor): Input tensor
        Returns:
            CTensor, the output
        """
        if training:
            self.input = a
        return singa.Sign(a)

    def backward(self, dy):
        """
        Args:
            dy (CTensor): the gradient tensor from upper operations
        Returns:
            CTensor, the gradient over input
        """
        dx = singa.MultFloat(dy, 0.0)
        return dx


def sign(a):
    """
    Calculate the sign of the given input tensor element-wise. If input > 0,
    output 1. if input < 0, output -1. if input == 0, output 0.
    Args:
        a (Tensor): Input tensor
    Returns:
        Tensor, the output
    """
    return Sign()(a)[0]


class Pow(Operator):
    """
    `f(x) = a^b`, is applied to the tensor elementwise.
    """

    def __init__(self):
        super(Pow, self).__init__()

    def forward(self, a, b):
        """
        Return `a^b`, where a and b are CTensor.
        """
        res = singa.Pow(a, b)
        if training:
            self.input = (a, b)
            self.shape0 = list(a.shape())
            self.shape1 = list(b.shape())
            self.shape3 = list(res.shape())
        return res

    def backward(self, dy):
        """
        Args:
            dy (CTensor): the gradient tensor from upper operations
        Returns:
            a tuple for (da, db), da is data for dL / da, db is data
                for dL / db.
        """
        da1 = singa.__mul__(
            self.input[1],
            singa.Pow(self.input[0], singa.SubFloat(self.input[1], 1.0)))
        dx0 = singa.__mul__(da1, dy)
        db1 = singa.__mul__(singa.Pow(self.input[0], self.input[1]),
                            singa.Log(self.input[0]))
        dx1 = singa.__mul__(db1, dy)
        if (type(dy) == float) or self.shape0 == self.shape1:
            assert self.shape0 == self.shape1, ('should have same shape')
            return dx0, dx1
        # handle broadcast
        dx0 = back_broadcast(self.shape3, self.shape0, dx0)
        dx1 = back_broadcast(self.shape3, self.shape1, dx1)
        return dx0, dx1


def pow(a, b):
    """
    Return `a^b`, where a and b are Tensor.
    """
    return Pow()(a, b)[0]


class SoftSign(Operator):
    """
    Calculates the softsign `(x/(1+|x|))` of the given input tensor element-wise.
    """

    def __init__(self):
        super(SoftSign, self).__init__()

    def forward(self, x):
        """
        Return `(x/(1+|x|))`, where x is CTensor.
        """
        # y = x / (1 + np.abs(x))
        if training:
            self.input = x
        x1 = singa.AddFloat(singa.Abs(x), 1.0)
        y = singa.__div__(x, x1)

        return y

    def backward(self, dy):
        """
        Args:
            dy (CTensor): the gradient tensor from upper operations
        Returns:
            CTensor, the gradient over input
        """
        dx = singa.AddFloat(singa.Abs(self.input), 1.0)
        dx = singa.PowFloat(singa.Square(dx), -1.0)
        dx = singa.__mul__(dy, dx)
        return dx


def softsign(x):
    """
    Return `(x/(1+|x|))`, where x is Tensor.
    """
    return SoftSign()(x)[0]


class Sqrt(Operator):
    """
    `y = x^0.5`, is applied to the tensor elementwise.
    """

    def __init__(self):
        super(Sqrt, self).__init__()

    def forward(self, x):
        """
        Return `x^0.5`, where x is CTensor.
        """
        if training:
            self.input = x
        return singa.Sqrt(x)

    def backward(self, dy):
        """
        Args:
            dy (CTensor): the gradient tensor from upper operations
        Returns:
            CTensor, the gradient over input
        """
        dx = singa.PowFloat(self.input, -0.5)
        dx = singa.MultFloat(dx, 0.5)
        dx = singa.__mul__(dy, dx)
        return dx


def sqrt(x):
    """
    Return `x^0.5`, where x is Tensor.
    """
    return Sqrt()(x)[0]


class SoftPlus(Operator):
    """
    `y = ln(exp(x) + 1)` is applied to the tensor elementwise.
    """

    def __init__(self):
        super(SoftPlus, self).__init__()

    def forward(self, x):
        """
        Return `ln(exp(x) + 1)`, where x is CTensor.
        """
        #f(x) = ln(exp(x) + 1)
        if training:
            self.input = x
        x1 = singa.AddFloat(singa.Exp(x), 1.0)
        y = singa.Log(x1)
        return y

    def backward(self, dy):
        """
        Args:
            dy (CTensor): the gradient tensor from upper operations
        Returns:
            CTensor, the gradient over input
        """
        dx = singa.Exp(singa.MultFloat(self.input, -1.0))
        dx = singa.PowFloat(singa.AddFloat(dx, 1.0), -1.0)
        dx = singa.__mul__(dy, dx)
        return dx


def softplus(x):
    """
    Return `ln(exp(x) + 1)`, where x is Tensor.
    """
    return SoftPlus()(x)[0]


class Sub(Operator):
    """
    Performs element-wise binary subtraction (with Numpy-style broadcasting
    support).
    """

    def __init__(self):
        super(Sub, self).__init__()

    def forward(self, a, b):
        """
        Return `a-b`, where x is CTensor.
        """
        res = singa.__sub__(a, b)
        if training:
            self.shape0 = list(a.shape())
            self.shape1 = list(b.shape())
            self.shape3 = list(res.shape())
        return res

    def backward(self, dy):
        """
        Args:
            dy (CTensor): the gradient tensor from upper operations
        Returns:
            a tuple for (da, db), da is data for dL / da, db is data
                for dL / db.
        """
        dx0 = dy
        dx1 = singa.MultFloat(dy, -1.0)
        if (type(dy) == float) or self.shape0 == self.shape1:
            assert self.shape0 == self.shape1, ('should have same shape')
            return dx0, dx1
        # handle broadcast
        dx0 = back_broadcast(self.shape3, self.shape0, dx0)
        dx1 = back_broadcast(self.shape3, self.shape1, dx1)
        return dx0, dx1


def sub(a, b):
    """
    Return a-b, where a and b are Tensor.
    """
    return Sub()(a, b)[0]


# optimize min to support multi inputs
class Min(Operator):
    """
    Element-wise min of each of the input tensors (with Numpy-style
    broadcasting support).
    """

    def __init__(self):
        super(Min, self).__init__()
        self.masks = []

    def _min(self, a, b):
        """
        Args:
            a (CTensor): First operand
            b (CTensor): Second operand
        Returns:
            CTensor, the output
            tuple of CTensor, mask tensor
        """
        m = singa.__sub__(a, b)
        mask0 = singa.LEFloat(m, 0)
        mask1 = singa.GTFloat(m, 0)
        res = singa.__add__(singa.__mul__(mask0, a), singa.__mul__(mask1, b))
        return res, (mask0, mask1)

    def forward(self, *x):
        """
        Args:
            *x (a list of CTensor): List of tensors for max.
        Returns:
            CTensor, the output
        """
        assert (len(x) > 0)
        self.l = len(x)
        if len(x) == 1:
            res, masks = self._min(x[0], x[0])
            self.masks.append(masks)
            return x[0]
        res, masks = self._min(x[0], x[1])
        self.masks.append(masks)
        for i in range(2, len(x)):
            res, masks = self._min(res, x[i])
            self.masks.append(masks)
        return res

    def backward(self, dy):
        """
        Args:
            dy (CTensor): the gradient tensor from upper operations
        Returns:
            a tuple for (*dx), dx is data for dL / dx.
        """
        if self.l == 1:
            return self.masks[0][0]
        else:
            ret = []
            cumulation = None
            for mask0, mask1 in self.masks[::-1]:
                if not cumulation:
                    ret.insert(0, mask1)
                    cumulation = mask0
                else:
                    ret.insert(0, singa.__mul__(cumulation, mask1))
                    cumulation = singa.__mul__(cumulation, mask0)
            ret.insert(0, cumulation)
            return tuple(ret)


def min(*l):
    """
    Element-wise min of each of the input tensors (with Numpy-style
    broadcasting support).
    Args:
        *x (a list of Tensor): List of tensors for max.
    Returns:
        Tensor, the output
    """
    return Min()(*l)[0]


class Log(Operator):
    """
    `y = log(x)`, is applied to the tensor elementwise.
    """

    def __init__(self):
        super(Log, self).__init__()

    def forward(self, x):
        """
        Return `log(x)`, where x is CTensor.
        """
        if training:
            self.input = x
        return singa.Log(x)

    def backward(self, dy):
        """
        Args:
            dy (CTensor): the gradient tensor from upper operations
        Returns:
            CTensor, the gradient over input
        """
        dx = singa.PowFloat(self.input, -1)
        dx = singa.__mul__(dy, dx)
        return dx


def log(x):
    """
    Return log(x), where x is Tensor.
    """
    return Log()(x)[0]


class HardSigmoid(Operator):
    """
    `y = max(0, min(1, alpha * x + beta))`, is applied to the tensor elementwise.
    """

    def __init__(self, alpha=0.2, gamma=0.5):
        """
        Args:
            alpha (float): Value of alpha.
            gamma (float): Value of beta.
        """
        super(HardSigmoid, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x):
        """
        Args:
            x (CTensor): matrix
        Returns:
            a CTensor for the result
        """
        x = singa.AddFloat(singa.MultFloat(x, self.alpha), self.gamma)
        if training:
            self.cache = x

        x = singa.ReLU(x)
        mask1 = singa.LTFloat(x, 1.0)
        mask2 = singa.GEFloat(x, 1.0)

        ans = singa.__add__(singa.__mul__(x, mask1), mask2)
        return singa.ReLU(ans)

    def backward(self, dy):
        """
        Args:
            dy (CTensor): the gradient tensor from upper operations
        Returns:
            CTensor, the gradient over input
        """
        mask0 = singa.GTFloat(self.cache, 0.0)
        mask1 = singa.LTFloat(self.cache, 1.0)
        mask = singa.__mul__(mask0, mask1)
        return singa.__mul__(singa.MultFloat(mask, self.alpha), dy)


def hardsigmoid(x, alpha=0.2, gamma=0.5):
    """
    `y = max(0, min(1, alpha * x + beta))`, is applied to the tensor elementwise.
    Args:
        x (Tensor): matrix
        alpha (float): Value of alpha.
        gamma (float): Value of beta.
    Returns:
        a Tensor for the result
    """
    return HardSigmoid(alpha, gamma)(x)[0]


class Squeeze(Operator):
    """
    Remove single-dimensional entries from the shape of a tensor. Takes a
    parameter axes with a list of axes to squeeze. If axes is not provided,
    all the single dimensions will be removed from the shape. If an axis is
    selected with shape entry not equal to one, an error is raised.
    """

    def __init__(self, axis=[]):
        """
        Args:
            axis (list of ints): List of integers indicating the dimensions
                to squeeze. Negative value means counting dimensions from
                the back. Accepted range is [-r, r-1] where r = rank(data).
        """
        super(Squeeze, self).__init__()
        self.axis = axis

    def forward(self, x):
        """
        Args:
            x (CTensor): Input tensor
        Returns:
            CTensor, the output
        """
        self.cache = x.shape()
        newshape = []
        if (self.axis == []):
            newshape = list(filter(lambda i: i != 1, self.cache))
        else:
            for id, i in enumerate(self.axis):
                assert i < len(self.cache)
                self.axis[id] = i % len(self.cache)
                assert self.cache[
                    i] == 1, "the length of axis {} is {}, which should be 1".format(
                        i, self.cache[i])
            for ind, v in enumerate(self.cache):
                if ind not in self.axis:
                    newshape.append(v)
        # todo, need optimize after we have scalar tensor
        if newshape == []:
            return x
        return singa.Reshape(x, newshape)

    def backward(self, dy):
        """
        Args:
            dy (CTensor): the gradient tensor from upper operations
        Returns:
            CTensor, the gradient over input
        """
        return singa.Reshape(dy, self.cache)


def squeeze(x, axis=[]):
    """
    Remove single-dimensional entries from the shape of a tensor. Takes a
    parameter axes with a list of axes to squeeze. If axes is not provided,
    all the single dimensions will be removed from the shape. If an axis is
    selected with shape entry not equal to one, an error is raised.
    Args:
        x (Tensor): Input tensor
        axis (list of ints): List of integers indicating the dimensions
            to squeeze. Negative value means counting dimensions from
            the back. Accepted range is [-r, r-1] where r = rank(data).
    Returns:
        Tensor, the output
    """
    return Squeeze(axis)(x)[0]


class Div(Operator):
    """
    Performs element-wise binary division (with Numpy-style broadcasting support).
    """

    def __init__(self):
        super(Div, self).__init__()

    def forward(self, a, b):
        """
        Return `np.div(a,b)`, where a and b are CTensor.
        """
        res = singa.__mul__(a, singa.PowFloat(b, -1.0))
        # res = singa.__div__(a, b)
        if training:
            self.input = (singa.MultFloat(a, -1.0), singa.PowFloat(b, -1.0)
                         )  # -a, 1/b
            self.shape0 = list(a.shape())
            self.shape1 = list(b.shape())
            self.shape3 = list(res.shape())
        return res

    def backward(self, dy):
        """
        Args:
            dy (CTensor): the gradient tensor from upper operations
        Returns:
            a CTensor tuple for (da, db), da is data for dL / da, db is data
                for dL / db.
        """
        #dy/dx_0 = b^(-1)
        #dy/dx_1 = (-a)*b^(-2)
        dx0 = singa.__mul__(dy, self.input[1])
        dx1 = singa.__mul__(self.input[0], singa.PowFloat(self.input[1], 2.0))
        dx1 = singa.__mul__(dy, dx1)
        if (type(dy) == float) or self.shape0 == self.shape1:
            assert self.shape0 == self.shape1, ('should have same shape')
            return dx0, dx1
        # handle broadcast
        dx0 = back_broadcast(self.shape3, self.shape0, dx0)
        dx1 = back_broadcast(self.shape3, self.shape1, dx1)
        return dx0, dx1


def div(a, b):
    """
    Return `np.div(a,b)`, where a and b are Tensor.
    """
    return Div()(a, b)[0]


class Shape(Operator):
    """
    Takes a tensor as input and outputs a tensor containing the shape of the
    input tensor.
    """

    def __init__(self):
        super(Shape, self).__init__()

    def forward(self, x):
        """
        Args:
            x (CTensor): Input tensor
        Returns:
            CTensor, the output
        """
        cur = list(x.shape())
        cur = tensor.from_numpy(np.array(cur))
        cur.to_device(x.device())
        return cur.data

    def backward(self, dy):
        """
        Args:
            dy (CTensor): the gradient tensor from upper operations
        Returns:
            list of int, the shape of dy
        """
        return list(dy.shape())


def shape(x):
    """
    Takes a tensor as input and outputs a tensor containing the shape of the
    input tensor.
    Args:
        x (Tensor): Input tensor
    Returns:
        Tensor, the output
    """
    return Shape()(x)[0]


# optimize max to support multi inputs
class Max(Operator):
    """
    Element-wise max of each of the input tensors (with Numpy-style
    broadcasting support).
    """

    def __init__(self):
        super(Max, self).__init__()
        self.masks = []

    def _max(self, a, b):
        """
        Args:
            a (CTensor): First operand
            b (CTensor): Second operand
        Returns:
            CTensor, the output
            tuple of CTensor, mask tensor
        """
        m = singa.__sub__(a, b)
        mask0 = singa.GEFloat(m, 0)
        mask1 = singa.LTFloat(m, 0)
        res = singa.__add__(singa.__mul__(mask0, a), singa.__mul__(mask1, b))
        return res, (mask0, mask1)

    def forward(self, *x):
        """
        Args:
            *x (a list of CTensor): List of tensors for max.
        Returns:
            CTensor, the output
        """
        assert (len(x) > 0)
        self.l = len(x)
        if len(x) == 1:
            res, masks = self._max(x[0], x[0])
            self.masks.append(masks)
            return x[0]
        res, masks = self._max(x[0], x[1])
        self.masks.append(masks)
        for i in range(2, len(x)):
            res, masks = self._max(res, x[i])
            self.masks.append(masks)
        return res

    def backward(self, dy):
        """
        Args:
            dy (CTensor): the gradient tensor from upper operations
        Returns:
            a tuple for (*dx), dx is data for dL / dx.
        """
        if self.l == 1:
            return self.masks[0][0]
        else:
            ret = []
            cumulation = None
            for mask0, mask1 in self.masks[::-1]:
                if not cumulation:
                    ret.insert(0, mask1)
                    cumulation = mask0
                else:
                    ret.insert(0, singa.__mul__(cumulation, mask1))
                    cumulation = singa.__mul__(cumulation, mask0)
            ret.insert(0, cumulation)
            return tuple(ret)


def max(*l):
    """
    Element-wise max of each of the input tensors (with Numpy-style broadcasting support).
    Args:
        *x (a list of Tensor): List of tensors for max.
    Returns:
        Tensor, the output
    """
    return Max()(*l)[0]


class And(Operator):
    """
    Returns the tensor resulted from performing the and logical operation elementwise on the input tensors A and B (with Numpy-style broadcasting support).
    """

    def __init__(self):
        super(And, self).__init__()

    def forward(self, a, b):
        """
        Return `np.logical_and(a,b)`, where a and b are CTensor.
        """
        m = singa.__mul__(a, b)
        cur = singa.PowFloat(singa.Sign(m), 2)

        return cur

    def backward(self, dy):
        """
        Args:
            dy (CTensor): the gradient tensor from upper operations
        Raises:
            AssertionError: no backward function for this operator
        """
        assert False, ('no gradient for backward function')


def _and(a, b):
    """
    Return `np.logical_and(a,b)`, where a and b are Tensor.
    """
    return And()(a, b)[0]


class Or(Operator):
    """
    Returns the tensor resulted from performing the or logical operation elementwise on the input tensors A and B (with Numpy-style broadcasting support).
    """

    def __init__(self):
        super(Or, self).__init__()

    def forward(self, a, b):
        """
        Return `np.logical_or(a,b)`, where a and b are CTensor.
        """
        m = singa.__add__(singa.PowFloat(singa.Sign(a), 2.0),
                          singa.PowFloat(singa.Sign(b), 2.0))
        cur = singa.Sign(m)

        return cur

    def backward(self, dy):
        """
        Args:
            dy (CTensor): data for the `dL / dy`, L is the loss.
        Raises:
            AssertionError: no backward function for this operator
        """
        assert False, ('no gradient for backward function')


def _or(a, b):
    """
    Return np.logical_or(a,b), where a and b are Tensor.
    """
    return Or()(a, b)[0]


class Not(Operator):
    """
    Returns the negation of the input tensor element-wise.
    """

    def __init__(self):
        super(Not, self).__init__()

    def forward(self, x):
        """
        Return `np.logical_not(x)`, where x is CTensor.
        """
        mask0 = singa.GEFloat(x, 0)
        mask1 = singa.LEFloat(x, 0)
        cur = singa.__mul__(mask0, mask1)

        return cur

    def backward(self, dy):
        """
        Args:
            dy (CTensor): the gradient tensor from upper operations
        Raises:
            AssertionError: no backward function for this operator
        """
        assert False, ('no gradient for backward function')


def _not(x):
    """
    Return `np.logical_not(x)`, where x is Tensor.
    """
    return Not()(x)[0]


class Xor(Operator):
    """
    Performing the xor logical operation elementwise on the input tensors A and B (with Numpy-style broadcasting support).
    """

    def __init__(self):
        super(Xor, self).__init__()

    def forward(self, a, b):
        """
        Return `np.logical_xor(a,b)`, where a and b are CTensor.
        """
        m = singa.__sub__(singa.PowFloat(singa.Sign(a), 2.0),
                          singa.PowFloat(singa.Sign(b), 2.0))
        cur = singa.PowFloat(singa.Sign(m), 2.0)

        return cur

    def backward(self, dy):
        """
        Args:
            dy (CTensor): the gradient tensor from upper operations
        Raises:
            AssertionError: no backward function for this operator
        """
        assert False, ('no gradient for backward function')


def _xor(a, b):
    """
    Return `np.logical_xor(a,b)`, where a and b are Tensor.
    """
    return Xor()(a, b)[0]


class Negative(Operator):
    """
    `y = -x`, is applied to the tensor elementwise.
    """

    def __init__(self):
        super(Negative, self).__init__()

    def forward(self, x):
        """
        Return `-x`, where x is CTensor.
        """
        #y=-x
        return singa.MultFloat(x, -1)

    def backward(self, dy):
        """
        Args:
            dy (CTensor): the gradient tensor from upper operations
        Returns:
            CTensor, the gradient over input
        """
        return singa.MultFloat(dy, -1)


def negative(x):
    """
    Return `-x`, where x is Tensor.
    """
    return Negative()(x)[0]


class Reciprocal(Operator):
    """
    `y = 1/x`, is applied to the tensor elementwise.
    """

    def __init__(self):
        super(Reciprocal, self).__init__()

    def forward(self, x):
        """
        Return `1/x`, where x is CTensor.
        """
        #y=1/x elementwise
        if training:
            self.input = x

        return singa.PowFloat(x, -1)

    def backward(self, dy):
        """
        Args:
            dy (CTensor): the gradient tensor from upper operations
        Returns:
            CTensor, the gradient over input
        """
        #dy/dx = -1/x**2
        dx = singa.MultFloat(singa.PowFloat(self.input, -2), -1)
        return singa.__mul__(dy, dx)


def reciprocal(x):
    """
    Return 1/x, where x is Tensor.
    """
    return Reciprocal()(x)[0]


class Gemm(Operator):
    """
    Init a General Matrix multiplication(Gemm) operator. Compute `Y = alpha *
    A' * B' + beta * C`, where input tensor A has shape (M, K) or (K, M), input
    tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to
    shape (M, N), and output tensor Y has shape (M, N).
    `A' = transpose(A)` if transA else A
    `B' = transpose(B)` if transB else B
    """

    def __init__(self, alpha=1.0, beta=1.0, transA=0, transB=0):
        """
        Args:
            alpha (float): Scalar multiplier for the product of input tensors
                A * B.
            beta (float): Scalar multiplier for input tensor C.
            ransA (int): Whether A should be transposed
            transB (int): Whether B should be transposed
        Returns:
            CTensor, the output
        """
        super(Gemm, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.transA = transA
        self.transB = transB

    def forward(self, A, B, C=None):
        """
        forward propogation of Gemm
        Args:
            A (CTensor): The shape of A should be (M, K) if transA is 0, or
                (K, M) if transA is non-zero.
            B (CTensor): The shape of B should be (K, N) if transB is 0, or
                (N, K) if transB is non-zero.
            C (CTensor): (optional), Optional input tensor C. If not specified,
                the computation is done as if C is a scalar 0. The shape of C
                should be unidirectional broadcastable to (M, N).
        Returns:
            tensor, the output
        """
        _A = singa.DefaultTranspose(A) if self.transA == 1 else A
        _B = singa.DefaultTranspose(B) if self.transB == 1 else B
        if training:
            self.inputs = (_A, _B, C)
        tmpM = singa.MultFloat(singa.Mult(_A, _B), self.alpha)
        if C:
            tmpM = singa.__add__(tmpM, singa.MultFloat(C, self.beta))
        return tmpM

    def backward(self, dy):
        """
        backward propogation of Gemm
        Args:
            dy (CTensor): The shape of A should be (M, K) if transA is 0, or (K, M) if transA is non-zero.
        Returns:
            CTensor, the gradient over A
            CTensor, the gradient over B
            CTensor(optional), the gradient over C
        """
        _A, _B, C = self.inputs
        # y = alpha * A  * B  => da = alpha * dy * BT
        # y = alpha * A  * BT => da = alpha * dy * B
        # y = alpha * AT * B  => da = alpha * B * dyT = alpha * (dy * BT)T
        # y = alpha * AT * BT => da = alpha * BT * dyT = alpha * (dy * B)T
        da = singa.MultFloat(singa.Mult(dy, singa.DefaultTranspose(_B)),
                             self.alpha)
        if self.transA:
            da = singa.DefaultTranspose(da)

        # y = alpha * A  * B  => db = alpha * AT * dy
        # y = alpha * AT * B  => db = alpha * A * dy
        # y = alpha * A  * BT => db = alpha * dyT * A = alpha * (AT * dy)T
        # y = alpha * AT * BT => db = alpha * dyT * AT = alpha * (A * dy)T
        db = singa.MultFloat(singa.Mult(singa.DefaultTranspose(_A), dy),
                             self.alpha)
        if self.transB:
            db = singa.DefaultTranspose(db)
        if C:
            dc = back_broadcast(dy.shape(), C.shape(),
                                singa.MultFloat(dy, self.beta))
            return da, db, dc
        else:
            return da, db


def gemm(A, B, C=None, alpha=1.0, beta=1.0, transA=0, transB=0):
    """
    Init a General Matrix multiplication(Gemm) operator. Compute `Y = alpha *
    A' * B' + beta * C`, where input tensor A has shape (M, K) or (K, M), input
    tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to
    shape (M, N), and output tensor Y has shape (M, N).
    `A' = transpose(A)` if transA else A
    `B' = transpose(B)` if transB else B
    Args:
        A (Tensor): The shape of A should be (M, K) if transA is 0, or
            (K, M) if transA is non-zero.
        B (Tensor): The shape of B should be (K, N) if transB is 0, or
            (N, K) if transB is non-zero.
        C (Tensor): (optional), Optional input tensor C. If not specified,
            the computation is done as if C is a scalar 0. The shape of C
            should be unidirectional broadcastable to (M, N).
        alpha (float): Scalar multiplier for the product of input tensors A * B.
        beta (float): Scalar multiplier for input tensor C.
        ransA (int): Whether A should be transposed
        transB (int): Whether B should be transposed
    Returns:
        Tensor, the output
    """
    if C:
        return Gemm(alpha, beta, transA, transB)(A, B, C)[0]
    else:
        return Gemm(alpha, beta, transA, transB)(A, B)[0]


class GlobalAveragePool(Operator):
    """
    Init a GlobalAveragePool operator
    """

    def __init__(self, data_format='channels_first'):
        """
        Args:
            data_format (string): A string, we support two formats:
                channels_last and channels_first, default is channels_first.
                channels_first means the format of input is (N x C x H x W)
                channels_last means the format of input is (N x H x W x C)
        """
        super(GlobalAveragePool, self).__init__()
        self.data_format = data_format

    def forward(self, x):
        """
        forward propogation of GlobalAveragePool
        Args:
            x (CTensor): the input tensor
        Returns:
            CTensor, the output
        """
        if training:
            self.mask = singa.Tensor(x.shape(), x.device())

        shape = list(x.shape())

        # (N x C x H x W) for channels_first
        if self.data_format == 'channels_first':
            axes = tuple(i for i in range(2, len(shape)))
            self.shape_divisor = 1 / np.prod(shape[2:])
        else:  # (N x H x W x C) for channels_last
            axes = tuple(i for i in range(1, len(shape) - 1))
            self.shape_divisor = 1 / np.prod(shape[1:-1])

        # output shape
        # (N x C x 1 x 1) for channels_first
        # (N x 1 x 1 x C) for channels_last
        for i in axes:
            shape[i] = 1

        x = tensor.from_raw_tensor(x)
        x = tensor.sum(x, axis=axes)
        x = tensor.reshape(x, shape)
        return singa.MultFloat(x.data, self.shape_divisor)

    def backward(self, dy):
        """
        backward propogation of GlobalAveragePool
        Args:
            dy (CTensor): the gradient tensor from upper operations
        Returns:
            CTensor, the gradient over input
        """
        self.mask.SetFloatValue(self.shape_divisor)
        return singa.__mul__(self.mask, dy)


def globalaveragepool(x, data_format='channels_first'):
    """
    GlobalAveragePool operator
    Args:
        x (Tensor): the input tensor
        data_format (string): A string, we support two formats:
            channels_last and channels_first, default is channels_first.
            channels_first means the format of input is (N x C x H x W)
            channels_last means the format of input is (N x H x W x C)
    Returns:
        Tensor, the output
    """
    return GlobalAveragePool(data_format)(x)[0]


class ConstantOfShape(Operator):
    """
    Init a ConstantOfShape, generate a tensor with given value and shape.
    """

    def __init__(self, value=0.):
        """
        Args:
            value (float): (Optional) The value of the output elements. Should
                be a one-element value. If not specified, it defaults to 0 and
                datatype float32
        """
        super(ConstantOfShape, self).__init__()
        self.value = value

    def forward(self, x):
        """
        forward of ConstantOfShape
        Args:
            x: CTensor, 1D tensor. The shape of the expected output tensor.
                All values must be >= 0.
        Returns:
            the output CTensor. If attribute 'value' is specified, the value
                and datatype of the output tensor is taken from 'value'. If
                attribute 'value' is not specified, the value in the output
                defaults to 0, and the datatype defaults to float32.
        """
        x_shape = tensor.to_numpy(tensor.from_raw_tensor(x)).astype(
            np.int64).tolist()
        assert np.min(x_shape) >= 0, ('shape cannot be negative')
        x = CTensor(x_shape, x.device())
        x.SetFloatValue(self.value)
        return x

    def backward(self, dy):
        """
        backward of ConstantOfShape
        Args:
            dy (CTensor): gradient tensor.
        Raises:
            AssertionError: no backward function for this operator
        """
        assert False, ('no gradient for backward function')


def constant_of_shape(x, value=0):
    """
    Init a ConstantOfShape, generate a tensor with given value and shape.
    Args:
        x: Tensor, 1D tensor. The shape of the expected output tensor.
            All values must be >= 0.
        value (float): (Optional) The value of the output elements. Should
            be a one-element value. If not specified, it defaults to 0 and
            datatype float32
    Returns:
        the output Tensor. If attribute 'value' is specified, the value
            and datatype of the output tensor is taken from 'value'. If
            attribute 'value' is not specified, the value in the output
            defaults to 0, and the datatype defaults to float32.
    """
    return ConstantOfShape(value)(x)[0]


class Dropout(Operator):
    """
    Init a Dropout, which scales the masked input data by the following equation:
    `output = scale * data * mask`, `scale = 1. / (1. - ratio)`.
    """

    def __init__(self, ratio=0.5):
        """
        Args:
            ratio (float): the ratio of random dropout, with value in [0, 1).
        """
        super(Dropout, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        """
        forward of Dropout
        Args:
            x (CTensor): input tensor.
        Returns:
            the output CTensor.
        """
        if training:
            self.scale = 1 / 1 - self.ratio
            self.mask = singa.Tensor(list(x.shape()), x.device())
            singa.Bernoulli(1 - self.ratio, self.mask)
            x = singa.MultFloat(singa.__mul__(self.mask, x), self.scale)
        return x

    def backward(self, dy):
        """
        backward of Dropout
        Args:
            dy (CTensor): gradient tensor.
        Returns:
            the gradient tensor over input tensor.
        """
        if training:
            dy = singa.MultFloat(singa.__mul__(self.mask, dy), self.scale)
        return dy


def dropout(x, ratio=0.5):
    """
    Init a Dropout, which scales the masked input data by the following
    equation: `output = scale * data * mask`, `scale = 1. / (1. - ratio)`.
    Args:
        x (Tensor): input tensor.
        ratio (float): the ratio of random dropout, with value in [0, 1).
    Returns:
        the output Tensor.
    """
    return Dropout(ratio)(x)[0]


class ReduceSum(Operator):
    """
    Init a ReduceSum, computes the sum of the input tensor's element along
    the provided axes.
    """

    def __init__(self, axes=None, keepdims=1):
        """
        Args:
            axes (list of int): A list of integers, along which to reduce.
                Accepted range is [-r, r-1] where r = rank(data). The default
                is None, which reduces over all the dimensions of the input tensor.
            keepdims (int): Keep the reduced dimension or not, default 1 mean
                keep reduced dimension.
        """
        super(ReduceSum, self).__init__()
        self.axes = axes
        self.keepdims = keepdims

    def forward(self, x):
        """
        forward of ReduceSum
        Args:
            x (CTensor): input tensor.
        Returns:
            the output CTensor.
        """
        _x = tensor.from_raw_tensor(x)
        x_shape = list(_x.shape)
        # handle the special axes
        if self.axes is None:
            self.axes = [i for i in range(len(x_shape))]  # axes = None
        else:
            self.axes = [i if i >= 0 else len(x_shape) + i for i in self.axes
                        ]  # axes has negative
        self.axes.sort(reverse=True)
        for axis in self.axes:
            _x = tensor.sum(_x, axis)
            x_shape[axis] = 1
        if self.keepdims == 1:
            _x = tensor.reshape(_x, x_shape)
        self.cache = (x_shape, x)
        return _x.data

    def backward(self, dy):
        """
        backward of ReduceSum
        Args:
            dy (CTensor): gradient tensor.
        Returns:
            the gradient tensor over input tensor.
        """
        x_shape, x = self.cache
        dy = singa.Reshape(dy, x_shape)
        scale = np.prod(x_shape) / np.prod(x.shape())
        mask = singa.Tensor(list(x.shape()), x.device())
        mask.SetFloatValue(scale)
        dy = singa.__mul__(mask, dy)
        return dy


def reduce_sum(x, axes=None, keepdims=1):
    """
    Init a ReduceSum, computes the sum of the input tensor's element along
    the provided axes.
    Args:
        x (Tensor): input tensor.
        axes (list of int): A list of integers, along which to reduce.
            Accepted range is [-r, r-1] where r = rank(data). The default
            is None, which reduces over all the dimensions of the input tensor.
        keepdims (int): Keep the reduced dimension or not, default 1 mean
            keep reduced dimension.
    Returns:
        the output Tensor.
    """
    return ReduceSum(axes, keepdims)(x)[0]


class ReduceMean(Operator):
    """
    Init a ReduceMean, computes the mean of the input tensor's element along
    the provided axes.
    """

    def __init__(self, axes=None, keepdims=1):
        """
        Args:
            axes (list of int): A list of integers, along which to reduce.
                Accepted range is [-r, r-1] where r = rank(data). The default
                is None, which reduces over all the dimensions of the input tensor.
            keepdims (int): Keep the reduced dimension or not, default 1 mean
                keep reduced dimension.
        """
        super(ReduceMean, self).__init__()
        self.axes = axes
        self.keepdims = keepdims

    def forward(self, x):
        """
        forward of ReduceMean
        Args:
            x (CTensor): input tensor.
        Returns:
            the output CTensor.
        """
        _x = tensor.from_raw_tensor(x)
        x_shape = list(_x.shape)
        # handle the special axes
        if self.axes is None:
            self.axes = [i for i in range(len(x_shape))]  # axes = None
        else:
            self.axes = [i if i >= 0 else len(x_shape) + i for i in self.axes
                        ]  # axes has negative
        self.axes.sort(reverse=True)
        for axis in self.axes:
            _x = tensor.sum(_x, axis)
            x_shape[axis] = 1
        if self.keepdims == 1:
            _x = tensor.reshape(_x, x_shape)
        self.cache = (x_shape, x)
        scale = np.prod(x_shape) / np.prod(x.shape())
        _x = singa.MultFloat(_x.data, scale)
        return _x

    def backward(self, dy):
        """
        backward of ReduceMean
        Args:
            dy (CTensor): gradient tensor.
        Returns:
            the gradient tensor over input tensor.
        """
        x_shape, x = self.cache
        dy = singa.Reshape(dy, x_shape)
        mask = singa.Tensor(list(x.shape()), x.device())
        mask.SetFloatValue(1.0)
        dy = singa.__mul__(mask, dy)
        return dy


def reduce_mean(x, axes=None, keepdims=1):
    """
    Init a ReduceMean, computes the mean of the input tensor's element along
    the provided axes.
    Args:
        x (Tensor): input tensor.
        axes (list of int): A list of integers, along which to reduce.
            Accepted range is [-r, r-1] where r = rank(data). The default
            is None, which reduces over all the dimensions of the input tensor.
        keepdims (int): Keep the reduced dimension or not, default 1 mean
            keep reduced dimension.
    Returns:
        the output Tensor.
    """
    return ReduceMean(axes, keepdims)(x)[0]


class Slice(Operator):
    """
    Init a Slice, Produces a slice of the input tensor along multiple axes.
    Similar to numpy: https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
    """

    def __init__(self, starts, ends, axes=None, steps=None):
        """
        Args:
            starts (list of int): starting indices of corresponding axis
            ends (list of int): ending indices of corresponding axis
            axes (list of int): axes that `starts` and `ends` apply to.
                Negative value means counting dimensions from the back.
                Accepted range is [-r, r-1] where r = rank(data).
            steps (list of int): slice step of corresponding axis in `axes`.
                Negative value means slicing backward. 'steps' cannot be 0.
                Defaults to 1.
        """
        super(Slice, self).__init__()
        self.starts = starts
        self.ends = ends
        self.axes = axes
        self.steps = steps

    def forward(self, x):
        """
        forward of Slice
        Args:
            x (CTensor): input tensor.
        Returns:
            the output CTensor.
        """
        x_shape = list(x.shape())
        # handle the special axes
        if self.axes is None:
            self.axes = [i for i in range(len(x_shape))]  # axes = None
        else:
            self.axes = [i if i >= 0 else len(x_shape) + i for i in self.axes
                        ]  # axes has negative
        self.cache = []
        # handle the special steps
        if self.steps is None:
            self.steps = [1] * len(x_shape)  # steps = None
        for idx, axis in enumerate(self.axes):
            axis = int(axis)
            start, end, step = self.starts[idx], self.ends[idx], self.steps[idx]
            if end > x_shape[axis]:
                end = x_shape[axis]
            self.cache.append((axis, x_shape[axis], start, end, step))
            xs = []
            for step_idx in range(x_shape[axis])[start:end:step]:
                xs.append(singa.SliceOn(x, step_idx, step_idx + 1, axis))
            assert len(xs) > 0, "Cannot support empty tensor"
            x = singa.VecTensor(xs)
            x = singa.ConcatOn(x, axis)
        return x

    def backward(self, dy):
        """
        backward of Slice
        Args:
            dy (CTensor): gradient tensor.
        Returns:
            the gradient tensor over input tensor.
        """
        for axis, shape, start, end, step in self.cache[::-1]:
            data_idxes = tuple(range(shape)[start:end:step])
            dys = []
            data_idx = 0
            for step_idx in range(shape):
                if step_idx in data_idxes:
                    tmp_tensor = singa.SliceOn(dy, data_idx, data_idx + 1, axis)
                    data_idx += 1
                else:
                    tmp_shape = list(dy.shape())
                    tmp_shape[axis] = 1
                    tmp_tensor = singa.Tensor(tmp_shape, dy.device())
                    tmp_tensor.SetFloatValue(0.)
                dys.append(tmp_tensor)
            dys = singa.VecTensor(dys)
            dy = singa.ConcatOn(dys, axis)
        return dy


def slice(x, starts, ends, axes=None, steps=None):
    """
    Init a Slice, Produces a slice of the input tensor along multiple axes.
    Similar to numpy: https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
    Args:
        x (Tensor): input tensor.
        starts (list of int): starting indices of corresponding axis
        ends (list of int): ending indices of corresponding axis
        axes (list of int): axes that `starts` and `ends` apply to.
            Negative value means counting dimensions from the back.
            Accepted range is [-r, r-1] where r = rank(data).
        steps (list of int): slice step of corresponding axis in `axes`.
            Negative value means slicing backward. 'steps' cannot be 0.
            Defaults to 1.
    Returns:
        the output Tensor.
    """
    return Slice(starts, ends, axes, steps)(x)[0]


class Ceil(Operator):
    """
    Ceil takes one input data (Tensor) and produces one output data (Tensor)
    where the ceil is, `y = ceil(x)`, is applied to the tensor elementwise.
    """

    def __init__(self):
        super(Ceil, self).__init__()

    def forward(self, x):
        """
        forward of Ceil
        Args:
            x (CTensor): input tensor.
        Returns:
            the output CTensor.
        """
        return singa.Ceil(x)

    def backward(self, dy):
        """
        backward of Ceil
        Args:
            dy (CTensor): gradient tensor.
        Returns:
            the gradient tensor over input tensor.
        """
        dy = singa.Tensor(dy.shape(), dy.device())
        dy.SetFloatValue(0.)
        return dy


def ceil(x):
    """
    Ceil takes one input data (Tensor) and produces one output data (Tensor)
    where the ceil is, `y = ceil(x)`, is applied to the tensor elementwise.
    Args:
        x (Tensor): input tensor.
    Returns:
        the output Tensor.
    """
    return Ceil()(x)[0]


class Split(Operator):
    """
    Init a Split, Split a tensor into a list of tensors, along the specified
    'axis'.
    """

    def __init__(self, axis, parts, num_output=None):
        """
        Args:
            axis (int): which axis to split on. A negative value means
                counting dimensions from the back. Accepted range is
                [-rank, rank-1] where r = rank(input).
            parts (list of int): length of each output, which can be specified
                using argument 'parts'. Otherwise, the tensor is parts to equal
                sized parts.
            num_output (bool): once parts is none, the tensor is split to equal
                sized parts for each output.
        """
        super(Split, self).__init__()
        self.axis = axis
        self.parts = parts
        self.num_output = num_output
        if self.parts is None:
            assert self.num_output is not None, "For (parts, num_output), it at least requires one."

    def forward(self, x):
        """
        forward of Split
        Args:
            x (CTensor): input tensor.
        Returns:
            the output CTensor.
        """
        x_shape = list(x.shape())
        self.axis = self.axis % len(x_shape)
        if self.parts is None:
            self.parts = [x_shape[self.axis] // self.num_output
                         ] * self.num_output
        xs = []
        _s = 0
        for _l in self.parts:
            xs.append(singa.SliceOn(x, _s, _s + _l, self.axis))
            _s += _l
        return tuple(xs)

    def backward(self, *dys):
        """
        backward of Split
        Args:
            dys: list of CTensor, gradient tensor.
        Returns:
            the gradient tensor over input tensor.
        """
        dys = singa.VecTensor(dys)
        dy = singa.ConcatOn(dys, self.axis)
        return dy


def split(x, axis, parts, num_output=None):
    """
    Init a Split, Split a tensor into a list of tensors, along the specified
    'axis'.
    Args:
        x (Tensor): input tensor.
        axis (int): which axis to split on. A negative value means
            counting dimensions from the back. Accepted range is
            [-rank, rank-1] where r = rank(input).
        parts (list of int): length of each output, which can be specified
            using argument 'parts'. Otherwise, the tensor is parts to equal
            sized parts.
        num_output (bool): once parts is none, the tensor is split to equal
            sized parts for each output.
    Returns:
        the output Tensor.
    """
    return Split(axis, parts, num_output)(x)


class Gather(Operator):
    """
    Init a Gather, Given data tensor of rank r >= 1, and indices tensor of
    rank q, gather entries of the axis dimension of data (by default outer-most
    one as axis=0) indexed by indices, and concatenates them in an output tensor of rank `q + (r - 1)`.
    """

    def __init__(self, axis, indices):
        """
        Args:
            axis (int): which axis to slice on. A negative value means counting
                dimensions from the back. Accepted range is [-rank, rank-1]
                where r = rank(input).
            indices (list of int): entries of the axis dimension of data.
        """
        super(Gather, self).__init__()
        self.axis = axis
        self.indices = indices

    def forward(self, x):
        """
        forward of Gather
        Args:
            x (CTensor): input tensor.
        Returns:
            the output CTensor.
        """
        self.x_shape = list(x.shape())
        self.axis = self.axis % len(self.x_shape)  # handle the negative value
        _shape = self.x_shape[self.axis]
        xs = []
        for indice in self.indices:
            # each indice is a sub-indice
            if isinstance(indice, tuple) or isinstance(indice, list):
                sub_xs = []
                for idx in indice:
                    idx = idx % _shape
                    tmp_tensor = singa.SliceOn(x, idx, idx + 1, self.axis)
                    sub_xs.append(tmp_tensor)
                sub_xs = singa.VecTensor(sub_xs)
                tmp_tensor = singa.ConcatOn(sub_xs, self.axis)
                _slice_shape = list(tmp_tensor.shape())
                _slice_shape.insert(self.axis, 1)  # add a new axis to concat
                tmp_tensor = singa.Reshape(tmp_tensor, _slice_shape)
            else:
                indice = int(indice % _shape)
                tmp_tensor = singa.SliceOn(x, indice, indice + 1, self.axis)
            xs.append(tmp_tensor)
        xs = singa.VecTensor(xs)
        return singa.ConcatOn(xs, self.axis)

    def backward(self, dy):
        """
        backward of Gather
        Args:
            dy (CTensor): gradient tensor.
        Returns:
            the gradient tensor over input tensor.
        """
        _shape = self.x_shape[self.axis]

        def construct_dx(dy, axis, indices, _shape):
            dys = []
            data_idx = 0
            data_idxes = tuple(indices)
            for step_idx in range(_shape):
                if step_idx in data_idxes:
                    tmp_tensor = singa.SliceOn(dy, data_idx, data_idx + 1, axis)
                    data_idx += 1
                else:
                    tmp_shape = list(dy.shape())
                    tmp_shape[axis] = 1
                    tmp_tensor = singa.Tensor(tmp_shape, dy.device())
                    tmp_tensor.SetFloatValue(0.)
                dys.append(tmp_tensor)
            dys = singa.VecTensor(dys)
            dy = singa.ConcatOn(dys, axis)
            return dy

        if isinstance(self.indices[0], tuple) or isinstance(
                self.indices[0], list):
            dx = singa.Tensor(self.x_shape, dy.device())
            dx.SetFloatValue(0.)
            for data_idx in range(len(self.indices)):
                # get a piece of the dy and remove its new axis added at forward
                tmp_tensor = singa.SliceOn(dy, data_idx, data_idx + 1,
                                           self.axis)
                _slice_shape = list(tmp_tensor.shape())
                del _slice_shape[self.axis]
                tmp_tensor = singa.Reshape(tmp_tensor, _slice_shape)
                # construct dx and sum them together
                tmp_tensor = construct_dx(tmp_tensor, self.axis,
                                          self.indices[data_idx],
                                          self.x_shape[self.axis])
                dx = singa.__add__(dx, tmp_tensor)
            return dx
        else:
            return construct_dx(dy, self.axis, self.indices, _shape)


def gather(x, axis, indices):
    """
    Init a Gather, Given data tensor of rank r >= 1, and indices tensor of
    rank q, gather entries of the axis dimension of data (by default outer-most
    one as axis=0) indexed by indices, and concatenates them in an output tensor of rank `q + (r - 1)`.
    Args:
        x (Tensor): input tensor.
        axis (int): which axis to slice on. A negative value means counting
            dimensions from the back. Accepted range is [-rank, rank-1]
            where r = rank(input).
        indices (list of int): entries of the axis dimension of data.
    Returns:
        the output Tensor.
    """
    return Gather(axis, indices)(x)[0]


class Tile(Operator):
    """
    Init a Tile, Constructs a tensor by tiling a given tensor. This is the same
    as function tile in Numpy: https://docs.scipy.org/doc/numpy/reference/generated/numpy.tile.html
    """

    def __init__(self, repeats):
        """
        Args:
            repeats (list of int): 1D int matrix of the same length as input's
                dimension number, includes numbers of repeated copies along
                input's dimensions.
        """
        super(Tile, self).__init__()
        self.repeats = [repeats] if isinstance(repeats, int) else repeats

    def forward(self, x):
        """
        forward of Tile
        Args:
            x (CTensor): input tensor.
        Returns:
            the output CTensor.
        """
        self.x_shape = list(x.shape())
        # add new axis from head
        if len(self.x_shape) < len(self.repeats):
            append_len = len(self.repeats) - len(self.x_shape)
            new_shape = [1] * append_len + self.x_shape
            x = singa.Reshape(x, new_shape)
        for axis, rp in enumerate(self.repeats):
            if rp == 1:
                continue
            xs = []
            for idx in range(rp):
                xs.append(x.Clone())
            xs = singa.VecTensor(xs)
            x = singa.ConcatOn(xs, axis)
        return x

    def backward(self, dy):
        """
        backward of Tile
        Args:
            dy (CTensor): gradient tensor.
        Returns:
            the gradient tensor over input tensor.
        """
        for axis, rp in enumerate(self.repeats):
            if rp == 1:
                continue
            _slice_shape = list(dy.shape())
            ori_len = _slice_shape[axis] // rp
            _slice_shape[axis] = ori_len
            _dy = singa.Tensor(_slice_shape, dy.device())
            _dy.SetFloatValue(0.)

            for idx in range(rp):
                tmp_tensor = singa.SliceOn(dy, ori_len * idx,
                                           ori_len * (idx + 1), axis)
                _dy = singa.__add__(_dy, tmp_tensor)
            dy = _dy
        # remove the new axis we added at forward
        if len(self.x_shape) < len(self.repeats):
            dy = singa.Reshape(dy, self.x_shape)
        return dy


def tile(x, repeats):
    """
    Init a Tile, Constructs a tensor by tiling a given tensor. This is the same
    as function tile in Numpy: https://docs.scipy.org/doc/numpy/reference/generated/numpy.tile.html
    Args:
        x (Tensor): input tensor.
        repeats (list of int): 1D int matrix of the same length as input's
            dimension number, includes numbers of repeated copies along
            input's dimensions.
    Returns:
        the output Tensor.
    """
    return Tile(repeats)(x)[0]


class NonZero(Operator):
    """
    Init a NonZero, Constructs a tensor by tiling a given tensor. This is the same
    as function tile in Numpy: https://docs.scipy.org/doc/numpy/reference/generated/numpy.tile.html
    """

    def __init__(self):
        super(NonZero, self).__init__()

    def forward(self, x):
        """
        forward of NonZero
        Args:
            x (CTensor): input tensor.
        Returns:
            the output CTensor.
        """
        y = tensor.to_numpy(tensor.from_raw_tensor(x))
        y = np.array((np.nonzero(y))).astype(np.int32)
        y = tensor.from_numpy(y)
        y.to_device(x.device())
        return y.data

    def backward(self, dy):
        """
        backward of NonZero
        Args:
            dy (CTensor): gradient tensor.
        Raises:
            AssertionError: no backward function for this operator
        """
        assert False, ('no gradient for backward function')


def nonzero(x):
    """
    Init a NonZero, Constructs a tensor by tiling a given tensor. This is the same
    as function tile in Numpy: https://docs.scipy.org/doc/numpy/reference/generated/numpy.tile.html
    Args:
        x (Tensor): input tensor.
    Returns:
        the output Tensor.
    """
    return NonZero()(x)[0]


class Cast(Operator):
    """
    The operator casts the elements of a given input tensor to a data type
    specified by the 'to' argument and returns an output tensor of the same
    size in the converted type.
    """

    def __init__(self, to):
        """
        Args:
            to (int): data type, float32 = 0; int = 2.
        """
        super(Cast, self).__init__()
        self.to = to

    def forward(self, x):
        """
        forward of Cast
        Args:
            x (CTensor): input tensor.
        Returns:
            the output CTensor.
        """
        if x.data_type() != self.to:
            x = x.AsType(self.to)
        return x

    def backward(self, dy):
        """
        backward of Cast
        Args:f
            dy (CTensor), gradient tensor.
        Raises:
            AssertionError: no backward function for this operator
        """
        assert False, ('no gradient for backward function')


def cast(x, to):
    """
    The operator casts the elements of a given input tensor to a data type
    specified by the 'to' argument and returns an output tensor of the same
    size in the converted type.
    Args:
        x (Tensor): input tensor.
        to (int): data type, float32 = 0; int = 2.
    Returns:
        the output Tensor.
    """
    return Cast(to)(x)[0]


class OneHot(Operator):
    """
    Produces a one-hot tensor based on inputs.
    """

    def __init__(self, axis, depth, values):
        """
        Args:
            axis (int): Axis along which one-hot representation in added.
                Default: axis=-1. axis=-1 means that the additional dimension
                will be inserted as the innermost/last dimension in the output
                tensor.
            depth (int): Scalar specifying the number of classes in one-hot
                tensor. This is also the size of the one-hot dimension
                (specified by 'axis' attribute) added on in the output tensor.
                The values in the 'indices' input tensor are expected to be in
                the range [-depth, depth-1].
            values (float): Rank 1 tensor containing exactly two elements, in
                the format [off_value, on_value], where 'on_value' is the
                value used for filling locations specified in 'indices' input
                tensor,
        """
        super(OneHot, self).__init__()
        self.axis = axis
        self.depth = depth
        self.values = values

    def forward(self, indices):
        """
        forward of OneHot, we borrow this function from onnx
        Args:
            indices (CTensor): Scalar specifying the number of classes in
                one-hot tensor. The values in the 'indices' input tensor are
                expected to be in the range [-depth, depth-1].
        Returns:
            the output CTensor.
        """
        values = tensor.to_numpy(tensor.from_raw_tensor(indices))
        rank = len(values.shape)
        depth_range = np.arange(self.depth)
        if self.axis < 0:
            self.axis += (rank + 1)
        ls = values.shape[0:self.axis]
        rs = values.shape[self.axis:rank]
        targets = np.reshape(depth_range, (1,) * len(ls) + depth_range.shape +
                             (1,) * len(rs))
        values = np.reshape(np.mod(values, self.depth), ls + (1,) + rs)
        np_tensor = np.asarray(targets == values, dtype=np.float32)
        np_tensor = np_tensor * (self.values[1] -
                                 self.values[0]) + self.values[0]
        tmp_tensor = tensor.from_numpy(np_tensor)
        tmp_tensor.to_device(indices.device())
        return tmp_tensor.data

    def backward(self, dy):
        """
        backward of OneHot
        Args:
            dy (CTensor):gradient tensor.
        Raises:
            AssertionError: no backward function for this operator
        """
        assert False, ('no gradient for backward function')


def onehot(axis, indices, depth, values):
    """
    Produces a one-hot tensor based on inputs.
    Args:
        axis (int): Axis along which one-hot representation in added.
            Default: axis=-1. axis=-1 means that the additional dimension
            will be inserted as the innermost/last dimension in the output
            tensor.
        indices (Tensor): Scalar specifying the number of classes in
            one-hot tensor. The values in the 'indices' input tensor are
            expected to be in the range [-depth, depth-1].
        depth (int): Scalar specifying the number of classes in one-hot
            tensor. This is also the size of the one-hot dimension
            (specified by 'axis' attribute) added on in the output tensor.
            The values in the 'indices' input tensor are expected to be in
            the range [-depth, depth-1].
        values (float): Rank 1 tensor containing exactly two elements, in
            the format [off_value, on_value], where 'on_value' is the
            value used for filling locations specified in 'indices' input
            tensor,
    Returns:
        the output Tensor.
    """
    return OneHot(axis, depth, values)(indices)[0]


class _RNN(Operator):
    """ RNN operation with c++ backend
    """

    def __init__(self, handle, return_sequences=False):
        assert singa.USE_CUDA, "Not able to run without CUDA"
        super(_RNN, self).__init__()
        self.handle = handle
        self.return_sequences = return_sequences

    def forward(self, x, hx, cx, w):
        if training:
            (y, hy, cy) = singa.GpuRNNForwardTraining(x, hx, cx, w, self.handle)
            self.inputs = {
                'x': x,
                'hx': hx,
                'cx': cx,
                'w': w,
                'y': y,
                'hy': hy,
                'cy': cy
            }
        else:
            (y, hy, cy) = singa.GpuRNNForwardInference(x, hx, cx, w,
                                                       self.handle)

        if self.return_sequences:
            return y
        else:
            last_y_shape = (y.shape()[1], y.shape()[2])
            last_y = singa.Tensor(list(last_y_shape), x.device())

            src_offset = y.Size() - last_y.Size()
            # def copy_data_to_from(dst, src, size, dst_offset=0, src_offset=0):
            singa.CopyDataToFrom(last_y, y, last_y.Size(), 0, src_offset)
            return last_y

    def backward(self, grad):
        assert training is True and hasattr(
            self, "inputs"), "Please set training as True before do BP. "

        dy = None
        if self.return_sequences:
            assert grad.shape() == self.inputs['y'].shape(), (
                "grad shape %s != y shape %s" %
                (grad.shape(), self.inputs['y'].shape()))
            dy = grad
        else:
            assert grad.shape() == (self.inputs['y'].shape()[1],
                                    self.inputs['y'].shape()[2]), (
                                        "grad y shape %s != last y shape %s" %
                                        (grad.shape(),
                                         (self.inputs['y'].shape()[1],
                                          self.inputs['y'].shape()[2])))
            dy = singa.Tensor(list(self.inputs['y'].shape()), grad.device())
            dy.SetFloatValue(0.0)
            # grad shape (bs, directions*hidden)
            # dy shape (seq, bs, directions*hidden)
            dst_offset = dy.Size() - grad.Size()
            singa.CopyDataToFrom(dy, grad, grad.Size(), dst_offset, 0)

        dhy = singa.Tensor(list(self.inputs['hy'].shape()), grad.device())
        dhy.SetFloatValue(0.0)
        dcy = singa.Tensor(list(self.inputs['cy'].shape()), grad.device())
        dcy.SetFloatValue(0.0)

        (dx, dhx, dcx) = singa.GpuRNNBackwardx(self.inputs['y'], dy, dhy, dcy,
                                               self.inputs['w'],
                                               self.inputs['hx'],
                                               self.inputs['cx'], self.handle)
        dW = singa.GpuRNNBackwardW(self.inputs['x'], self.inputs['hx'],
                                   self.inputs['y'], self.handle)

        return dx, dhx, dcx, dW


class CosSim(Operator):
    """
    Init a cos similarity operator
    """

    def __init__(self):
        super(CosSim, self).__init__()

    @classmethod
    def dot(cls, a, b):
        """ 
        dot multiply
        Args:
            a (CTensor): 2d input tensor.
            b (CTensor): 2d input tensor.
        Returns:
            CTensor: the output CTensor.
        """
        batch_size = a.shape()[0]
        ret = []
        for indice in range(batch_size):
            tmp_a = singa.SliceOn(a, indice, indice + 1, 0)  # 1 * d
            tmp_b = singa.SliceOn(b, indice, indice + 1, 0)  # 1 * d
            tmp_b = singa.DefaultTranspose(tmp_b)
            tmp_tensor = singa.Mult(tmp_a, tmp_b)  # 1 * d * d * 1
            ret.append(tmp_tensor)
        ret = singa.VecTensor(ret)
        ret = singa.ConcatOn(ret, 0)  # b * 1
        return singa.Reshape(ret, [ret.shape()[0]])  # b

    def forward(self, a, b):
        """
        forward of CosSim
        Args:
            a (CTensor): input tensor.
            b (CTensor): input tensor.
        Returns:
            the output CTensor.
        """
        ad = CosSim.dot(a, a)
        bd = CosSim.dot(b, b)
        ap = singa.PowFloat(ad, 0.5)
        bp = singa.PowFloat(bd, 0.5)
        ret = singa.__div__(CosSim.dot(a, b), singa.__mul__(ap, bp))
        if training:
            self.cache = (a, b, ad, bd, ap, bp, ret)
        return ret

    def backward(self, dy):
        """
        backward of CosSim
        follow https://math.stackexchange.com/a/1923705
        Args:
            dy (CTensor): gradient tensor.
        Raises:
            the gradient tensor over input tensor.
        """
        a, b, ad, bd, ap, bp, ret = self.cache
        ab = singa.__mul__(ap, bp)
        ab = singa.Reshape(ab, list(ab.shape()) + [1])  # b * 1
        ad = singa.Reshape(ad, list(ad.shape()) + [1])  # b * 1
        bd = singa.Reshape(bd, list(bd.shape()) + [1])  # b * 1
        ret = singa.Reshape(ret, list(ret.shape()) + [1])  # b * 1
        da = singa.__sub__(singa.__div__(b, ab),
                           singa.__mul__(ret, singa.__div__(a, ad)))
        db = singa.__sub__(singa.__div__(a, ab),
                           singa.__mul__(ret, singa.__div__(b, bd)))
        return da, db


def cossim(a, b):
    """
    Produces a cos similarity operator
    Args:
        a (CTensor): input tensor.
        b (CTensor): input tensor.
    Returns:
        the output Tensor.
    """
    return CosSim()(a, b)[0]


''' alias for Operator and Layers
'''
Operation = Operator
''' import layer at the end to resolve circular import
'''
from singa import layer
Linear = layer.Linear
Conv2d = layer.Conv2d
SeparableConv2d = layer.SeparableConv2d
BatchNorm2d = layer.BatchNorm2d
Pooling2d = layer.Pooling2d
MaxPool2d = layer.MaxPool2d
AvgPool2d = layer.AvgPool2d
MaxPool1d = layer.MaxPool1d
AvgPool1d = layer.AvgPool1d
RNN_Base = layer.RNN_Base
RNN = layer.RNN
LSTM = layer.LSTM
