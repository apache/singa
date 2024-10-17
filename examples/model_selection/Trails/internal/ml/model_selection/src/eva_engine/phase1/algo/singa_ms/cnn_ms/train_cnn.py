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

from singa import singa_wrap as singa
from singa import device
from singa import tensor
from singa import opt
from singa import autograd
from singa.opt import Optimizer
from singa.opt import DecayScheduler
from singa.opt import Constant
import numpy as np
import time
import argparse
from PIL import Image

np_dtype = {"float16": np.float16, "float32": np.float32}

singa_dtype = {"float16": tensor.float16, "float32": tensor.float32}


### MSOptimizer
class MSOptimizer(Optimizer):

    def __call__(self, loss):
        pn_p_g_list = self.call_with_returns(loss)
        self.step()
        return pn_p_g_list

    def call_with_returns(self, loss):
        # print ("call_with_returns loss.data: \n", loss.data)
        pn_p_g_list = []
        for p, g in autograd.backward(loss):
            if p.name is None:
                p.name = id(p)
            self.apply(p.name, p, g)
            # print ("call with returns")
            # print ("p.name: \n", p.name)
            # print ("p.data: \n", p.data)
            # print ("g.data: \n", g.data)
            pn_p_g_list.append([p.name, p, g])  # need iterables
        return pn_p_g_list


# MSSGD -- actually no change of code
class MSSGD(MSOptimizer):
    """Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from `On the importance of initialization and momentum in deep learning`__.

    Args:
        lr(float): learning rate
        momentum(float, optional): momentum factor(default: 0)
        weight_decay(float, optional): weight decay(L2 penalty)(default: 0)
        dampening(float, optional): dampening for momentum(default: 0)
        nesterov(bool, optional): enables Nesterov momentum(default: False)

    Typical usage example:
        >> > from singa import opt
        >> > optimizer = opt.SGD(lr=0.1, momentum=0.9)
        >> > optimizer.update()

    __ http: // www.cs.toronto.edu / %7Ehinton / absps / momentum.pdf

    .. note::
        The implementation of SGD with Momentum / Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and: math: `\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self,
                 lr=0.1,
                 momentum=0,
                 dampening=0,
                 weight_decay=0,
                 nesterov=False,
                 dtype=tensor.float32):
        super(MSSGD, self).__init__(lr, dtype)

        # init momentum
        if type(momentum) == float or type(momentum) == int:
            if momentum < 0.0:
                raise ValueError("Invalid momentum value: {}".format(momentum))
            self.momentum = Constant(momentum)
        elif isinstance(momentum, DecayScheduler):
            self.momentum = momentum
            momentum = momentum.init_value
        else:
            raise TypeError("Wrong momentum type")
        self.mom_value = self.momentum(self.step_counter).as_type(self.dtype)

        # init dampening
        if type(dampening) == float or type(dampening) == int:
            self.dampening = Constant(dampening)
        elif isinstance(dampening, DecayScheduler):
            self.dampening = dampening
            dampening = dampening.init_value
        else:
            raise TypeError("Wrong dampening type")
        self.dam_value = self.dampening(self.step_counter).as_type(self.dtype)

        # init weight_decay
        if type(weight_decay) == float or type(weight_decay) == int:
            if weight_decay < 0.0:
                raise ValueError(
                    "Invalid weight_decay value: {}".format(weight_decay))
            self.weight_decay = Constant(weight_decay)
        elif isinstance(weight_decay, DecayScheduler):
            self.weight_decay = weight_decay
        else:
            raise TypeError("Wrong weight_decay type")
        self.decay_value = self.weight_decay(self.step_counter).as_type(
            self.dtype)

        # init other params
        self.nesterov = nesterov
        self.moments = dict()

        # check value
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening")

    def apply(self, param_name, param_value, param_grad):
        """Performs a single optimization step.

        Args:
                param_name(String): the name of the param
                param_value(Tensor): param values to be update in-place
                grad(Tensor): param gradients; the values may be updated
                        in this function; cannot use it anymore
        """
        assert param_value.shape == param_grad.shape, ("shape mismatch",
                                                       param_value.shape,
                                                       param_grad.shape)
        self.device_check(param_value, self.step_counter, self.lr_value,
                          self.mom_value, self.dam_value, self.decay_value)

        # derive dtype from input
        assert param_value.dtype == self.dtype

        # TODO add branch operator
        # if self.decay_value != 0:
        if self.weight_decay.init_value != 0:
            singa.Axpy(self.decay_value.data, param_value.data, param_grad.data)

        if self.momentum.init_value != 0:
            if param_name not in self.moments:
                flag = param_value.device.graph_enabled()
                param_value.device.EnableGraph(False)
                self.moments[param_name] = tensor.zeros_like(param_value)
                param_value.device.EnableGraph(flag)

            buf = self.moments[param_name]
            buf *= self.mom_value
            alpha = 1.0 - self.dam_value
            singa.Axpy(alpha.data, param_grad.data, buf.data)

            if self.nesterov:
                singa.Axpy(self.mom_value.data, buf.data, param_grad.data)
            else:
                param_grad = buf

        minus_lr = 0.0 - self.lr_value
        singa.Axpy(minus_lr.data, param_grad.data, param_value.data)

    def step(self):
        # increment step counter, lr and moment
        super().step()
        mom_value = self.momentum(self.step_counter).as_type(self.dtype)
        dam_value = self.dampening(self.step_counter).as_type(self.dtype)
        decay_value = self.weight_decay(self.step_counter).as_type(self.dtype)
        self.mom_value.copy_from(mom_value)
        self.dam_value.copy_from(dam_value)
        self.decay_value.copy_from(decay_value)

    def get_states(self):
        states = super().get_states()
        if self.mom_value > 0:
            states[
                'moments'] = self.moments  # a dict for 1st order moments tensors
        return states

    def set_states(self, states):
        super().set_states(states)
        if 'moments' in states:
            self.moments = states['moments']
            self.mom_value = self.momentum(self.step_counter)


# Data augmentation
def augmentation(x, batch_size):
    xpad = np.pad(x, [[0, 0], [0, 0], [4, 4], [4, 4]], 'symmetric')
    for data_num in range(0, batch_size):
        offset = np.random.randint(8, size=2)
        x[data_num, :, :, :] = xpad[data_num, :,
                                    offset[0]:offset[0] + x.shape[2],
                                    offset[1]:offset[1] + x.shape[2]]
        if_flip = np.random.randint(2)
        if (if_flip):
            x[data_num, :, :, :] = x[data_num, :, :, ::-1]
    return x


# Calculate accuracy
def accuracy(pred, target):
    # y is network output to be compared with ground truth (int)
    y = np.argmax(pred, axis=1)
    a = y == target
    correct = np.array(a, "int").sum()
    return correct


# Data partition according to the rank
def partition(global_rank, world_size, train_x, train_y, val_x, val_y):
    # Partition training data
    data_per_rank = train_x.shape[0] // world_size
    idx_start = global_rank * data_per_rank
    idx_end = (global_rank + 1) * data_per_rank
    train_x = train_x[idx_start:idx_end]
    train_y = train_y[idx_start:idx_end]

    # Partition evaluation data
    data_per_rank = val_x.shape[0] // world_size
    idx_start = global_rank * data_per_rank
    idx_end = (global_rank + 1) * data_per_rank
    val_x = val_x[idx_start:idx_end]
    val_y = val_y[idx_start:idx_end]
    return train_x, train_y, val_x, val_y


# Function to all reduce NUMPY accuracy and loss from multiple devices
def reduce_variable(variable, dist_opt, reducer):
    reducer.copy_from_numpy(variable)
    dist_opt.all_reduce(reducer.data)
    dist_opt.wait()
    output = tensor.to_numpy(reducer)
    return output


def resize_dataset(x, image_size):
    num_data = x.shape[0]
    dim = x.shape[1]
    X = np.zeros(shape=(num_data, dim, image_size, image_size),
                 dtype=np.float32)
    for n in range(0, num_data):
        for d in range(0, dim):
            X[n, d, :, :] = np.array(Image.fromarray(x[n, d, :, :]).resize(
                (image_size, image_size), Image.BILINEAR),
                                     dtype=np.float32)
    return X


def run(global_rank,
        world_size,
        local_rank,
        max_epoch,
        batch_size,
        model,
        data,
        mssgd,
        graph,
        verbosity,
        dist_option='plain',
        spars=None,
        precision='float32'):
    # dev = device.create_cuda_gpu_on(local_rank)  # need to change to CPU device for CPU-only machines
    dev = device.get_default_device()
    dev.SetRandSeed(0)
    np.random.seed(0)

    if data == 'cifar10':
        from data import cifar10
        train_x, train_y, val_x, val_y = cifar10.load()
    elif data == 'cifar100':
        from data import cifar100
        train_x, train_y, val_x, val_y = cifar100.load()
    elif data == 'mnist':
        from data import mnist
        train_x, train_y, val_x, val_y = mnist.load()

    num_channels = train_x.shape[1]
    image_size = train_x.shape[2]
    data_size = np.prod(train_x.shape[1:train_x.ndim]).item()
    num_classes = (np.max(train_y) + 1).item()

    if model == 'resnet':
        from model import resnet
        model = resnet.resnet50(num_channels=num_channels,
                                num_classes=num_classes)
    elif model == 'xceptionnet':
        from model import xceptionnet
        model = xceptionnet.create_model(num_channels=num_channels,
                                         num_classes=num_classes)
    elif model == 'cnn':
        from model import cnn
        model = cnn.create_model(num_channels=num_channels,
                                 num_classes=num_classes)
    elif model == 'alexnet':
        from model import alexnet
        model = alexnet.create_model(num_channels=num_channels,
                                     num_classes=num_classes)
    elif model == 'mlp':
        import os, sys, inspect
        current = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe())))
        parent = os.path.dirname(current)
        sys.path.insert(0, parent)
        from mlp import model
        model = model.create_model(data_size=data_size, num_classes=num_classes)

    elif model == 'msmlp':
        import os, sys, inspect
        current = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe())))
        parent = os.path.dirname(current)
        sys.path.insert(0, parent)
        from msmlp import model
        model = model.create_model(data_size=data_size, num_classes=num_classes)

    # For distributed training, sequential has better performance
    if hasattr(mssgd, "communicator"):
        DIST = True
        sequential = True
    else:
        DIST = False
        sequential = False

    if DIST:
        train_x, train_y, val_x, val_y = partition(global_rank, world_size,
                                                   train_x, train_y, val_x,
                                                   val_y)

    if model.dimension == 4:
        tx = tensor.Tensor(
            (batch_size, num_channels, model.input_size, model.input_size), dev,
            singa_dtype[precision])
    elif model.dimension == 2:
        tx = tensor.Tensor((batch_size, data_size), dev, singa_dtype[precision])
        np.reshape(train_x, (train_x.shape[0], -1))
        np.reshape(val_x, (val_x.shape[0], -1))

    ty = tensor.Tensor((batch_size,), dev, tensor.int32)
    num_train_batch = train_x.shape[0] // batch_size
    num_val_batch = val_x.shape[0] // batch_size
    idx = np.arange(train_x.shape[0], dtype=np.int32)

    # Attach model to graph
    model.set_optimizer(mssgd)
    model.compile([tx], is_train=True, use_graph=graph, sequential=sequential)
    dev.SetVerbosity(verbosity)

    # Training and evaluation loop
    for epoch in range(max_epoch):
        start_time = time.time()
        np.random.shuffle(idx)

        if global_rank == 0:
            print('Starting Epoch %d:' % (epoch))

        # Training phase
        train_correct = np.zeros(shape=[1], dtype=np.float32)
        test_correct = np.zeros(shape=[1], dtype=np.float32)
        train_loss = np.zeros(shape=[1], dtype=np.float32)

        model.train()
        print("num_train_batch: \n", num_train_batch)
        print()
        for b in range(num_train_batch):
            if b % 200 == 0:
                print("b: \n", b)
            # Generate the patch data in this iteration
            x = train_x[idx[b * batch_size:(b + 1) * batch_size]]
            if model.dimension == 4:
                x = augmentation(x, batch_size)
                if (image_size != model.input_size):
                    x = resize_dataset(x, model.input_size)
            x = x.astype(np_dtype[precision])
            y = train_y[idx[b * batch_size:(b + 1) * batch_size]]

            synflow_flag = False
            # Train the model
            if epoch == (max_epoch - 1) and b == (
                    num_train_batch -
                    1):  ### synflow calcuation for the last batch
                print("last epoch calculate synflow")
                synflow_flag = True
                ### step 1: all one input
                # Copy the patch data into input tensors
                tx.copy_from_numpy(np.ones(x.shape, dtype=np.float32))
                ty.copy_from_numpy(y)
                ### step 2: all weights turned to positive (done)
                ### step 3: new loss (done)
                pn_p_g_list, out, loss = model(tx, ty, dist_option, spars,
                                               synflow_flag)
                ### step 4: calculate the multiplication of weights
                synflow_score = 0.0
                for pn_p_g_item in pn_p_g_list:
                    print("calculate weight param * grad parameter name: \n",
                          pn_p_g_item[0])
                    if len(pn_p_g_item[1].shape
                          ) == 2:  # param_value.data is "weight"
                        print("pn_p_g_item[1].shape: \n", pn_p_g_item[1].shape)
                        synflow_score += np.sum(
                            np.absolute(
                                tensor.to_numpy(pn_p_g_item[1]) *
                                tensor.to_numpy(pn_p_g_item[2])))
                print("synflow_score: \n", synflow_score)
            elif epoch == (max_epoch - 1) and b == (
                    num_train_batch - 2):  # all weights turned to positive
                # Copy the patch data into input tensors
                tx.copy_from_numpy(x)
                ty.copy_from_numpy(y)
                pn_p_g_list, out, loss = model(tx, ty, dist_option, spars,
                                               synflow_flag)
                train_correct += accuracy(tensor.to_numpy(out), y)
                train_loss += tensor.to_numpy(loss)[0]
                # all params turned to positive
                for pn_p_g_item in pn_p_g_list:
                    print("absolute value parameter name: \n", pn_p_g_item[0])
                    pn_p_g_item[1] = tensor.abs(
                        pn_p_g_item[1])  # tensor actually ...
            else:  # normal train steps
                # Copy the patch data into input tensors
                tx.copy_from_numpy(x)
                ty.copy_from_numpy(y)
                # print ("normal before model(tx, ty, synflow_flag, dist_option, spars)")
                # print ("train_cnn tx: \n", tx)
                # print ("train_cnn ty: \n", ty)
                pn_p_g_list, out, loss = model(tx, ty, dist_option, spars,
                                               synflow_flag)
                # print ("normal after model(tx, ty, synflow_flag, dist_option, spars)")
                train_correct += accuracy(tensor.to_numpy(out), y)
                train_loss += tensor.to_numpy(loss)[0]

        if DIST:
            # Reduce the evaluation accuracy and loss from multiple devices
            reducer = tensor.Tensor((1,), dev, tensor.float32)
            train_correct = reduce_variable(train_correct, mssgd, reducer)
            train_loss = reduce_variable(train_loss, mssgd, reducer)

        if global_rank == 0:
            print('Training loss = %f, training accuracy = %f' %
                  (train_loss, train_correct /
                   (num_train_batch * batch_size * world_size)),
                  flush=True)

        # Evaluation phase
        model.eval()
        for b in range(num_val_batch):
            x = val_x[b * batch_size:(b + 1) * batch_size]
            if model.dimension == 4:
                if (image_size != model.input_size):
                    x = resize_dataset(x, model.input_size)
            x = x.astype(np_dtype[precision])
            y = val_y[b * batch_size:(b + 1) * batch_size]
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)
            out_test = model(tx)
            test_correct += accuracy(tensor.to_numpy(out_test), y)

        if DIST:
            # Reduce the evaulation accuracy from multiple devices
            test_correct = reduce_variable(test_correct, mssgd, reducer)

        # Output the evaluation accuracy
        if global_rank == 0:
            print('Evaluation accuracy = %f, Elapsed Time = %fs' %
                  (test_correct / (num_val_batch * batch_size * world_size),
                   time.time() - start_time),
                  flush=True)

    dev.PrintTimeProfiling()


if __name__ == '__main__':
    # Use argparse to get command config: max_epoch, model, data, etc., for single gpu training
    parser = argparse.ArgumentParser(
        description='Training using the autograd and graph.')
    parser.add_argument(
        'model',
        choices=['cnn', 'resnet', 'xceptionnet', 'mlp', 'msmlp', 'alexnet'],
        default='cnn')
    parser.add_argument('data',
                        choices=['mnist', 'cifar10', 'cifar100'],
                        default='mnist')
    parser.add_argument('-p',
                        choices=['float32', 'float16'],
                        default='float32',
                        dest='precision')
    parser.add_argument('-m',
                        '--max-epoch',
                        default=3,
                        type=int,
                        help='maximum epochs',
                        dest='max_epoch')
    parser.add_argument('-b',
                        '--batch-size',
                        default=64,
                        type=int,
                        help='batch size',
                        dest='batch_size')
    parser.add_argument('-l',
                        '--learning-rate',
                        default=0.005,
                        type=float,
                        help='initial learning rate',
                        dest='lr')
    # Determine which gpu to use
    parser.add_argument('-i',
                        '--device-id',
                        default=0,
                        type=int,
                        help='which GPU to use',
                        dest='device_id')
    parser.add_argument('-g',
                        '--disable-graph',
                        default='True',
                        action='store_false',
                        help='disable graph',
                        dest='graph')
    parser.add_argument('-v',
                        '--log-verbosity',
                        default=0,
                        type=int,
                        help='logging verbosity',
                        dest='verbosity')

    args = parser.parse_args()

    mssgd = MSSGD(lr=args.lr,
                  momentum=0.9,
                  weight_decay=1e-5,
                  dtype=singa_dtype[args.precision])
    run(0,
        1,
        args.device_id,
        args.max_epoch,
        args.batch_size,
        args.model,
        args.data,
        mssgd,
        args.graph,
        args.verbosity,
        precision=args.precision)
