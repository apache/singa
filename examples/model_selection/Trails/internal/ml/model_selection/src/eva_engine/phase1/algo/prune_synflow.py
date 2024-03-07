#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from src.eva_engine.phase1.algo.alg_base import Evaluator
from src.common.constant import Config

from singa import singa_wrap as singa
from singa import device as singa_device
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
from numpy import linalg as LA

np_dtype = {"float16": np.float16, "float32": np.float32}

# singa_dtype = {"float16": tensor.float16, "float32": tensor.float32}
singa_dtype = {"float32": tensor.float32}


### MSOptimizer
class MSOptimizer(Optimizer):
    def __call__(self, loss):
        pn_p_g_list = self.call_with_returns(loss)
        # print ("optimizer1 before self.step()")
        # print ("optimizer1 before print len(pn_p_g_list): \n", len(pn_p_g_list))
        self.step()
        # print ("optimizer1 after print len(pn_p_g_list): \n", len(pn_p_g_list))
        # print ("optimizer1 after self.step()")
        return pn_p_g_list

    def call_with_returns(self, loss):
        # print ("call_with_returns before apply loss.data: \n", loss.data)
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
        # print ("call_with_returns after apply loss.data: \n", loss.data)
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
        super(MSSGD, self).__init__(lr)

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
        # self.dtype = dtype
        # self.mom_value = self.momentum(self.step_counter).as_type(self.dtype)
        self.mom_value = self.momentum(self.step_counter)

        # init dampening
        if type(dampening) == float or type(dampening) == int:
            self.dampening = Constant(dampening)
        elif isinstance(dampening, DecayScheduler):
            self.dampening = dampening
            dampening = dampening.init_value
        else:
            raise TypeError("Wrong dampening type")
        # self.dam_value = self.dampening(self.step_counter).as_type(self.dtype)
        self.dam_value = self.dampening(self.step_counter)

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
        # self.decay_value = self.weight_decay(self.step_counter).as_type(self.dtype)
        self.decay_value = self.weight_decay(self.step_counter)

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
        # assert param_value.dtype == self.dtype

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
        # print ("before super step")
        super().step()
        # print ("after super step")
        # print ("before custiomized step")
        # mom_value = self.momentum(self.step_counter).as_type(self.dtype)
        # dam_value = self.dampening(self.step_counter).as_type(self.dtype)
        # decay_value = self.weight_decay(self.step_counter).as_type(self.dtype)
        mom_value = self.momentum(self.step_counter)
        dam_value = self.dampening(self.step_counter)
        decay_value = self.weight_decay(self.step_counter)
        self.mom_value.copy_from(mom_value)
        self.dam_value.copy_from(dam_value)
        self.decay_value.copy_from(decay_value)
        # print ("after customized step")

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


import torch


class SynFlowEvaluator(Evaluator):

    def __init__(self):
        super().__init__()

    def evaluate(self, arch, device, batch_data: object, batch_labels: torch.Tensor, space_name: str) -> float:
        """
        This is implementation of paper
        "Pruning neural networks without any data by iteratively conserving synaptic flow"
        The score takes 5 steps:
            1. For each layer, for each parameter, calculate the absolute value |0|
            2. Use a single all-one-vector with dim = [1, c, h, w] to run a forward,
               Since only consider linear and Con2d operation, the forward output is multiple( [ |0l| for l in L] )
            3. New loss function R = sum(output), and then run backward
            4. for each layer, calculate Sl = Hadamard product( df/dw, w), where Sij=aij×bij
            5. score = sum( [ Sl for l in layers ] )
        Comments:
            1. this is data-Agnostic
            2. only compute on a single example
        """

        ### singa configs
        mssgd = MSSGD(lr=0.005, momentum=0.9, weight_decay=1e-5, dtype=singa_dtype['float32'])
        device_id = 0
        max_epoch = 1
        model = arch
        graph = True
        verbosity = 0
        dist_option = 'plain'
        spars = None
        precision = 'float32'
        global_rank = 0
        world_size = 1

        ### singa setups
        # print ("device: \n", device)
        if device == 'cpu':
            dev = singa_device.get_default_device()
        else:  # GPU
            dev = singa_device.create_cuda_gpu_on(local_rank)  # need to change to CPU device for CPU-only machines
        dev.SetRandSeed(0)
        np.random.seed(0)

        # For distributed training, sequential has better performance
        if hasattr(mssgd, "communicator"):
            DIST = True
            sequential = True
        else:
            DIST = False
            sequential = False

        model.train()

        ### process batch_data
        x = batch_data.cpu().numpy()  # Size([1, 100]) and all ones
        x = x.astype(np_dtype[precision])
        y = np.ones(x.shape[0], dtype=np.int32)
        if model.dimension == 2:  # input data dimension
            tx = tensor.Tensor(x.shape, dev, singa_dtype[precision])
        ty = tensor.Tensor((x.shape[0],), dev, tensor.int32)

        model.set_optimizer(mssgd)
        model.compile([tx], is_train=True, use_graph=graph, sequential=sequential)
        dev.SetVerbosity(verbosity)

        # 1. Convert params to their abs.
        synflow_flag = True  ### just change the model to the absolute value
        tx.copy_from_numpy(x)  # dtype=np.float32
        ty.copy_from_numpy(y)
        # print ("before model forward ...")
        pn_p_g_list, out, loss = model(tx, ty, dist_option, spars, synflow_flag)
        # print ("---------------------------------------")
        # print ("before absolute prune_synflow !!!nemb input vector!!! tensor.to_numpy(loss)[0]: ", tensor.to_numpy(loss)[0])
        # print ("before absolute prune_synflow !!!nemb input vector!!! tensor.to_numpy(loss): ", tensor.to_numpy(loss))
        # train_correct += accuracy(tensor.to_numpy(out), y)
        # train_loss += tensor.to_numpy(loss)[0]
        # all params turned to positive
        for pn_p_g_item in pn_p_g_list:
            # print ("absolute value parameter name: \n", pn_p_g_item[0])
            param_np = tensor.to_numpy(pn_p_g_item[1])
            # print ("param_np shape: \n", param_np.shape)
            # print ("param_np sqrt norm: \n", np.sqrt(LA.norm(param_np)/param_np.size))
            # print ("before abs np.min(tensor.to_numpy(pn_p_g_item[1])): \n", np.min(tensor.to_numpy(pn_p_g_item[1])))
            pn_p_g_item[1] = tensor.abs(pn_p_g_item[1])  # tensor actually ..
            # print ("after abs np.min(tensor.to_numpy(pn_p_g_item[1])): \n", np.min(tensor.to_numpy(pn_p_g_item[1])))
            # print ("after abs pn_p_g_item[1][0]: \n", pn_p_g_item[1][0])

        # 2. Compute gradients with input of one dummy example ( 1-vector with dimension [1, c, h, w] )
        # 3.R = sum(output)
        # 4. Select the gradients that we want to use for search/prune
        # 5. Sum over all parameter's results to get the final score.
        # score = sum([grad.sum() for grad in grads_abs])

        # print ("calculate synflow")
        synflow_flag = True
        ### step 1: all one input
        # Copy the patch data into input tensors
        # tx.copy_from_numpy(np.ones(x.shape, dtype=np.float32))
        tx.copy_from_numpy(x)  # dtype=np.float32 # actually it is all ones ... --> np.ones(x.shape, dtype=np.float32)
        ty.copy_from_numpy(y)
        ### step 2: all weights turned to positive (done)
        ### step 3: new loss (done)
        # print ("before model forward ...")
        pn_p_g_list, out, loss = model(tx, ty, dist_option, spars, synflow_flag)
        # print ("prune_synflow !!!nemb input vector!!! synflow step tensor.to_numpy(loss)[0]: ", tensor.to_numpy(loss)[0])
        ### step 4: calculate the multiplication of weights
        score = 0.0
        for pn_p_g_item in pn_p_g_list:
            # print ("calculate weight param * grad parameter name: \n", pn_p_g_item[0])
            if len(pn_p_g_item[1].shape) == 2:  # param_value.data is "weight"
                # print ("pn_p_g_item[1].shape: \n", pn_p_g_item[1].shape)
                # print ("tensor.to_numpy(pn_p_g_item[1][0]): ", tensor.to_numpy(pn_p_g_item[1][0]))
                # print ("calculate synflow parameter name: \n", pn_p_g_item[0])
                # print ("should be positive np.min(tensor.to_numpy(pn_p_g_item[1])): ", np.min(tensor.to_numpy(pn_p_g_item[1])))
                # print ("weight should be positive tensor.to_numpy(pn_p_g_item[1][0])[0, :10]: ", tensor.to_numpy(pn_p_g_item[1][0])[0, :10])
                # print ("gradients tensor.to_numpy(pn_p_g_item[2][0])[0, :10]: ", tensor.to_numpy(pn_p_g_item[2][0])[0, :10])
                # print ()
                score += np.sum(np.absolute(tensor.to_numpy(pn_p_g_item[1]) * tensor.to_numpy(pn_p_g_item[2])))
        # print ("layer_hidden_list: \n", layer_hidden_list)
        # print ("prune_synflow !!!one-hot input vector!!! absolute step tensor.to_numpy(loss)[0]: ", tensor.to_numpy(loss)[0])
        print("score: \n", score)

        return score
