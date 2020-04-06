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
'''This module includes a set of optimizers for updating model parameters.
It replaces the old optimizers from optimizer.py'''

from singa import tensor
from singa import autograd
from . import singa_wrap as singa


class Optimizer(object):
    """Base optimizer.

    Args:
        config (Dict): specify the default values of configurable variables.
    """

    def __init__(self, config):
        self.default_config = config
        self.iter = 0
        self.param2config = {}
        self.param2state = {}

    def update(self, param, grad):
        """Update the param values with given gradients.

        Args:
            param(Tensor): param values to be updated in-place
            grad(Tensor): param gradients; the values may be updated
                    in this function; do not use it anymore
        """
        pass

    def step(self):
        """To increment the step counter"""
        self.iter += 1

    def register(self, param_group, config):
        for param in param_group:
            assert param not in self.param2config, 'param is already registered'

            self.param2config[param] = config

    def load(self):
        pass

    def save(self):
        pass


class SGD(Optimizer):
    """Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

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
                 nesterov=False):
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr,
                        momentum=momentum,
                        dampening=dampening,
                        weight_decay=weight_decay,
                        nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(defaults)

    def update(self, param, grad):
        """Performs a single optimization step.

        Args:
                param(Tensor): param values to be update in-place
                grad(Tensor): param gradients; the values may be updated
                        in this function; cannot use it anymore
        """
        assert param.shape == grad.shape, ("shape mismatch", param.shape,
                                           grad.shape)
        group = self.default_config
        if param in self.param2config:
            group = self.param2config[param]
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']

        if weight_decay != 0:
            singa.Axpy(weight_decay, param.data, grad.data)
        if momentum != 0:
            if param not in self.param2state:
                self.param2state[param] = {}
            param_state = self.param2state[param]
            if 'momentum_buffer' not in param_state:
                flag = param.device.graph_enabled()
                param.device.EnableGraph(False)
                buf = param_state['momentum_buffer'] = tensor.zeros_like(param)
                param.device.EnableGraph(flag)

                buf *= momentum
                singa.Axpy(1.0, grad.data, buf.data)
            else:
                buf = param_state['momentum_buffer']
                buf *= momentum
                singa.Axpy(1.0 - dampening, grad.data, buf.data)
            if nesterov:
                singa.Axpy(momentum, buf.data, grad.data)
            else:
                grad = buf
        singa.Axpy(-group['lr'], grad.data, param.data)

    def backward_and_update(self, loss):
        """Performs backward propagation from the loss and parameter update.

        From the loss, it performs backward propagation to get the gradients
        and do the parameter update.

        Args:
                loss(Tensor): loss is the objective function of the deep learning model
                optimization, e.g. for classification problem it can be the output of the
                softmax_cross_entropy function.
        """
        for p, g in autograd.backward(loss):
            self.update(p, g)


class DistOpt(object):
    """The class is designed to wrap an optimizer to do distributed training.

    This class is used to wrap an optimizer object to perform distributed training based
    on multiprocessing. Each process has an individual rank, which gives information of
    which GPU the individual process is using. The training data is partitioned, so that
    each process can evaluate the sub-gradient based on the partitioned training data.
    Once the sub-graident is calculated on each processes, the overall stochastic gradient
    is obtained by all-reducing the sub-gradients evaluated by all processes. The all-reduce
    operation is supported by the NVidia Collective Communication Library (NCCL).

    Args:
        opt(Optimizer): The optimizer to be wrapped.
        nccl_id(NcclIdHolder): an nccl id holder object for a unique communication id
        local_rank(int): local rank of a process on the current node
        world_size(int): total number of processes
        buffSize(int): the buffSize in terms of number of elements used in nccl communicator

    Attributes:
        world_size(int): total number of processes
        local_rank(int): local rank of a process on the current node
        global_rank(int): global rank of a process

    Typical usage example:
        >> > from singa import opt
        >> > optimizer = opt.SGD(lr=0.1, momentum=0.9)
        >> > optimizer = opt.DistOpt(sgd)

    """

    def __init__(self,
                 opt=SGD(),
                 nccl_id=None,
                 local_rank=None,
                 world_size=None,
                 buffSize=4194304):
        self.opt = opt
        if nccl_id is None:
            # constructure for application using MPI
            self.communicator = singa.Communicator(buffSize)
        else:
            # constructor for application using python multi-process module
            self.communicator = singa.Communicator(local_rank, world_size, nccl_id,
                                                   buffSize)

        self.world_size = self.communicator.world_size
        self.local_rank = self.communicator.local_rank
        self.global_rank = self.communicator.global_rank

    def update(self, param, grad):
        """Performs a single optimization step.

        Args:
                param(Tensor): param values to be update
                grad(Tensor): param gradients
        """
        grad /= self.world_size
        self.opt.update(param, grad)

    def all_reduce(self, tensor):
        """Performs all reduce of a tensor for distributed training.

        Args:
                tensor(Tensor): a tensor to be all-reduced
        """
        self.communicator.synch(tensor)

    def fused_all_reduce(self, tensor, send=True):
        """Performs all reduce of the tensors after fusing them in a buffer.

        Args:
                tensor(List of Tensors): a list of tensors to be all-reduced
                send(bool): When send is False, the tensor won't be send to the
                target device immediately, it will be copied to the buffer first
        """
        tensor = singa.VecTensor(tensor)
        self.communicator.fusedSynch(tensor, send)

    def all_reduce_half(self, tensor):
        """Performs all reduce of a tensor after converting to FP16.

        Args:
                tensor(Tensor): a tensor to be all-reduced
        """
        self.communicator.synchHalf(tensor)

    def fused_all_reduce_half(self, tensor, send=True):
        """Performs all reduce of the tensors after fusing and converting them to FP16.

        Args:
                tensor(List of Tensors): a list of tensors to be all-reduced
                send(bool): When send is False, the tensor won't be send to the
                target device immediately, it will be copied to the buffer first
        """
        tensor = singa.VecTensor(tensor)
        self.communicator.fusedSynchHalf(tensor, send)

    def sparsification(self, tensor, accumulation, spars, topK):
        """Performs all reduce of a tensor after sparsification.

        Args:
                tensor(Tensor): a tensor to be all-reduced
                accumulation(Tensor): local gradient accumulation
                spars(float): a parameter to control sparsity as defined below
                topK(bool): When topK is False, it sparsifies the gradient with absolute
                value >= sparsWhen topK is True, it sparsifies a fraction of total gradient
                number equals to spars,  E.g. when spars = 0.01, it sparsifies 1 % of the
                total gradient elements
        """
        if accumulation is None:
            self.communicator.sparsification(tensor, spars, topK)
        else:
            self.communicator.sparsification(tensor, accumulation, spars, topK)

    def fused_sparsification(self, tensor, accumulation, spars, topK):
        """Performs all reduce of the tensors after fusing and sparsification.

        Args:
                tensor(List of Tensors): a list of tensors to be all-reduced
                accumulation(Tensor): local gradient accumulation
                spars(float): a parameter to control sparsity as defined below
                topK(bool): When topK is False, it sparsifies the gradient with absolute
                value >= sparsWhen topK is True, it sparsifies a fraction of total gradient
                number equals to spars,  E.g. when spars = 0.01, it sparsifies 1 % of the
                total gradient elements
        """
        tensor = singa.VecTensor(tensor)
        if accumulation is None:
            self.communicator.fusedSparsification(tensor, spars, topK)
        else:
            self.communicator.fusedSparsification(tensor, accumulation, spars,
                                                  topK)

    def wait(self):
        """Wait for the cuda streams used by the communicator to finish their operations."""
        self.communicator.wait()

    def backward_and_update(self, loss, threshold=2097152):
        """Performs backward propagation from the loss and parameter update.

        From the loss, it performs backward propagation to get the gradients and do the parameter
        update. For gradient communication, it fuses all the tensor smaller than the threshold
        value to reduce network latency.

        Args:
                loss(Tensor): loss is the objective function of the deep learning model
                optimization, e.g. for classification problem it can be the output of the
                softmax_cross_entropy function.
                threshold(int): threshold is a parameter to control performance in fusing
                the tensors. For the tensors of sizes smaller than threshold, they are to
                be accumulated and fused before the all reduce operation. For the tensors
                of its size larger than the threshold value, they are to be reduced directly
                without fusion.
        """
        plist = []
        acc = 0
        glist = []
        for p, g in autograd.backward(loss):
            if g.size() > threshold:
                # larger than threshold -> reduced directly
                self.all_reduce(g.data)
            else:
                # smaller than threshold -> accumulate
                glist.append(g.data)
                self.fused_all_reduce([g.data], send=False)
                acc += g.size()
                if (acc > threshold):
                    self.fused_all_reduce(glist)
                    acc = 0
                    glist = []
            plist.append((p, g))
        if glist:
            self.fused_all_reduce(glist)
        self.wait()
        for p, g in plist:
            self.update(p, g)

    def backward_and_update_half(self,
                                 loss,
                                 threshold=2097152,
                                 clipping=False,
                                 clip_Value=100):
        """Performs backward propagation and parameter update, with FP16 precision communication.

        THIS IS A EXPERIMENTAL FUNCTION FOR RESEARCH PURPOSE:
        From the loss, it performs backward propagation to get the gradients and do the parameter
        update. For gradient communication, it fuses all the tensor smaller than the threshold value
        to reduce network latency, as well as converting them to FP16 half precision format before
        sending them out. To assist training, this functions provide an option to perform gradient
        clipping.

        Args:
                loss(Tensor): loss is the objective function of the deep learning model
                optimization, e.g. for classification problem it can be the output of the
                softmax_cross_entropy function.
                threshold(int): threshold is a parameter to control performance in fusing
                the tensors. For the tensors of sizes smaller than threshold, they are to
                be accumulated and fused before the all reduce operation. For the tensors
                of its size larger than the threshold value, they are to be reduced directly
                without fusion.
                clipping(bool): a boolean flag to choose whether to clip the gradient value
                clip_value(float): the clip value to be used when clipping is True
        """
        plist = []
        acc = 0
        glist = []
        for p, g in autograd.backward(loss):
            if clipping:
                g = autograd.clip(g, -clip_Value, clip_Value)
            if g.size() > threshold:
                # larger than threshold -> reduced directly
                self.all_reduce_half(g.data)
            else:
                # smaller than threshold -> accumulate
                glist.append(g.data)
                self.fused_all_reduce_half([g.data], send=False)
                acc += g.size()
                if (acc > threshold):
                    self.fused_all_reduce_half(glist)
                    acc = 0
                    glist = []
            plist.append((p, g))
        if glist:
            self.fused_all_reduce_half(glist)
        self.wait()
        for p, g in plist:
            self.update(p, g)

    def backward_and_partial_update(self, loss, threshold=2097152):
        """Performs backward propagation from the loss and parameter update using asychronous training.

        THIS IS A EXPERIMENTAL FUNCTION FOR RESEARCH PURPOSE:
        From the loss, it performs backward propagation to get the gradients and do the parameter
        update. It fuses the tensors smaller than the threshold value to reduce network latency,
        as well as performing asychronous training where one parameter partition is all-reduced
        per iteration. The size of the parameter partition depends on the threshold value.

        Args:
                loss(Tensor): loss is the objective function of the deep learning model
                optimization, e.g. for classification problem it can be the output of the
                softmax_cross_entropy function.
                threshold(int): threshold is a parameter to control performance in fusing
                the tensors. For the tensors of sizes smaller than threshold, they are to
                be accumulated and fused before the all reduce operation. For the tensors
                of its size larger than the threshold value, they are to be reduced directly
                without fusion.

        Attributes:
                self.partial(int): A counter to determine which partition to perform all-reduce.
                This counter resets to zero automatlly after an update cycle of the full parameter
                set.
        """
        if not hasattr(self, "partial"):
            self.partial = 0
        self.partial += 1
        k = 0
        plist = []
        acc = 0
        tenlist = []
        reduced = []
        for p, g in autograd.backward(loss):
            # every parameters update locally
            self.opt.update(p, g)
            # then do the partial parameter sychronization
            if p.size() > threshold:
                # larger than threshold -> reduced directly
                # k is the partition number of the full gradient set
                k += 1
                if (k == self.partial):
                    self.all_reduce(p.data)
                    reduced.append(p)
            else:
                # smaller than threshold -> accumulate
                plist.append(p.data)
                tenlist.append(p)
                acc += p.size()
                if (acc > threshold):
                    k += 1
                    if (k == self.partial):
                        self.fused_all_reduce(plist, send=False)
                        self.fused_all_reduce(plist)
                        reduced = tenlist
                    acc = 0
                    plist = []
                    tenlist = []
        if plist:
            k += 1
            if (k == self.partial):
                self.fused_all_reduce(plist, send=False)
                self.fused_all_reduce(plist)
                reduced = tenlist
        self.wait()
        # the all-reduced parameters needed to be averaged
        for r in reduced:
            r /= self.world_size
        # the counter returns to zero after a cycle of partial update
        if (k == self.partial):
            self.partial = 0

    def backward_and_sparse_update(self,
                                   loss,
                                   threshold=2097152,
                                   spars=0.05,
                                   topK=False,
                                   corr=True):
        """ Performs backward propagation from the loss and parameter update with sparsification.

        THIS IS A EXPERIMENTAL FUNCTION FOR RESEARCH PURPOSE:
        From the loss, it performs backward propagation to get the gradients and do the parameter
        update. It fuses the tensors with size smaller than the threshold value to reduce network
        latency, as well as using sparsification schemes to transfer only the gradient elements which
        are significant.

        Args:
                loss(Tensor): loss is the objective function of the deep learning model
                optimization, e.g. for classification problem it can be the output of the
                softmax_cross_entropy function.
                threshold(int): threshold is a parameter to control performance in fusing
                the tensors. For the tensors of sizes smaller than threshold, they are to
                be accumulated and fused before the all reduce operation. For the tensors
                of its size larger than the threshold value, they are to be reduced directly
                without fusion.
                spars(float): a parameter to control sparsity as defined below
                topK(bool): When topK is False, it sparsifies the gradient with absolute
                value >= sparsWhen topK is True, it sparsifies a fraction of total gradient
                number equals to spars,  E.g. when spars = 0.01, it sparsifies 1 % of the
                total gradient elements
                corr(bool): whether to use the local accumulate gradient for correction

        Attributes:
                self.sparsInit: A counter to determine which partition to perform all-reduce.
                self.gradAccumulation: Local gradient accumulation
        """
        if ((not hasattr(self, "sparsInit")) and corr):
            self.gradAccumulation = []
            self.sparsInit = False
        plist = []
        acc = 0
        k = -1
        glist = []
        for p, g in autograd.backward(loss):
            if g.size() > threshold:
                # larger than threshold -> reduced directly
                k += 1
                if (corr and (not self.sparsInit)):
                    # create a tensor for the gradient accumulation
                    flag = p.device.graph_enabled()
                    p.device.EnableGraph(False)
                    self.gradAccumulation.append(
                        tensor.Tensor((g.size(),), p.device, p.dtype))
                    self.gradAccumulation[k].set_value(0.0)
                    p.device.EnableGraph(flag)
                if corr:
                    self.sparsification(g.data, self.gradAccumulation[k].data,
                                        spars, topK)
                else:
                    self.sparsification(g.data, None, spars, topK)
            else:
                # smaller than threshold -> accumulate
                glist.append(g.data)
                acc += g.size()
                if (acc > threshold):
                    k += 1
                    if (corr and (not self.sparsInit)):
                        # create a tensor for the gradient accumulation
                        flag = p.device.graph_enabled()
                        p.device.EnableGraph(False)
                        self.gradAccumulation.append(
                            tensor.Tensor((acc,), p.device, p.dtype))
                        self.gradAccumulation[k].set_value(0.0)
                        p.device.EnableGraph(flag)
                    if corr:
                        self.fused_sparsification(glist,
                                                  self.gradAccumulation[k].data,
                                                  spars, topK)
                    else:
                        self.fused_sparsification(glist, None, spars, topK)
                    acc = 0
                    glist = []
            plist.append((p, g))
        if glist:
            k += 1
            if (corr and (not self.sparsInit)):
                # create a tensor for the gradient accumulation
                flag = p.device.graph_enabled()
                p.device.EnableGraph(False)
                self.gradAccumulation.append(
                    tensor.Tensor((acc,), p.device, p.dtype))
                self.gradAccumulation[k].set_value(0.0)
                p.device.EnableGraph(flag)
            if corr:
                self.fused_sparsification(glist, self.gradAccumulation[k].data,
                                          spars, topK)
            else:
                self.fused_sparsification(glist, None, spars, topK)
        self.wait()
        for p, g in plist:
            self.update(p, g)
        self.sparsInit = True
