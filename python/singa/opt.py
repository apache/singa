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
from singa.tensor import Tensor
from singa import autograd
from . import singa_wrap as singa

from deprecated import deprecated


class DecayScheduler:
    # to be used for decaying learning rate or regularization coefficient or momentum, etc.
    def __init__(self, init_value):
        self.init_value = init_value

    def __call__(self, step):
        assert isinstance(step, Tensor)
        return self.call(step)

    def call(self, step) -> Tensor:
        # step is a Tensor with a single scalar value
        # return the current value as a Tensor
        raise NotImplementedError


class Constant(DecayScheduler):

    def call(self, step: Tensor) -> Tensor:
        # TODO should be an in-place operator
        ret = Tensor((1,), step.device)
        ret.set_value(self.init_value)
        return ret


class ExponentialDecay(DecayScheduler):

    def __init__(self, init_value, decay_steps, decay_rate, staircase=False):
        super(ExponentialDecay, self).__init__(init_value)

        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    def call(self, step):
        if self.staircase:
            s = step // self.decay_steps
        else:
            s = step / self.decay_steps
        ret = Tensor((1,), s.device)
        ret.set_value(self.decay_rate)
        return self.init_value * tensor.pow(ret, s)


class Optimizer(object):
    """Base optimizer.

    Args:
        config (Dict): specify the default values of configurable variables.
    """

    def __init__(self, lr, dtype=tensor.float32):
        # init lr(could be a constant scalar or a learning rate scheduler)
        if type(lr) == float or type(lr) == int:
            self.lr = Constant(lr)
        elif isinstance(lr, DecayScheduler):
            self.lr = lr
        else:
            raise TypeError("Wrong learning rate type")

        # init step counter
        self.dtype = dtype
        # TODO change type to int32
        self.step_counter = Tensor((1,), dtype=tensor.float32)
        self.step_counter.set_value(0)
        self.lr_value = self.lr(self.step_counter)

    def get_states(self):
        # skip DecayScheduler as it does not have persistent states
        return {'step_counter': tensor.to_numpy(self.step_counter)[0]}

    def set_states(self, states):
        self.step_counter = Tensor((1,))
        self.step_counter.set_value(states['step_counter'])
        self.lr_value = self.lr(self.step_counter)

    def __call__(self, loss):
        self.call(loss)
        self.step()

    def call(self, loss):
        for p, g in autograd.backward(loss):
            if p.name is None:
                p.name = id(p)
            self.apply(p.name, p, g)

    def step(self):
        """To increment the step counter and update the lr"""
        self.step_counter.data += 1
        lr_value = self.lr(self.step_counter)
        self.lr_value.copy_from(lr_value)

    def apply(self, param_name, param_value, param_grad):
        """Performs a single optimization step.

        Args:
                param_name(String): the name of the param
                param_value(Tensor): param values to be update in-place
                grad(Tensor): param gradients; the values may be updated
                        in this function; cannot use it anymore
        """
        raise NotImplementedError

    @deprecated(
        reason=
        "Update is deprecated, use apply() to do update, refer to apply for more details."
    )
    def update(self, param, grad):
        """Update the param values with given gradients.

        Args:
            param(Tensor): param values to be updated in-place
            grad(Tensor): param gradients; the values may be updated
                    in this function; do not use it anymore
        """
        if param.name is None:
            param.name = id(param)
        self.apply(param.name, param, grad)

    def device_check(self, *inputs):
        flag = inputs[0].device.graph_enabled()
        inputs[0].device.EnableGraph(False)
        x_device = inputs[0].device
        x_dev_id = x_device.id()
        for var in inputs:
            if var.device.id() != x_dev_id:
                var.to_device(x_device)
        inputs[0].device.EnableGraph(flag)

    @deprecated(
        reason=
        "backward_and_update is deprecated, use __call__() to do update, refer to __call__ for more details."
    )
    def backward_and_update(self, loss):
        """Performs backward propagation from the loss and parameter update.

        From the loss, it performs backward propagation to get the gradients
        and do the parameter update.

        Args:
                loss(Tensor): loss is the objective function of the deep learning model
                optimization, e.g. for classification problem it can be the output of the
                softmax_cross_entropy function.
        """
        self.__call__(loss)


class SGD(Optimizer):
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
        super(SGD, self).__init__(lr, dtype)

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


class RMSProp(Optimizer):
    '''RMSProp optimizer.

    See the base Optimizer for all constructor args.

    Args:
        rho (float): float within [0, 1]
        epsilon (float): small value for preventing numeric error
    '''

    def __init__(self, lr=0.1, rho=0.9, epsilon=1e-8, weight_decay=0):
        super(RMSProp, self).__init__(lr)

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
        self.decay_value = self.weight_decay(self.step_counter)

        # init rho
        if type(rho) == float or type(rho) == int:
            self.rho = Constant(rho)
        elif isinstance(rho, DecayScheduler):
            self.rho = rho
        else:
            raise TypeError("Wrong rho type")
        self.rho_value = self.rho(self.step_counter)

        # init epsilon
        if type(epsilon) == float or type(epsilon) == int:
            self.epsilon = Constant(epsilon)
        elif isinstance(rho, DecayScheduler):
            self.epsilon = epsilon
        else:
            raise TypeError("Wrong epsilon type")
        self.epsilon_value = self.epsilon(self.step_counter)

        # init running average
        self.running_average = dict()

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
                          self.rho_value, self.epsilon_value, self.decay_value)

        # if self.decay_value != 0:
        if self.weight_decay.init_value != 0:
            singa.Axpy(self.decay_value.data, param_value.data, param_grad.data)

        if param_name not in self.running_average:
            flag = param_value.device.graph_enabled()
            param_value.device.EnableGraph(False)
            self.running_average[param_name] = tensor.zeros_like(param_value)
            param_value.device.EnableGraph(flag)

        # running_average = running_average * rho + param_grad * param_grad * (1 - rho)
        # param_value = param_value - lr * param_grad / sqrt(running_average + epsilon)

        self.running_average[param_name] *= self.rho_value

        tmp1 = singa.Square(param_grad.data)
        tmp2 = 1.0 - self.rho_value
        singa.Axpy(tmp2.data, tmp1, self.running_average[param_name].data)

        minus_lr = 0.0 - self.lr_value
        tmp3 = self.running_average[param_name] + self.epsilon_value
        tmp3 = singa.Sqrt(tmp3.data)
        tmp3 = singa.__div__(param_grad.data, tmp3)

        singa.Axpy(minus_lr.data, tmp3, param_value.data)

    def step(self):
        # increment step counter, lr and moment
        super().step()
        decay_value = self.weight_decay(self.step_counter)
        rho_value = self.rho(self.step_counter)
        epsilon_value = self.epsilon(self.step_counter)
        self.decay_value.copy_from(decay_value)
        self.rho_value.copy_from(rho_value)
        self.epsilon_value.copy_from(epsilon_value)

    def get_states(self):
        states = super().get_states()
        states['running_average'] = self.running_average
        return states

    def set_states(self, states):
        super().set_states(states)
        if 'running_average' in states:
            self.running_average = states['running_average']


class AdaGrad(Optimizer):
    '''AdaGrad optimizer.

    See the base Optimizer for all constructor args.

    Args:
        epsilon (float): small number for preventing numeric error.
    '''

    def __init__(self, lr=0.1, epsilon=1e-8, weight_decay=0):
        super(AdaGrad, self).__init__(lr)

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
        self.decay_value = self.weight_decay(self.step_counter)

        # init epsilon
        if type(epsilon) == float or type(epsilon) == int:
            self.epsilon = Constant(epsilon)
        elif isinstance(epsilon, DecayScheduler):
            self.epsilon = epsilon
        else:
            raise TypeError("Wrong epsilon type")
        self.epsilon_value = self.epsilon(self.step_counter)

        # init history
        self.history = dict()

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
                          self.epsilon_value, self.decay_value)

        # if self.decay_value != 0:
        if self.weight_decay.init_value != 0:
            singa.Axpy(self.decay_value.data, param_value.data, param_grad.data)

        if param_name not in self.history:
            flag = param_value.device.graph_enabled()
            param_value.device.EnableGraph(False)
            self.history[param_name] = tensor.zeros_like(param_value)
            param_value.device.EnableGraph(flag)

        # history = history + param_grad * param_grad
        # param_value = param_value - lr * param_grad / sqrt(history + epsilon)

        tmp = self.history[param_name].data
        tmp += singa.Square(param_grad.data)

        minus_lr = 0.0 - self.lr_value
        tmp = self.history[param_name] + self.epsilon_value
        tmp = singa.Sqrt(tmp.data)
        tmp = singa.__div__(param_grad.data, tmp)
        singa.Axpy(minus_lr.data, tmp, param_value.data)

    def step(self):
        # increment step counter, lr and moment
        super().step()
        decay_value = self.weight_decay(self.step_counter)
        epsilon_value = self.epsilon(self.step_counter)
        self.decay_value.copy_from(decay_value)
        self.epsilon_value.copy_from(epsilon_value)

    def get_states(self):
        states = super().get_states()
        states['history'] = self.history  # a dict for 1st order moments tensors
        return states

    def set_states(self, states):
        super().set_states(states)
        if 'history' in states:
            self.history = states['history']


class Adam(Optimizer):
    '''Adam optimizer.

    See the base Optimizer for all constructor args.

    Args:
        beta_1(float): coefficient of momentum
        beta_2(float): coefficient of aggregated squared gradient
        epsilon (float): small value for preventing numeric error
    '''

    def __init__(self,
                 lr=0.1,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-8,
                 weight_decay=0):
        super(Adam, self).__init__(lr)

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
        self.decay_value = self.weight_decay(self.step_counter)

        # init beta_1
        if type(beta_1) == float or type(beta_1) == int:
            self.beta_1 = Constant(beta_1)
        elif isinstance(beta_1, DecayScheduler):
            self.beta_1 = beta_1
        else:
            raise TypeError("Wrong beta_1 type")
        self.beta_1_value = self.beta_1(self.step_counter)

        # init beta_2
        if type(beta_2) == float or type(beta_2) == int:
            self.beta_2 = Constant(beta_2)
        elif isinstance(beta_2, DecayScheduler):
            self.beta_2 = beta_2
        else:
            raise TypeError("Wrong beta_2 type")
        self.beta_2_value = self.beta_2(self.step_counter)

        # init epsilon
        if type(epsilon) == float or type(epsilon) == int:
            self.epsilon = Constant(epsilon)
        elif isinstance(epsilon, DecayScheduler):
            self.epsilon = epsilon
        else:
            raise TypeError("Wrong epsilon type")
        self.epsilon_value = self.epsilon(self.step_counter)

        # init m and v
        self.m = dict()
        self.v = dict()

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
                          self.beta_1_value, self.beta_2_value,
                          self.epsilon_value, self.decay_value)

        # if self.decay_value != 0:
        if self.weight_decay.init_value != 0:
            singa.Axpy(self.decay_value.data, param_value.data, param_grad.data)

        if param_name not in self.m:
            flag = param_value.device.graph_enabled()
            param_value.device.EnableGraph(False)
            self.m[param_name] = tensor.zeros_like(param_value)
            self.v[param_name] = tensor.zeros_like(param_value)
            param_value.device.EnableGraph(flag)

        # overall steps
        # m := beta_1 * m + (1 - beta_1) * grad
        # v := beta_2 * v + (1 - beta_2) * grad * grad
        # m_norm = m / (1 - beta_1 ^ step)
        # v_norm = v / (1 - beta_2 ^ step)
        # param := param - (lr * m_norm) / ( sqrt(v_norm) + epsilon) )

        step = self.step_counter + 1.0

        # m := beta_1 * m + (1 - beta_1) * grad
        tmp = 1.0 - self.beta_1_value
        self.m[param_name] *= self.beta_1_value
        singa.Axpy(tmp.data, param_grad.data, self.m[param_name].data)

        # v := beta_2 * v + (1 - beta_2) * grad * grad
        tmp = 1.0 - self.beta_2_value
        self.v[param_name] *= self.beta_2_value
        singa.Axpy(tmp.data, singa.Square(param_grad.data),
                   self.v[param_name].data)

        # m_norm = m / (1 - beta_1 ^ step)
        tmp = tensor.pow(self.beta_1_value, step)
        tmp = 1.0 - tmp
        m_norm = self.m[param_name] / tmp

        # v_norm = v / (1 - beta_2 ^ step)
        tmp = tensor.pow(self.beta_2_value, step)
        tmp = 1.0 - tmp
        v_norm = self.v[param_name] / tmp

        # param := param - (lr * m_norm) / ( sqrt(v_norm) + epsilon) )
        a = tensor.sqrt(v_norm) + self.epsilon_value
        tmp = m_norm / a

        minus_lr = 0.0 - self.lr_value
        singa.Axpy(minus_lr.data, tmp.data, param_value.data)

    def step(self):
        # increment step counter, lr and moment
        super().step()
        decay_value = self.weight_decay(self.step_counter)
        beta_1_value = self.beta_1(self.step_counter)
        beta_2_value = self.beta_2(self.step_counter)
        self.decay_value.copy_from(decay_value)
        self.beta_1_value.copy_from(beta_1_value)
        self.beta_2_value.copy_from(beta_2_value)

    def get_states(self):
        states = super().get_states()
        states['m'] = self.m  # a dict for 1st order moments tensors
        states['v'] = self.v  # a dict for 2nd order moments tensors
        return states

    def set_states(self, states):
        super().set_states(states)
        if 'm' in states:
            self.m = states['m']
        if 'v' in states:
            self.v = states['v']


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
            self.communicator = singa.Communicator(local_rank, world_size,
                                                   nccl_id, buffSize)

        self.world_size = self.communicator.world_size
        self.local_rank = self.communicator.local_rank
        self.global_rank = self.communicator.global_rank

    def __call__(self, loss):
        self.backward_and_update(loss)

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
        self.opt.step()

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
            assert p.dtype == tensor.float32, (
                'This function is only available for input tensor precision 32 bit, '
                'which are converted into 16 bits before transmit')
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
        self.opt.step()

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
        self.opt.step()

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
        self.opt.step()
