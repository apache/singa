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

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from src.tools import utils

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
import json

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
    # print ("in accuracy y shape: ", y.shape)
    # print ("in accuracy target shape: ", target.shape)
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


class ModelTrainer:

    @classmethod
    def fully_train_arch(cls,
                         model,
                         use_test_acc: bool,
                         epoch_num,
                         train_loader: DataLoader,
                         val_loader: DataLoader,
                         test_loader: DataLoader,
                         args,
                         logger=None
                         ) -> (float, float, dict):
        """
        Args:
            model:
            use_test_acc:
            epoch_num: how many epoch, set by scheduler
            train_loader:
            val_loader:
            test_loader:
            args:
        Returns:
        """

        if logger is None:
            from src.logger import logger
            logger = logger
        logger.info(f'begin to train, batch size = {args.batch_size}')
        start_time, best_valid_auc = time.time(), 0.

        num_labels = args.num_labels
        lr = args.lr
        iter_per_epoch = args.iter_per_epoch
        # report_freq = args.report_freq
        # given_patience = args.patience

        # assign new values
        args.epoch_num = epoch_num

        # for multiple classification
        # opt_metric = nn.CrossEntropyLoss(reduction='mean').to(device)
        # this is only sutiable when output is dimension 1,
        # opt_metric = nn.BCEWithLogitsLoss(reduction='mean').to(device)

        # optimizer
        # optimizer = optim.Adam(model.parameters(), lr=lr)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max=epoch_num,  # Maximum number of iterations.
        #     eta_min=1e-4)  # Minimum learning rate.
        precision = 'float32'
        mssgd = MSSGD(lr=args.lr, momentum=0.9, weight_decay=1e-4, dtype=singa_dtype[precision])
        device_id = 0
        max_epoch = epoch_num
        # model = arch
        graph = True
        verbosity = 0
        dist_option = 'plain'
        spars = None
        global_rank = 0
        world_size = 1
        # gradient clipping, set the gradient value to be -1 - 1
        # for p in model.parameters():
        #     p.register_hook(lambda grad: torch.clamp(grad, -1., 1.))

        # training params
        # device = args.device
        if args.device == 'cpu':
            dev = singa_device.get_default_device()
        else:  # GPU
            dev = singa_device.create_cuda_gpu_on(args.local_rank)  # need to change to CPU device for CPU-only machines
        dev.SetRandSeed(0)

        # For distributed training, sequential has better performance
        if hasattr(mssgd, "communicator"):
            DIST = True
            sequential = True
        else:
            DIST = False
            sequential = False

        info_dic = {}
        valid_auc = -1
        valid_loss = 0

        ### singa data
        tx = tensor.Tensor((args.batch_size, args.nfeat), dev, singa_dtype[precision])
        ty = tensor.Tensor((args.batch_size,), dev, tensor.int32)
        ### singa data

        model.set_optimizer(mssgd)
        model.compile([tx], is_train=True, use_graph=graph, sequential=sequential)
        dev.SetVerbosity(verbosity)

        # synflow_flag = False ### just change the model to the absolute value
        # for epoch in range(epoch_num):
        # logger.info(f'Epoch [{epoch:3d}/{epoch_num:3d}]')
        # train and eval
        # print("begin to train...")
        # logger.info(f"Begin to train.....")
        # train_auc, train_loss = ModelTrainer.run(logger,
        #                                          epoch, iter_per_epoch, model, train_loader, opt_metric, args,
        #                                          optimizer=optimizer, namespace='train')
        # scheduler.step()
        # logger.info(f"Begin to evaluate on valid.....")
        # print("begin to evaluate...")
        # valid_auc, valid_loss = ModelTrainer.run(logger,
        #                                          epoch, iter_per_epoch, model, val_loader,
        #                                          opt_metric, args, namespace='val')

        # if use_test_acc:
        #     logger.info(f"Begin to evaluate on test.....")
        #     test_auc, test_loss = ModelTrainer.run(logger,
        #                                            epoch, iter_per_epoch, model, test_loader,
        #                                            opt_metric, args, namespace='test')
        # else:
        #     test_auc = -1

        # info_dic[epoch] = {
        #     "train_auc": train_auc,
        #     "valid_auc": valid_auc,
        #     "train_loss": train_loss,
        #     "valid_loss": valid_loss,
        #     "train_val_total_time": time.time() - start_time}

        # record best auc and save checkpoint
        # if valid_auc >= best_valid_auc:
        #     best_valid_auc, best_test_auc = valid_auc, test_auc
        #     logger.info(f'best valid auc: valid {valid_auc:.4f}, test {test_auc:.4f}')
        # else:
        #     logger.info(f'valid {valid_auc:.4f}, test {test_auc:.4f}')

        # Training and evaluation loop
        for epoch in range(max_epoch):
            start_time = time.time()
            logger.info(f'Epoch [{epoch:3d}/{epoch_num:3d}]')
            # np.random.shuffle(idx)

            if global_rank == 0:
                print('Starting Epoch %d:' % (epoch))
                logger.info('Starting Epoch %d:' % (epoch))

            # Training phase
            train_correct = np.zeros(shape=[1], dtype=np.float32)
            test_correct = np.zeros(shape=[1], dtype=np.float32)
            train_loss = np.zeros(shape=[1], dtype=np.float32)

            model.train()
            # print ("num_train_batch: \n", num_train_batch)
            # print ()
            batch_idx = 0
            # for b in range(num_train_batch):
            for batch_idx, batch in enumerate(train_loader, start=1):
                if batch_idx % 50 == 0:
                    print("trainer.py train batch_idx: \n", batch_idx)
                    logger.info("trainer.py train batch_idx: \n", batch_idx)
                # Generate the batch data in this iteration
                # x = train_x[idx[b * batch_size:(b + 1) * batch_size]]
                # if model.dimension == 4:
                #     x = augmentation(x, batch_size)
                #     if (image_size != model.input_size):
                #         x = resize_dataset(x, model.input_size)
                # x = x.astype(np_dtype[precision])
                # y = train_y[idx[b * batch_size:(b + 1) * batch_size]]

                y = batch['y'].cpu().numpy()
                batch['id'] = batch['id'].cpu().numpy().astype(int)
                # batch['value'] = batch['value'].to(args.device)
                x = np.zeros((batch['id'].shape[0], args.nfeat), dtype=np.float32)
                # print ("target shape: ", target.shape)
                # print ("target: ", target)
                # print ("batch['id'] shape: ", batch['id'].shape)
                # print ("batch['id']: ", batch['id'])
                # print ("batch['value'] shape: ", batch['value'].shape)
                # print ("batch['value']: ", batch['value'])
                # print ("batch['id'].cpu().numpy().astype(int): \n", batch['id'].cpu().numpy().astype(int))
                for i in range(batch['id'].shape[0]):
                    x[i][batch['id'][i]] = (np.float32)(1.0)
                x = x.astype(dtype=np.float32)
                y = y.astype(dtype=np.int32)

                if x.shape[0] != args.batch_size:  # last batch not processing
                    continue

                synflow_flag = False
                # Train the model
                # if True: # normal train steps
                # Copy the patch data into input tensors
                # print ("normal train steps\n")
                # print ("x.astype(np.float32): \n", x.astype(np.float32))
                # print ("y: \n", y)
                tx = tensor.Tensor(x.shape, dev, singa_dtype[precision])
                ty = tensor.Tensor((y.shape[0],), dev, tensor.int32)
                tx.copy_from_numpy(x)  # dtype=np.float32
                # print ("tx: \n", tx)
                ty.copy_from_numpy(y)
                # print ("ty: \n", ty)
                # print ("normal before model(tx, ty, synflow_flag, dist_option, spars)")
                # print ("train_cnn tx: \n", tx)
                # print ("train_cnn ty: \n", ty)
                # print ("trainer.py train before model forward ...")
                # print ("model: ", model)
                pn_p_g_list, out, loss = model(tx, ty, dist_option, spars, synflow_flag)
                # print ("trainer.py train normal after model(tx, ty, synflow_flag, dist_option, spars)")
                # print ("trainer.py train tx shape: ", tx.shape)
                # print ("trainer.py train ty shape: ", ty.shape)
                # print ("trainer.py train out.shape: ", out.shape)
                # print ("trainer.py train out: ", out)
                # print ("trainer.py train y shape: ", y.shape)
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
                       (batch_idx * args.batch_size * world_size)),
                      flush=True)
                print("train total batch_idx: ", batch_idx)

                logger.info('Training loss = %f, training accuracy = %f' %
                      (train_loss, train_correct /
                       (batch_idx * args.batch_size * world_size)))

                logger.info("train total batch_idx: ", batch_idx)
                train_metric = train_correct / (batch_idx * args.batch_size * world_size)

            # Evaluation phase
            model.eval()
            batch_idx = 0
            # for b in range(num_val_batch):
            # print ("evaluation begins")
            for batch_idx, batch in enumerate(test_loader, start=1):
                # print ("trainer.py test batch_idx: \n", batch_idx)
                # x = val_x[b * batch_size:(b + 1) * batch_size]
                # if model.dimension == 4:
                #     if (image_size != model.input_size):
                #         x = resize_dataset(x, model.input_size)
                # x = x.astype(np_dtype[precision])
                # y = val_y[b * batch_size:(b + 1) * batch_size]
                # batch['value'] = batch['value'].cpu().numpy().astype(np_dtype[precision])
                # x = batch['value'].cpu().numpy().astype(np_dtype[precision])

                y = batch['y'].cpu().numpy()
                batch['id'] = batch['id'].cpu().numpy().astype(int)
                # batch['value'] = batch['value'].to(args.device)
                x = np.zeros((batch['id'].shape[0], args.nfeat), dtype=np.float32)
                # print ("target shape: ", target.shape)
                # print ("target: ", target)
                # print ("batch['id'] shape: ", batch['id'].shape)
                # print ("batch['id']: ", batch['id'])
                # print ("batch['value'] shape: ", batch['value'].shape)
                # print ("batch['value']: ", batch['value'])
                # print ("batch['id'].cpu().numpy().astype(int): \n", batch['id'].cpu().numpy().astype(int))
                for i in range(batch['id'].shape[0]):
                    x[i][batch['id'][i]] = (np.float32)(1.0)
                # print ("x[1]: \n", x[1])
                x = x.astype(dtype=np.float32)
                y = y.astype(dtype=np.int32)

                if x.shape[0] != (args.batch_size * 8):  # last batch not processing
                    # print ("trainer.py test batch_idx: ", batch_idx)
                    # print ("trainer.py test x.shape: ", x.shape)
                    continue

                tx = tensor.Tensor(x.shape, dev, singa_dtype[precision])
                ty = tensor.Tensor((y.shape[0],), dev, tensor.int32)
                tx.copy_from_numpy(x)
                ty.copy_from_numpy(y)
                # print ("trainer.py test tx shape: ", tx.shape)
                out_test = model(tx)
                # print ("trainer.py test out_test shape: ", out_test.shape)
                # print ("trainer.py test y shape: ", y.shape)
                # print ("trainer.py out_test: ", out_test)
                # print ("trainer.py y: ", y)
                test_correct += accuracy(tensor.to_numpy(out_test), y)
                # print ("test_correct: ", test_correct)

            if DIST:
                # Reduce the evaulation accuracy from multiple devices
                test_correct = reduce_variable(test_correct, mssgd, reducer)

            # Output the evaluation accuracy
            if global_rank == 0:
                print('Evaluation accuracy = %f, Elapsed Time = %fs' %
                      (test_correct / (batch_idx * args.batch_size * 8 * world_size),
                       time.time() - start_time),
                      flush=True)

                logger.info('Evaluation accuracy = %f, Elapsed Time = %fs' %
                      (test_correct / (batch_idx * args.batch_size * 8 * world_size),
                       time.time() - start_time))
                # print ("test all batch_idx: ", batch_idx)
                test_metric = test_correct / (batch_idx * args.batch_size * 8 * world_size)

            info_dic[epoch] = {
                "train_metric": str(train_metric[0]),
                "test_metric": str(test_metric[0]),
                "train_loss": str(train_loss[0]),
                # "valid_loss": valid_loss,
                "train_test_total_time": str(time.time() - start_time)}

        dev.PrintTimeProfiling()

        # return valid_auc, time.time() - start_time, info_dic
        print("info_dic: ", info_dic)
        logger.info("info_dic: ", info_dic)

        logger.info(json.dumps(info_dic))

        test_metric = train_metric
        return test_metric, time.time() - start_time, info_dic

    @classmethod
    def fully_train_arch_origin(cls,
                                model: nn.Module,
                                use_test_acc: bool,
                                epoch_num,
                                train_loader: DataLoader,
                                val_loader: DataLoader,
                                test_loader: DataLoader,
                                args,
                                logger=None
                                ) -> (float, float, dict):
        """
        Args:
            model:
            use_test_acc:
            epoch_num: how many epoch, set by scheduler
            train_loader:
            val_loader:
            test_loader:
            args:
        Returns:
        """

        if logger is None:
            from src.logger import logger
            logger = logger

        start_time, best_valid_auc = time.time(), 0.

        # training params
        device = args.device
        num_labels = args.num_labels
        lr = args.lr
        iter_per_epoch = args.iter_per_epoch
        # report_freq = args.report_freq
        # given_patience = args.patience

        # assign new values
        args.epoch_num = epoch_num

        # for multiple classification
        opt_metric = nn.CrossEntropyLoss(reduction='mean').to(device)
        # this is only sutiable when output is dimension 1,
        # opt_metric = nn.BCEWithLogitsLoss(reduction='mean').to(device)

        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epoch_num,  # Maximum number of iterations.
            eta_min=1e-4)  # Minimum learning rate.

        # gradient clipping, set the gradient value to be -1 - 1
        for p in model.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -1., 1.))

        info_dic = {}
        valid_auc = -1
        valid_loss = 0
        for epoch in range(epoch_num):
            logger.info(f'Epoch [{epoch:3d}/{epoch_num:3d}]')
            # train and eval
            # print("begin to train...")
            logger.info(f"Begin to train.....")
            train_auc, train_loss = ModelTrainer.run(logger,
                                                     epoch, iter_per_epoch, model, train_loader, opt_metric, args,
                                                     optimizer=optimizer, namespace='train')
            scheduler.step()
            logger.info(f"Begin to evaluate on valid.....")
            # print("begin to evaluate...")
            valid_auc, valid_loss = ModelTrainer.run(logger,
                                                     epoch, iter_per_epoch, model, val_loader,
                                                     opt_metric, args, namespace='val')

            if use_test_acc:
                logger.info(f"Begin to evaluate on test.....")
                test_auc, test_loss = ModelTrainer.run(logger,
                                                       epoch, iter_per_epoch, model, test_loader,
                                                       opt_metric, args, namespace='test')
            else:
                test_auc = -1

            info_dic[epoch] = {
                "train_auc": train_auc,
                "valid_auc": valid_auc,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "train_val_total_time": time.time() - start_time}

            # record best auc and save checkpoint
            if valid_auc >= best_valid_auc:
                best_valid_auc, best_test_auc = valid_auc, test_auc
                logger.info(f'best valid auc: valid {valid_auc:.4f}, test {test_auc:.4f}')
            else:
                logger.info(f'valid {valid_auc:.4f}, test {test_auc:.4f}')

        return valid_auc, time.time() - start_time, info_dic

    @classmethod
    def fully_evaluate_arch(cls,
                            model: nn.Module,
                            use_test_acc: bool,
                            epoch_num,
                            val_loader: DataLoader,
                            test_loader: DataLoader,
                            args,
                            logger=None,
                            ) -> (float, float, dict):
        """
        Args:
            model:
            use_test_acc:
            epoch_num: how many epoch, set by scheduler
            val_loader:
            test_loader:
            args:
        Returns:
        """

        if logger is None:
            from src.logger import logger
            logger = logger

        start_time, best_valid_auc = time.time(), 0.

        device = args.device
        iter_per_epoch = args.iter_per_epoch
        args.epoch_num = epoch_num
        opt_metric = nn.CrossEntropyLoss(reduction='mean').to(device)

        info_dic = {}
        valid_auc = -1
        valid_loss = 0
        for epoch in range(epoch_num):
            logger.info(f'Epoch [{epoch:3d}/{epoch_num:3d}]')
            # print("begin to evaluate...")
            valid_auc, valid_loss = ModelTrainer.run(logger,
                                                     epoch, iter_per_epoch, model, val_loader,
                                                     opt_metric, args, namespace='val')

            if use_test_acc:
                test_auc, test_loss = ModelTrainer.run(logger,
                                                       epoch, iter_per_epoch, model, test_loader,
                                                       opt_metric, args, namespace='test')
            else:
                test_auc = -1

            # record best auc and save checkpoint
            if valid_auc >= best_valid_auc:
                best_valid_auc, best_test_auc = valid_auc, test_auc
                logger.info(f'best valid auc: valid {valid_auc:.4f}, test {test_auc:.4f}')
            else:
                logger.info(f'valid {valid_auc:.4f}, test {test_auc:.4f}')

        return valid_auc, time.time() - start_time, info_dic

    #  train one epoch of train/val/test
    @classmethod
    def run(cls, logger, epoch, iter_per_epoch, model, data_loader, opt_metric, args, optimizer=None,
            namespace='train'):
        if optimizer:
            model.train()
        else:
            model.eval()

        time_avg, timestamp = utils.AvgrageMeter(), time.time()
        loss_avg, auc_avg = utils.AvgrageMeter(), utils.AvgrageMeter()

        batch_idx = 0
        for batch_idx, batch in enumerate(data_loader):
            # if suer set this, then only train fix number of iteras
            # stop training current epoch for evaluation
            if namespace == 'train' and iter_per_epoch is not None and batch_idx >= iter_per_epoch:
                logger.info(f"Traing Iteration {batch_idx} > iter_per_epoch = {iter_per_epoch}, breakout")
                break

            target = batch['y'].type(torch.LongTensor).to(args.device)
            batch['id'] = batch['id'].to(args.device)
            batch['value'] = batch['value'].to(args.device)

            if namespace == 'train':
                y = model(batch)
                loss = opt_metric(y, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    y = model(batch)
                    loss = opt_metric(y, target)

            # for multiple classification
            auc = utils.roc_auc_compute_fn(torch.nn.functional.softmax(y, dim=1)[:, 1], target)
            # for binary classification
            # auc = utils.roc_auc_compute_fn(y, target)
            loss_avg.update(loss.item(), target.size(0))
            auc_avg.update(auc, target.size(0))

            time_avg.update(time.time() - timestamp)
            timestamp = time.time()
            if batch_idx % args.report_freq == 0:
                logger.info(f'Epoch [{epoch:3d}/{args.epoch_num}][{batch_idx:3d}/{len(data_loader)}]\t'
                            f'{time_avg.val:.3f} ({time_avg.avg:.3f}) AUC {auc_avg.val:4f} ({auc_avg.avg:4f}) '
                            f'Loss {loss_avg.val:8.4f} ({loss_avg.avg:8.4f})')

                # print(f'Epoch [{epoch:3d}/{args.epoch_num}][{batch_idx:3d}/{len(data_loader)}]\t'
                #       f'{time_avg.val:.3f} ({time_avg.avg:.3f}) AUC {auc_avg.val:4f} ({auc_avg.avg:4f}) '
                #       f'Loss {loss_avg.val:8.4f} ({loss_avg.avg:8.4f})')

        # record the last epoch information
        logger.info(f'Epoch [{epoch:3d}/{args.epoch_num}][{batch_idx:3d}/{len(data_loader)}]\t'
                    f'{time_avg.val:.3f} ({time_avg.avg:.3f}) AUC {auc_avg.val:4f} ({auc_avg.avg:4f}) '
                    f'Loss {loss_avg.val:8.4f} ({loss_avg.avg:8.4f})')

        # print(f'Epoch [{epoch:3d}/{args.epoch_num}][{batch_idx:3d}/{len(data_loader)}]\t'
        #       f'{time_avg.val:.3f} ({time_avg.avg:.3f}) AUC {auc_avg.val:4f} ({auc_avg.avg:4f}) '
        #       f'Loss {loss_avg.val:8.4f} ({loss_avg.avg:8.4f})')

        logger.info(f'{namespace}\tTime {utils.timeSince(s=time_avg.sum):>12s} '
                    f'AUC {auc_avg.avg:8.4f} Loss {loss_avg.avg:8.4f}')

        # print(f'{namespace}\tTime {utils.timeSince(s=time_avg.sum):>12s} '
        #       f'AUC {auc_avg.avg:8.4f} Loss {loss_avg.avg:8.4f}')

        return auc_avg.avg, loss_avg.avg
