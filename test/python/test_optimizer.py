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
# =============================================================================
from __future__ import division
from builtins import zip
from builtins import range

import unittest
import math
import numpy as np


import singa.tensor as tensor
import singa.optimizer as opt
import singa.device as device
from singa import singa_wrap

if singa_wrap.USE_CUDA:
    cuda = device.create_cuda_gpu()


def np_adam(plist, glist, mlist, vlist, lr, t, b1=0.9, b2=0.999):
    for p, g, m, v in zip(plist, glist, mlist, vlist):
        m *= b1
        m += (1-b1) * g
        v *= b2
        v += (1-b2) * g * g
        alpha = lr * math.sqrt(1. - math.pow(b2, t)) / (1. - math.pow(b1, t))
        p -= alpha * m / (np.sqrt(v) + 1e-8)
        
def np_rmsprop(plist, glist, vlist, lr, t, rho=0.9):
    for p, g, v in zip(plist, glist, vlist):
        v *= rho
        v += (1-rho) * g * g
        p -= lr * g / (np.sqrt(v + 1e-8))
        
def np_momentum(plist, glist, vlist, lr, t, momentum=0.9):
    for p, g, v in zip(plist, glist, vlist):
        v *= momentum
        v += lr * g
        p -= v

def np_adagrad(plist, glist, vlist, lr, t):
    for p, g, v in zip(plist, glist, vlist):
        v += g * g
        p -= lr * g / (np.sqrt(v + 1e-8))


class TestOptimizer(unittest.TestCase):

    def setUp(self):
        self.np_W = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        self.W = tensor.from_numpy(self.np_W)
        self.np_g = np.array([0.1, 0.3, 0.1, 0.2], dtype=np.float32)
        self.g = tensor.from_numpy(self.np_g)

    def to_cuda(self):
        self.W.to_device(cuda)
        self.g.to_device(cuda)

    def test_sgd(self):
        lr = 0.1
        sgd = opt.SGD(lr)
        sgd.apply(0, self.g, self.W, 'w')
        w = tensor.to_numpy(self.W)
        for i in range(self.W.size()):
            self.assertAlmostEqual(w[i], self.np_W[i] - lr * self.np_g[i])

    def test_adam(self):
        lr = 0.1
        n, m = 4, 6
        p1 = np.random.rand(n, m)
        p2 = np.random.rand(n, m)
        g1 = np.random.rand(n, m) * 0.01
        g2 = np.random.rand(n, m) * 0.01
        m1 = np.zeros((n, m))
        m2 = np.zeros((n, m))
        v1 = np.zeros((n, m))
        v2 = np.zeros((n, m))
        t1 = tensor.from_numpy(p1)
        t2 = tensor.from_numpy(p2)
        tg1 = tensor.from_numpy(g1)
        tg2 = tensor.from_numpy(g2)

        for t in range(1, 10):
            np_adam([p1, p2], [g1, g2], [m1, m2], [v1, v2], lr, t)

        adam = opt.Adam(lr=lr)
        for t in range(1, 10):
            adam.apply(0, tg1, t1, 'p1', t)
            adam.apply(0, tg2, t2, 'p2', t)

        t1 = tensor.to_numpy(t1)
        t2 = tensor.to_numpy(t2)
        for t, p in zip([t1, t2], [p1, p2]):
            for i in range(n):
                for j in range(m):
                    self.assertAlmostEqual(t[i, j], p[i, j], 6)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_sgd_cuda(self):
        lr = 0.1
        sgd = opt.SGD(lr)
        self.to_cuda()
        sgd.apply(0, self.g, self.W, 'w')
        self.W.to_host()
        w = tensor.to_numpy(self.W)
        for i in range(self.W.size()):
            self.assertAlmostEqual(w[i], self.np_W[i] - lr * self.np_g[i])

    def test_constraint(self):
        threshold = 0.02
        cons = opt.L2Constraint(threshold)
        cons.apply(0, self.W, self.g)
        g = tensor.to_numpy(self.g)
        nrm = np.linalg.norm(self.np_g) / self.np_g.size
        for i in range(g.size):
            self.assertAlmostEqual(g[i], self.np_g[i] * threshold / nrm)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_constraint_cuda(self):
        threshold = 0.02
        self.to_cuda()
        cons = opt.L2Constraint(threshold)
        cons.apply(0, self.W, self.g)
        self.g.to_host()
        g = tensor.to_numpy(self.g)
        nrm = np.linalg.norm(self.np_g) / self.np_g.size
        for i in range(g.size):
            self.assertAlmostEqual(g[i], self.np_g[i] * threshold / nrm)

    def test_regularizer(self):
        coefficient = 0.0001
        reg = opt.L2Regularizer(coefficient)
        reg.apply(0, self.W, self.g)
        g = tensor.to_numpy(self.g)
        for i in range(g.size):
            self.assertAlmostEqual(g[i],
                                   self.np_g[i] + coefficient * self.np_W[i])

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_regularizer_cuda(self):
        coefficient = 0.0001
        reg = opt.L2Regularizer(coefficient)
        self.to_cuda()
        reg.apply(0, self.W, self.g)
        self.g.to_host()
        g = tensor.to_numpy(self.g)
        for i in range(g.size):
            self.assertAlmostEqual(g[i],
                                   self.np_g[i] + coefficient * self.np_W[i])

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_adam_cuda(self):
        lr = 0.1
        n, m = 4, 6
        p1 = np.random.rand(n, m)
        p2 = np.random.rand(n, m)
        g1 = np.random.rand(n, m) * 0.01
        g2 = np.random.rand(n, m) * 0.01
        m1 = np.zeros((n, m))
        m2 = np.zeros((n, m))
        v1 = np.zeros((n, m))
        v2 = np.zeros((n, m))
        t1 = tensor.from_numpy(p1)
        t2 = tensor.from_numpy(p2)
        tg1 = tensor.from_numpy(g1)
        tg2 = tensor.from_numpy(g2)

        for t in range(1, 10):
            np_adam([p1, p2], [g1, g2], [m1, m2], [v1, v2], lr, t)

        adam = opt.Adam(lr=lr)
        self.to_cuda()
        for t in range(1, 10):
            adam.apply(0, tg1, t1, 'p1', t)
            adam.apply(0, tg2, t2, 'p2', t)

        t1 = tensor.to_numpy(t1)
        t2 = tensor.to_numpy(t2)
        for t, p in zip([t1, t2], [p1, p2]):
            for i in range(n):
                for j in range(m):
                    self.assertAlmostEqual(t[i, j], p[i, j], 6)

    def test_rmsprop(self):
        lr = 0.1
        n, m = 2, 2
        p1 = np.random.rand(n, m)
        p2 = np.random.rand(n, m)
        g1 = np.random.rand(n, m) * 0.01
        g2 = np.random.rand(n, m) * 0.01
        v1 = np.zeros((n, m))
        v2 = np.zeros((n, m))
        t1 = tensor.from_numpy(p1)
        t2 = tensor.from_numpy(p2)
        tg1 = tensor.from_numpy(g1)
        tg2 = tensor.from_numpy(g2)

        for t in range(1, 4):
            np_rmsprop([p1, p2], [g1, g2], [v1, v2], lr, t)

        rsmprop = opt.RMSProp(lr=lr)
        for t in range(1, 4):
            rsmprop.apply(0, tg1, t1, 'p1', t)
            rsmprop.apply(0, tg2, t2, 'p2', t)

        t1 = tensor.to_numpy(t1)
        t2 = tensor.to_numpy(t2)
        for t, p in zip([t1, t2], [p1, p2]):
            for i in range(n):
                for j in range(m):
                    self.assertAlmostEqual(t[i, j], p[i, j], 2)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_rmsprop_cuda(self):
        lr = 0.1
        n, m = 2, 2
        p1 = np.random.rand(n, m)
        p2 = np.random.rand(n, m)
        g1 = np.random.rand(n, m) * 0.01
        g2 = np.random.rand(n, m) * 0.01
        v1 = np.zeros((n, m))
        v2 = np.zeros((n, m))
        t1 = tensor.from_numpy(p1)
        t2 = tensor.from_numpy(p2)
        tg1 = tensor.from_numpy(g1)
        tg2 = tensor.from_numpy(g2)

        for t in range(1, 4):
            np_rmsprop([p1, p2], [g1, g2], [v1, v2], lr, t)

        rsmprop = opt.RMSProp(lr=lr)
        self.to_cuda()
        for t in range(1, 4):
            rsmprop.apply(0, tg1, t1, 'p1', t)
            rsmprop.apply(0, tg2, t2, 'p2', t)

        t1 = tensor.to_numpy(t1)
        t2 = tensor.to_numpy(t2)
        for t, p in zip([t1, t2], [p1, p2]):
            for i in range(n):
                for j in range(m):
                    self.assertAlmostEqual(t[i, j], p[i, j], 2)

    def test_momentum(self):
        lr = 0.1
        n, m = 2, 2
        p1 = np.random.rand(n, m)
        p2 = np.random.rand(n, m)
        g1 = np.random.rand(n, m) * 0.01
        g2 = np.random.rand(n, m) * 0.01
        v1 = np.zeros((n, m))
        v2 = np.zeros((n, m))
        t1 = tensor.from_numpy(p1)
        t2 = tensor.from_numpy(p2)
        tg1 = tensor.from_numpy(g1)
        tg2 = tensor.from_numpy(g2)

        for t in range(1, 4):
            np_momentum([p1, p2], [g1, g2], [v1, v2], lr, t)

        momentum = opt.SGD(lr, momentum=0.9)
        for t in range(1, 4):
            momentum.apply(0, tg1, t1, 'p1', t)
            momentum.apply(0, tg2, t2, 'p2', t)

        t1 = tensor.to_numpy(t1)
        t2 = tensor.to_numpy(t2)
        for t, p in zip([t1, t2], [p1, p2]):
            for i in range(n):
                for j in range(m):
                    self.assertAlmostEqual(t[i, j], p[i, j], 2)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_momentum_cuda(self):
        lr = 0.1
        n, m = 2, 2
        p1 = np.random.rand(n, m)
        p2 = np.random.rand(n, m)
        g1 = np.random.rand(n, m) * 0.01
        g2 = np.random.rand(n, m) * 0.01
        v1 = np.zeros((n, m))
        v2 = np.zeros((n, m))
        t1 = tensor.from_numpy(p1)
        t2 = tensor.from_numpy(p2)
        tg1 = tensor.from_numpy(g1)
        tg2 = tensor.from_numpy(g2)

        for t in range(1, 4):
            np_momentum([p1, p2], [g1, g2], [v1, v2], lr, t)

        momentum = opt.SGD(lr, momentum=0.9)
        self.to_cuda()
        for t in range(1, 4):
            momentum.apply(0, tg1, t1, 'p1', t)
            momentum.apply(0, tg2, t2, 'p2', t)

        t1 = tensor.to_numpy(t1)
        t2 = tensor.to_numpy(t2)
        for t, p in zip([t1, t2], [p1, p2]):
            for i in range(n):
                for j in range(m):
                    self.assertAlmostEqual(t[i, j], p[i, j], 2)

    def test_adagrad(self):
        lr = 0.1
        n, m = 2, 2
        p1 = np.random.rand(n, m)
        p2 = np.random.rand(n, m)
        g1 = np.random.rand(n, m) * 0.01
        g2 = np.random.rand(n, m) * 0.01
        v1 = np.zeros((n, m))
        v2 = np.zeros((n, m))
        t1 = tensor.from_numpy(p1)
        t2 = tensor.from_numpy(p2)
        tg1 = tensor.from_numpy(g1)
        tg2 = tensor.from_numpy(g2)

        for t in range(1, 4):
            np_adagrad([p1, p2], [g1, g2], [v1, v2], lr, t)

        adagrad = opt.AdaGrad(lr=lr)
        for t in range(1, 4):
            adagrad.apply(0, tg1, t1, 'p1', t)
            adagrad.apply(0, tg2, t2, 'p2', t)

        t1 = tensor.to_numpy(t1)
        t2 = tensor.to_numpy(t2)
        for t, p in zip([t1, t2], [p1, p2]):
            for i in range(n):
                for j in range(m):
                    self.assertAlmostEqual(t[i, j], p[i, j], 2)

    @unittest.skipIf(not singa_wrap.USE_CUDA, 'CUDA is not enabled')
    def test_adagrad_cuda(self):
        lr = 0.1
        n, m = 2, 2
        p1 = np.random.rand(n, m)
        p2 = np.random.rand(n, m)
        g1 = np.random.rand(n, m) * 0.01
        g2 = np.random.rand(n, m) * 0.01
        v1 = np.zeros((n, m))
        v2 = np.zeros((n, m))
        t1 = tensor.from_numpy(p1)
        t2 = tensor.from_numpy(p2)
        tg1 = tensor.from_numpy(g1)
        tg2 = tensor.from_numpy(g2)

        for t in range(1, 4):
            np_adagrad([p1, p2], [g1, g2], [v1, v2], lr, t)

        adagrad = opt.AdaGrad(lr=lr)
        self.to_cuda()
        for t in range(1, 4):
            adagrad.apply(0, tg1, t1, 'p1', t)
            adagrad.apply(0, tg2, t2, 'p2', t)

        t1 = tensor.to_numpy(t1)
        t2 = tensor.to_numpy(t2)
        for t, p in zip([t1, t2], [p1, p2]):
            for i in range(n):
                for j in range(m):
                    self.assertAlmostEqual(t[i, j], p[i, j], 2)


if __name__ == '__main__':
    unittest.main()
