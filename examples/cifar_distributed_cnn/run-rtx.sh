#!/usr/bin/env python -W ignore::DeprecationWarning
mpiexec -np 2 python train_cnn.py resnet cifar10 -l 0.015 -b 32

