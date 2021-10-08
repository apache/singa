#!/usr/bin/env python -W ignore::DeprecationWarning
mpiexec -np 8 python train_mpi.py resnet cifar10 -l 0.015 -b 32

