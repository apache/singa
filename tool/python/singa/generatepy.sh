#!/usr/bin/env bash
swig -c++ -python driver.i
g++ -fPIC /../../../src/driver.cc driver_wrap.cxx -shared -o _driver.so \
    -L../../../.libs/ -lsinga -DMSHADOW_USE_CUDA=0 \
    -DMSHADOW_USE_CBLAS=1 -DMSHADOW_USE_MKL=0 -std=c++11 \
    -I/../../../include \
    -I/usr/include/python2.7/
