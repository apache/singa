
swig -c++ -python driver.i
g++ -fPIC driver.cc driver_wrap.cxx -shared -o _driver.so -DMSHADOW_USE_CUDA=0 \
    -DMSHADOW_USE_CBLAS=1 -DMSHADOW_USE_MKL=0 -DUSE_GPU -std=c++11 \
    -I/usr/cuda-7.5/include -I../include -I/usr/include/python2.7/
