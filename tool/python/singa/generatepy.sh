
swig -c++ -python driver.i
g++ -fPIC /home/zhongle/incubator-singa/src/driver.cc driver_wrap.cxx -shared -o _driver.so \
    -L/home/zhongle/incubator-singa/.libs/ -lsinga -DMSHADOW_USE_CUDA=0 \
    -DMSHADOW_USE_CBLAS=1 -DMSHADOW_USE_MKL=0 -DUSE_GPU -std=c++11 \
    -I/usr/cuda-7.5/include -I/home/zhongle/local/include \
    -I/home/zhongle/incubator-singa/include \
    -I/usr/include/python2.7/
