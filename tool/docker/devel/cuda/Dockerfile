# Base unbuntu 16.04 image from nvidia/cuda
# Change tags to build with different cuda/cudnn versions:
#   FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04
FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

MAINTAINER incubator-singa dev@singa.incubator.apache.org

# install dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends git g++ cmake libprotobuf-dev libopenblas-dev protobuf-compiler python-dev python-pip swig wget\
    && apt-get clean && apt-get autoremove && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* \
    && pip --no-cache-dir install -U pip numpy setuptools

# set environment
ENV CPLUS_INCLUDE_PATH /usr/local/lib/python2.7/dist-packages/numpy/core/include:${CPLUS_INCLUDE_PATH}
ENV CMAKE_INCLUDE_PATH /usr/local/cuda/include:${CMAKE_INCLUDE_PATH}
ENV CMAKE_LIBRARY_PATH /usr/local/cuda/lib64:${CMAKE_LIBRARY_PATH}

# download singa source
RUN git clone https://github.com/apache/incubator-singa.git

RUN cd incubator-singa && mkdir build && cd build && cmake -DUSE_CUDA=ON .. && make && cd python && pip install . && cd ../../ && rm -rf build
