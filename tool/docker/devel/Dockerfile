# Base unbuntu 16.04 image
FROM ubuntu:latest

MAINTAINER incubator-singa dev@singa.incubator.apache.org

# install dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git cmake libprotobuf-dev libopenblas-dev protobuf-compiler python-dev python-pip swig wget\
    && apt-get clean && apt-get autoremove && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* \
    && pip --no-cache-dir install -U pip numpy setuptools

# set environment
ENV CPLUS_INCLUDE_PATH /usr/local/lib/python2.7/dist-packages/numpy/core/include:${CPLUS_INCLUDE_PATH}

# download singa source
RUN git clone https://github.com/apache/incubator-singa.git

# compile singa and install pysinga
RUN cd incubator-singa && mkdir build && cd build && cmake .. && make && cd python && pip install . && cd ../../ && rm -rf build
