# Base unbuntu 16.04, cuda8.0, cudnn5
FROM nvidia/cuda:8.0-cudnn5-runtime-ubuntu16.04

MAINTAINER incubator-singa dev@singa.incubator.apache.org

# install dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends git python python-pip \
    && apt-get clean && apt-get autoremove && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* \
    && pip --no-cache-dir install -U pip setuptools


# install pysinga TODO(wangwei) install debian package
RUN pip install --upgrade http://www.comp.nus.edu.sg/~dbsystem/singa/assets/file/wheel/linux/latest/ubuntu16.04-cuda8.0-cudnn5/singa-1.0.1-py2-none-any.whl
