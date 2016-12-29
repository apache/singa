# Base unbuntu 16.04 image
FROM ubuntu:latest

MAINTAINER incubator-singa dev@singa.incubator.apache.org

# install dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends git python python-pip \
    && apt-get clean && apt-get autoremove && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* \
    && pip --no-cache-dir install -U pip setuptools


# install pysinga
RUN pip install --upgrade http://www.comp.nus.edu.sg/~dbsystem/singa/assets/file/whl/latest/ubuntu16.04-cpp/singa-1.0.1-py2-none-any.whl
