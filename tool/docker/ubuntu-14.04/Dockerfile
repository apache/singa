FROM ubuntu:14.04
MAINTAINER Aaron WWF<dcswuw@gmail.com>


ENV PREFIX=/usr/local
ENV LIBRARY_PATH=/opt/OpenBLAS/lib:$PREFIX/lib:$LIBRARY_PATH
ENV LD_LIBRARY_PATH=/opt/OpenBLAS/lib:$PREFIX/lib:$LD_LIBRARY_PATH
ENV CPLUS_INCLUDE_PATH=/opt/OpenBLAS/include:$PREFIX/include:$CPLUS_INCLUDE_PATH
ENV SINGA_HOME=/opt/incubator-singa
ENV PATH=/opt/OpenBLAS/bin:$SINGA_HOME/bin:$PREFIX/bin:$PATH

RUN apt-get update && apt-get install -y git unzip make autotools-dev \
    automake autoconf wget gcc g++ libtool python2.7 python2.7-dev \
    build-essential swig python-pip libtiff5-dev libjpeg8-dev zlib1g-dev \
    libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python-tk

RUN pip install Image flask numpy

#RUN ln -s /usr/bin/python2.7 /usr/bin/python

#use /bin/bash instead of /bin/sh
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

COPY . $SINGA_HOME/

RUN cd $SINGA_HOME && ./thirdparty/install.sh all \
    && cd ./tool/python/singa && ./generatepy.sh \
    && cd $SINGA_HOME && ./autogen.sh && ./configure --enable-python \
    && make

