# Base unbuntu 14.04 image from nvidia/cuda
# Change tags to build with different cuda/cudnn versions:
#   FROM nvidia/cuda:8.0-cudnn5-devel
#   FROM nvidia/cuda:7.5-cudnn5-devel
#   FROM nvidia/cuda:7.5-cudnn4-devel
#   FROM nvidia/cuda:7.0-cudnn4-devel
FROM nvidia/cuda:7.5-cudnn5-devel

# install dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential autoconf libtool cmake libpcre3-dev python-dev python-pip wget openssh-server \
    && apt-get clean && apt-get autoremove && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* \
    && pip install -U pip wheel numpy setuptools unittest-xml-reporting protobuf


# install swig 3.0
RUN wget http://prdownloads.sourceforge.net/swig/swig-3.0.10.tar.gz && \
    tar zxf swig-3.0.10.tar.gz && cd swig-3.0.10 && \
    ./configure && make && make install

# set environment
ENV CPLUS_INCLUDE_PATH /usr/local/lib/python2.7/dist-packages/numpy/core/include:${CPLUS_INCLUDE_PATH}
ENV CMAKE_INCLUDE_PATH /usr/local/cuda/include:${CMAKE_INCLUDE_PATH}
ENV CMAKE_LIBRARY_PATH /usr/local/cuda/lib64:${CMAKE_LIBRARY_PATH}
ENV OS_VERSION ubuntu14.04

# download singa source
RUN git clone https://github.com/apache/incubator-singa.git

# config ssh service
RUN mkdir /var/run/sshd
RUN echo 'root:singa' | chpasswd
RUN sed -i 's/PermitRootLogin without-password/PermitRootLogin yes/' /etc/ssh/sshd_config
# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

# dump environment variables into files, so that ssh can see also
RUN env | grep _ >> /etc/environment

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
