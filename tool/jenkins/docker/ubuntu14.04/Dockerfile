# Base unbuntu 14.04 image from nvidia/cuda
# Change tags to build with different cuda/cudnn versions:
#   FROM nvidia/cuda:8.0-cudnn5-devel
#   FROM nvidia/cuda:7.5-cudnn5-devel
#   FROM nvidia/cuda:7.5-cudnn4-devel
#   FROM nvidia/cuda:7.0-cudnn4-devel
FROM nvidia/cuda:8.0-cudnn5-devel

# install dependencies
RUN apt-get update && apt-get install -y \
    git vim wget build-essential autoconf libtool cmake \
    libprotobuf-dev libopenblas-dev protobuf-compiler \
    libpcre3-dev python-dev python-pip && \
    pip install numpy unittest-xml-reporting

# install swig 3.0
RUN wget http://prdownloads.sourceforge.net/swig/swig-3.0.10.tar.gz && \
    tar zxf swig-3.0.10.tar.gz && cd swig-3.0.10 && \
    ./configure && make && make install

# set environment
ENV CPLUS_INCLUDE_PATH /usr/local/lib/python2.7/dist-packages/numpy/core/include:${CPLUS_INCLUDE_PATH}
ENV CMAKE_INCLUDE_PATH /usr/local/cuda/include:${CMAKE_INCLUDE_PATH}
ENV CMAKE_LIBRARY_PATH /usr/local/cuda/lib64:${CMAKE_LIBRARY_PATH}

# download singa source
RUN git clone https://github.com/apache/incubator-singa.git

# install ssh service
RUN apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo 'root:singa' | chpasswd
RUN sed -i 's/PermitRootLogin without-password/PermitRootLogin yes/' /etc/ssh/sshd_config
# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

# dump environment variables into files, so that ssh can see also
RUN env | grep _ >> /etc/environment

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
