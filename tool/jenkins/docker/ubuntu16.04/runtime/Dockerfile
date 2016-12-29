# Base unbuntu 16.04 image
FROM ubuntu:latest

MAINTAINER incubator-singa dev@singa.incubator.apache.org

# install dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends subversion git python-dev python-pip python-sphinx openssh-server\
    && apt-get clean && apt-get autoremove && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* \
    && pip --no-cache-dir install -U pip setuptools \
    && pip --no-cache-dir install recommonmark

RUN mkdir /var/run/sshd
RUN echo 'root:singa' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

# dump environment variables into files, so that ssh can see also
# RUN env | grep _ >> /etc/environment

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
