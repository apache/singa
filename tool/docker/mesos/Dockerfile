#build Mesos+Hadoop on top of SINGA
FROM singa/base
MAINTAINER Anh Dinh<ug93tad@gmail.com>

COPY .bashrc /root/.bashrc
COPY install.sh /opt/install.sh

RUN cd /opt && source ./install.sh

RUN apt-get -y install curl cmake libxml2 libxml2-dev uuid-dev protobuf-compiler libprotobuf-dev libgsasl7-dev libkrb5-dev libboost1.54-all-dev

RUN cd /opt && source /root/.bashrc && wget -c http://www.eu.apache.org/dist/hadoop/common/hadoop-2.6.0/hadoop-2.6.0.tar.gz && tar -zxvf hadoop-2.6.0.tar.gz && git clone https://github.com/PivotalRD/libhdfs3.git && cd libhdfs3 && mkdir build && cd build && cmake ../ && make && make install

#to be removed after SINGA-11 is merged (pulling from Anh's branch instead of pulling from the master)
#RUN source ~/.bashrc && mkdir /root/mesos && cd /root/mesos && git clone https://github.com/ug93tad/incubator-singa && cd incubator-singa && git checkout SINGA-11 && cp -r tool/mesos /root/incubator-singa/tool/ && cd /root/incubator-singa/tool/mesos && make 

RUN source ~/.bashrc && cd /root/incubator-singa/tool/mesos && make

COPY *.xml /opt/hadoop-2.6.0/etc/hadoop/

RUN ssh-keygen -t dsa -P '' -f ~/.ssh/id_dsa && cat ~/.ssh/id_dsa.pub >> ~/.ssh/authorized_keys && echo 'StrictHostKeyChecking no' >> ~/.ssh/config
