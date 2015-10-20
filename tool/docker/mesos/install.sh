#!/bin/bash
source /root/.bashrc
#download
cd /opt
wget -c http://archive.apache.org/dist/mesos/0.22.0/mesos-0.22.0.tar.gz
wget https://www.comp.nus.edu.sg/~dinhtta/files/mesos_patch 
tar -zxvf mesos-0.22.0.tar.gz

#patch and install mesos
cd /opt/mesos-0.22.0
patch -p5 < ../mesos_patch
mkdir build; cd build
../configure
make
sudo make install

