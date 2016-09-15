#singa-incubating-0.3.0 Release Notes

---

SINGA is a general distributed deep learning platform for training big deep
learning models over large datasets. It is designed with an intuitive
programming model based on the layer abstraction. SINGA supports a wide variety
of popular deep learning models.

This release includes following features:

  * GPU Support
    * [SINGA-131] Implement and optimize hybrid training using both CPU and GPU
    * [SINGA-136] Support cuDNN v4
    * [SINGA-134] Extend SINGA to run over a GPU cluster
    * [SINGA-157] Change the priority of cudnn library and install libsingagpu.so

  * Remove Dependences
    * [SINGA-156] Remove the dependency on ZMQ for single process training
    * [SINGA-155] Remove zookeeper for single-process training

  * Python Binding
    * [SINGA-126] Python Binding for Interactive Training

  * Other Improvements
    * [SINGA-80] New Blob Level and Address Level Math Operation Interface
    * [SINGA-130] Data Prefetching
    * [SINGA-145] New SGD based optimization Updaters: AdaDelta, Adam, AdamMax

  * Bugs Fixed
    * [SINGA-148] Race condition between Worker threads and Driver
    * [SINGA-150] Mesos Docker container failed
    * [SIGNA-141] Undesired Hash collision when locating process id to worker…
    * [SINGA-149] Docker build fail
    * [SINGA-143] The compilation cannot detect libsingagpu.so file


