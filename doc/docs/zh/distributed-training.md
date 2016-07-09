# Distributed Training

---

SINGA is designed for distributed training of large deep learning models with huge amount of training data.
We also provide high-level descriptions of design behind SINGA's distributed architecture. 

* [System Architecture](architecture.html)

* [Training Frameworks](frameworks.html)

* [System Communication](communication.html)

SINGA supports different options for training a model in parallel, includeing data parallelism, model parallelism and hybrid parallelism.

* [Hybrid Parallelism](hybrid.html)

SINGA is intergrated with Mesos, so that distributed training can be started as a Mesos framework. Currently, the Mesos cluster can be set up from SINGA containers, i.e. we provide Docker images that bundles Mesos and SINGA together. Refer to the guide below for instructions as how to start and use the cluster.

* [Distributed training on Mesos](mesos.html)

SINGA can run on top of distributed storage system to achieve scalability. The current version of SINGA supports HDFS.

* [Running SINGA on HDFS](hdfs.html)

