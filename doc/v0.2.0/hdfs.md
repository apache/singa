# Using HDFS with SINGA

This guide explains how to make use of HDFS as the data store for SINGA jobs. 

1. [Quick start using Docker](#quickstart)
2. [Setup HDFS](#hdfs)
3. [Examples](#examples)    

--
<a name="quickstart"></a>
## Quick start using Docker 

We provide a Docker container built on top of `singa/mesos` (see the <a href="http://singa.incubator.apache.org/docs/docker.html">guide on building SINGA on Docker</a>). 

```
git clone https://github.com/ug93tad/incubator-singa
cd incubator-singa
git checkout SINGA-97-docker
cd tool/docker/hdfs
sudo docker build -t singa/hdfs .
```

Once built, the container image `singa/hdfs` contains the installation of HDFS C++ library (`libhdfs3`) and the latest SINGA code. Many distributed nodes can be launched, and HDFS be set up, by following the <a href="http://singa.incubator.apache.org/docs/mesos.html">guide for running distributed SINGA on Mesos</a>. 

In the following, we assume the HDFS setup with `node0` being the namenode, and `nodei (i>0)` being the datanodes. 

<a name="hdfs"></a>
## Setup HDFS 
There are at least 2 C/C++ client libraries for interacting with HDFS. One is from Hadoop (`libhdfs`), which is a <a href="https://wiki.apache.org/hadoop/LibHDFS">JNI-based library</a>, meaning that communication will go through JVM. The other is `libhdfs3` which is a <a href="https://github.com/PivotalRD/libhdfs3">native C++ library developed by Pivotal</a>, in which the client communicate directly with HDFS via RPC. The current implementation uses the second one. 

1. Install `libhdfs3`: follow the <a href="https://github.com/PivotalRD/libhdfs3#installation">official guide</a>. 

2. **Additional setup**: recent versions of Hadoop (>2.4.x) support short-circuit local reads which bypass network communications (TCP sockets) when retrieving data at the local nodes. `libhdfs3` will throws errors (but will still continue to work) when it finds that short-circuit read is not set. To deal with this complaints, and improve performance, add the following configuration to `hdfs-site.xml` **and to `hdfs-client.xml`**
  
    ```
  <property>
    <name>dfs.client.read.shortcircuit</name>
    <value>true</value>
  </property>
  <property>
    <name>dfs.domain.socket.path</name>
    <value>/var/lib/hadoop-hdfs/dn_socket</value>
  </property>
    ``` 
    Next, at each client, set `LIBHDFS3_CONF` variable to point to `hdfs-client.xml` file:

    ```
  export LIBHDFS3_CONF=$HADOOP_HOME/etc/hadoop/hdfs-client.xml
    ```

<a name="examples"></a>
## Examples
We explain how to run CIFAR10 and MNIST examples. Before training, the data must be uploaded to HDFS. 

### CIFAR10
1. Upload the data to HDFS (done at any of the HDFS nodes)
    * Change `job.conf` to use HDFS: in `examples/cifar10/job.conf`, set `backend` property to `hdfsfile`
    * Create and upload data: 

    ```
    cd examples/cifar10
    cp Makefile.example Makefile
    make create
    hadoop dfs -mkdir /examples/cifar10
    hadoop dfs -copyFromLocal cifar-10-batches-bin /examples/cifar10/
    ```
    If successful, the files should be seen in HDFS via `hadoop dfs -ls /examples/cifar10`

2. Training:
    * Make sure `conf/singa.conf` has correct path to Zookeeper service: 

    ```
    zookeeper_host: "node0:2181"
    ```

    * Make sure `job.conf` has correct paths to the train and test datasets:

    ```
    // train layer
    path: "hdfs://node0:9000/examples/cifar10/train_data.bin"
    mean_file: "hdfs://node0:9000/examples/cifar10/image_mean.bin"
    // test layer
    path: "hdfs://node0:9000/examples/cifar10/test_data.bin"
    mean_file: "hdfs://node0:9000/examples/cifar10/image_mean.bin"
    ```

    * Start training: execute the following command at every node

    ```
    ./singa -conf examples/cifar10/job.conf -singa_conf singa.conf -singa_job 0
    ```

### MNIST
1. Upload the data to HDFS (done at any of the HDFS nodes)
    * Change `job.conf` to use HDFS: in `examples/mnist/job.conf`, set `backend` property to `hdfsfile`
    * Create and upload data:

    ```
    cd examples/mnist
    cp Makefile.example Makefile
    make create
    make compile
    ./create_data.bin train-images-idx3-ubyte train-labels-idx1-ubyte hdfs://node0:9000/examples/mnist/train_data.bin
    ./create_data.bin t10k-images-idx3-ubyte t10k-labels-idx1-ubyte hdfs://node0:9000/examples/mnist/test_data.bin
    ```
    If successful, the files should be seen in HDFS via `hadoop dfs -ls /examples/mnist`

2. Training:
    * Make sure `conf/singa.conf` has correct path to Zookeeper service: 

    ```
    zookeeper_host: "node0:2181"
    ```

    * Make sure `job.conf` has correct paths to the train and test datasets:

    ```
    // train layer
    path: "hdfs://node0:9000/examples/mnist/train_data.bin"
    // test layer
    path: "hdfs://node0:9000/examples/mnist/test_data.bin"
    ```

    * Start training: execute the following command at every node

    ```
    ./singa -conf examples/mnist/job.conf -singa_conf singa.conf -singa_job 0
    ```
