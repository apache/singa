# Code Structure

---

<!--

### Worker Side

#### Main Classes

<img src="../images/code-structure/main.jpg" style="width: 550px"/>

* **Worker**: start the solver to conduct training or resume from previous training snapshots.
* **Solver**: construct the neural network and run training algorithms over it. Validation and testing is also done by the solver along the training.
* **TableDelegate**: delegate for the parameter table physically stored in parameter servers.
    it runs a thread to communicate with table servers for parameter transferring.
* **Net**: the neural network consists of multiple layers constructed from input configuration file.
* **Layer**: the core abstraction, read data (neurons) from connecting layers, and compute the data
    of itself according to layer specific ComputeFeature functions. Data from the bottom layer is forwarded
    layer by layer to the top.

#### Data types

<img src="../images/code-structure/layer.jpg" style="width: 700px"/>

* **ComputeFeature**: read data (neurons) from in-coming layers, and compute the data
    of itself according to layer type. This function can be overrided to implement different
    types layers.
* **ComputeGradient**: read gradients (and data) from in-coming layers and compute
    gradients of parameters and data w.r.t the learning objective (loss).

We adpat the implementation for **PoolingLayer**, **Im2colLayer** and **LRNLayer** from [Caffe](http://caffe.berkeleyvision.org/).


<img src="../images/code-structure/darray.jpg" style="width: 400px"/>

* **DArray**: provide the abstraction of distributed array on multiple nodes,
    supporting array/matrix operations and element-wise operations. Users can use it as a local structure.
* **LArray**: the local part for the DArray. Each LArray is treated as an
    independent array, and support all array-related operations.
* **MemSpace**: manage the memory used by DArray. Distributed memory are allocated
    and managed by armci. Multiple DArray can share a same MemSpace, the memory
    will be released when no DArray uses it anymore.
* **Partition**: maintain both global shape and local partition information.
    used when two DArray are going to interact.
* **Shape**: basic class for representing the scope of a DArray/LArray
* **Range**: basic class for representing the scope of a Partition

### Parameter Server

#### Main classes

<img src="../images/code-structure/uml.jpg" style="width: 750px"/>

* **NetworkService**: provide access to the network (sending and receiving messages). It maintains a queue for received messages, implemented by NetworkQueue.
* **RequestDispatcher**: pick up next message (request) from the queue, and invoked a method (callback) to process them.
* **TableServer**: provide access to the data table (parameters). Register callbacks for different types of requests to RequestDispatcher.
* **GlobalTable**: implement the table. Data is partitioned into multiple Shard objects per table. User-defined consistency model supported by extending TableServerHandler for each table.

#### Data types

<img src="../images/code-structure/type.jpg" style="width: 400px"/>

Table related messages are either of type **RequestBase** which contains different types of request, or of type **TableData** containing a key-value tuple.

#### Control flow and thread model

<img src="../images/code-structure/threads.jpg" alt="uml" style="width: 1000px"/>

The figure above shows how a GET request sent from a worker is processed by the
table server. The control flow for other types of requests is similar. At
the server side, there are at least 3 threads running at any time: two by
NetworkService for sending and receiving message, and at least one by the
RequestDispatcher for dispatching requests.

-->
