# SINGA Architecture

---

## Logical Architecture

<img src="../_static/images/logical.png" style="width: 550px"/>
<p><strong> Fig.1 - Logical system architecture</strong></p>

SINGA has flexible architecture to support different distributed
[training frameworks](frameworks.html) (both synchronous and asynchronous).
The logical system architecture is shown in Fig.1.
The architecture consists of multiple server groups and worker groups:

* **Server group**
  A server group maintains a complete replica of the model parameters,
  and is responsible for handling get/update requests from worker groups.
  Neighboring server groups synchronize their parameters periodically.
  Typically, a server group contains a number of servers,
  and each server manages a partition of model parameters.
* **Worker group**
  Each worker group communicates with only one server group.
  A worker group trains a complete model replica
  against a partition of the training dataset,
  and is responsible for computing parameter gradients.
  All worker groups run and communicate with the corresponding
  server groups asynchronously.
  However, inside each worker group,
  the workers synchronously compute parameter updates for the model replica.

There are different strategies to distribute the training workload among workers
within a group:

  * **Model parallelism**. Each worker computes a subset of parameters
  against all data partitioned to the group.
  * **Data parallelism**. Each worker computes all parameters
  against a subset of data.
  * [**Hybrid parallelism**](hybrid.html). SINGA also supports hybrid parallelism.


## Implementation
In SINGA, servers and workers are execution units running in separate threads.
They communicate through [messages](communication.html).
Every process runs the main thread as a stub that aggregates local messages
and forwards them to corresponding (remote) receivers.

Each server group and worker group have a *ParamShard*
object representing a complete model replica. If workers and servers
resident in the same process, their *ParamShard* (partitions) can
be configured to share the same memory space. In this case, the
messages transferred between different execution units just contain
pointers to the data, which reduces the communication cost.
Unlike in inter-process cases,
the messages have to include the parameter values.
