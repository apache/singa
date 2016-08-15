# 分散トレーニング フレームワーク

---

## クラスタトポロジーの設定

クラスタトポロジーは `JobProto` の `cluster` フィールドで設定します。
`cluster` の type は `ClusterProto` です。

    message ClusterProto {
      optional int32 nworker_groups = 1;
      optional int32 nserver_groups = 2;
      optional int32 nworkers_per_group = 3 [default = 1];
      optional int32 nservers_per_group = 4 [default = 1];
      optional int32 nworkers_per_procs = 5 [default = 1];
      optional int32 nservers_per_procs = 6 [default = 1];

      // servers and workers in different processes?
      optional bool server_worker_separate = 20 [default = false];

      ......
    }


下記のフィールドを設定して、クラスタトポロジーをカスタマイズします。

  * `nworkers_per_group` と `nworkers_per_procs`
  decide the partitioning of worker side ParamShard.

  * `nservers_per_group` と `nservers_per_procs`
  decide the partitioning of server side ParamShard.

  * `server_worker_separate`
  separate servers and workers in different processes.

## トレーニング フレームワーク

In SINGA, worker groups run asynchronously and　workers within one group run synchronously.
Users can leverage this general design to run　both **synchronous** and **asynchronous** training frameworks.
Here we illustrate how to configure　popular distributed training frameworks in SINGA.

<img src="../_static/images/frameworks.png" style="width: 800px"/>
<p><strong> Fig.1 - Training frameworks in SINGA</strong></p>

### Sandblaster

Google Brain で使われている **synchronous** フレームワークです。
Fig.2(a)に、SINGA で実装された Sandblaster を示します。
cluster フィールドの設定は次のとおりです。

    cluster {
        nworker_groups: 1
        nserver_groups: 1
        nworkers_per_group: 3
        nservers_per_group: 2
        server_worker_separate: true
    }

A single server group is launched to handle all requests from workers.
A worker computes on its partition of the model,
and only communicates with servers handling related parameters.


### AllReduce

Baidu's DeepImage で使われている **synchronous** フレームワークです。
Fig.2(b)に、SINGA で実装された AllReduce を示します。
cluster フィールドの設定は次のとおりです。

    cluster {
        nworker_groups: 1
        nserver_groups: 1
        nworkers_per_group: 3
        nservers_per_group: 3
        server_worker_separate: false
    }

We bind each worker with a server on the same node, so that each
node is responsible for maintaining a partition of parameters and
collecting updates from all other nodes.

### Downpour

Google Brain で使われている **asynchronous** フレームワークです。
Fig.2(c)に、SINGA で実装された Downpour を示します。
cluster フィールドの設定は次のとおりです。

    cluster {
        nworker_groups: 2
        nserver_groups: 1
        nworkers_per_group: 2
        nservers_per_group: 2
        server_worker_separate: true
    }

Similar to the synchronous Sandblaster, all workers send
requests to a global server group. We divide workers into several
worker groups, each running independently and working on parameters
from the last *update* response.

###Distributed Hogwild

Caffe で使われている **asynchronous** フレームワークです。
Fig.2(d)に、SINGA で実装された Distributed Hogwild を示します。
cluster フィールドの設定は次のとおりです。

    cluster {
        nworker_groups: 3
        nserver_groups: 3
        nworkers_per_group: 1
        nservers_per_group: 1
        server_worker_separate: false
    }

Each node contains a complete server group and a complete worker group.
Parameter updates are done locally, so that communication cost
during each training step is minimized.
However, the server group must periodically synchronize with
neighboring groups to improve the training convergence.
