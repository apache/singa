# 분산 트레이닝

---

## Cluster Topology 설정

SINGA 에서 다양한 분산 트레이닝 프레임워크를 실행하는 방법을 설명합니다.

cluster topology 는 `JobProto` 속의 `cluster` field 를 설정해줍니다.
`cluster` 는 `ClusterProto` 타입 입니다. 예를 들어

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


자주 사용되는 field 는 다음과 같습니다:

* `nworkers_per_group` and `nworkers_per_procs`:
    decide the partitioning of worker side ParamShard.

* `nservers_per_group` and `nservers_per_procs`:
    decide the partitioning of server side ParamShard.

* `server_worker_separate`:
    separate servers and workers in different processes.

## 다양한 트레이닝 프레임워크

SINGA 에서 worker groups 들은 비동기적으로, group 속에서 workers 들은 동기적으로 실행됩니다. 유저는 이 일반디저인을 이용해서 **synchronous** 와 **asynchronous** 트레이닝 프레임워크를 실행 할수 있습니다. 널리 알려진 분산 트레이닝을 어떻게 설정하고 실행하는지 설명하겠습니다.

<img src="../_static/images/frameworks.png" style="width: 800px"/>
<p><strong> Fig.1 - 다양한 트레이닝 프레임워크</strong></p>

###Sandblaster

Google Brain 에서 쓰이는 **synchronous** 프레임워크.
Fig.2(a) 는 SINGA에서 Sandblaster 프레임워크를 실행하기 위한 cluster 의 설정 예입니다.

    cluster {
        nworker_groups: 1
        nserver_groups: 1
        nworkers_per_group: 3
        nservers_per_group: 2
        server_worker_separate: true
    }

각 server group 는 모든 workers 의 requests 를 처리합니다.
각 worker 는 뉴럴네트 모델의 한 부분을 담당하여 계산을 하고, 모든 servers 와 통신을 하여 관련 parameters 값을 얻습니다.


###AllReduce

Baidu's DeepImage 에서 쓰이는 **synchronous** 프레임워크.
Fig.2(b) 는 SINGA에서 AllReduce 프레임워크를 실행하기 위한 cluster 의 설정 예입니다.

    cluster {
        nworker_groups: 1
        nserver_groups: 1
        nworkers_per_group: 3
        nservers_per_group: 3
        server_worker_separate: false
    }

각 node 에서 1 worker 와 1 server 를 실행하여, 각 node 가 parameters 의 한 부분을 담당하고 계산을 하도록 설정합니다. 다른 nodes 와 업데이트 된 정보를 교환합니다.

###Downpour

Google Brain 에서 쓰이는 **asynchronous** 프레임워크.
Fig.2(c) 는 SINGA에서 Downpour 프레임워크를 실행하기 위한 cluster 의 설정 예입니다.

    cluster {
        nworker_groups: 2
        nserver_groups: 1
        nworkers_per_group: 2
        nservers_per_group: 2
        server_worker_separate: true
    }

synchronous Sandblaster 와 비슷하게, 모든 workers 는 1 server group 에 requests 를 보냅니다. 여기서는 workers 들을 여러 worker groups 으로 나누어서, 각 worker 가 *update* reply 에서 받은 최신 parameters 를 써서 계산 하도록 설정하였습니다.

###Distributed Hogwild

Caffe 에서 쓰이는 **asynchronous** 프레임워크.
Fig.2(d) 는 SINGA에서 Hogwild 프레임워크를 실행하기 위한 cluster 의 설정 예입니다.

    cluster {
        nworker_groups: 3
        nserver_groups: 3
        nworkers_per_group: 1
        nservers_per_group: 1
        server_worker_separate: false
    }

각 node 는 1 server group 와 1 worker group 를 실행합니다.
Parameter updates 를 node 에서 각각 실행시킴으로써 통신코스트와 트레이닝 스텝을 최소화 합니다. 그러나 server groups 들은 정기적으로 네이버링 groups 들과 동기 시켜야 됩니다.
