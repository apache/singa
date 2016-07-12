# 퀵 스타트

---

## SINGA 인스톨

SINGA 인스톨은 [여기](installation.html)를 참조하십시오.

### Zookeeper 실행

SINGA 트레이닝은 [zookeeper](https://zookeeper.apache.org/)를 이용합니다. 우선 zookeeper 서비스가 시작되어 있는지 확인하십시오.

준비된 thirdparty 스크립트를 사용하여 zookeeper를 설치 한 경우 다음 스크립트를 실행하십시오.

    #goto top level folder
    cd SINGA_ROOT
    ./bin/zk-service.sh start

(`./bin/zk-service.sh stop` // zookeeper 중지).

기본 포트를 사용하지 않고 zookeeper를 시작시킬 때는 `conf/singa.conf`을 편집하십시오.

    zookeeper_host : "localhost : YOUR_PORT"

## Stand-alone 모드에서 실행

Stand-alone 모드에서 SINGA을 실행할 때, [Mesos](http://mesos.apache.org/) 와 [YARN](http://hadoop.apache.org/docs/current/hadoop-yarn/hadoop-yarn-site/YARN.html) 과 같은 클러스터 관리툴을 이용하지 않는 경우를 말합니다.

### Single 노드에서의 트레이닝

하나의 프로세스가 시작됩니다.
예를 들어,
[CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html) 데이터 세트를 이용하여
[CNN 모델](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)을 트레이닝 시킵니다.
하이퍼 파라미터는 [cuda-convnet](https://code.google.com/p/cuda-convnet/)에 따라 설정되어 있습니다.
자세한 내용은 [CNN 샘플](cnn.html) 페이지를 참조하십시오.


#### 데이터와 작업 설정

데이터 세트 다운로드와 Triaing 이나 Test 를 위한 데이터 샤드의 생성은 다음과 같이 실시합니다.

    cd examples/cifar10/
    cp Makefile.example Makefile
    make download
    make create

Training 과 Test 데이터 세트는 각각 *cifar10-train-shard*
그리고 *cifar10-test-shard* 폴더에 만들어집니다. 모든 이미지의 특징 평균을 기술한 *image_mean.bin* 파일도 함께 생성됩니다.

CNN 모델 트레이닝에 필요한 소스코드는 모두 SINGA에 포함되어 있습니다. 코드를 추가 할 필요는 없습니다.
작업 설정 파일(*job.conf*) 을 지정하여 스크립트(*../../bin/singa-run.sh*)를 실행합니다.
SINGA 코드를 변경하거나 추가 할 경우는, 프로그래밍가이드 (programming-guide.html)를 참조하십시오.

#### 병렬화 없이 트레이닝

Cluster Topology의 기본값은 하나의 worker와 하나의 server가 있습니다.
데이터와 모델의 병렬 처리는 되지 않습니다.

트레이닝을 시작하기 위하여 다음 스크립트를 실행합니다.

    # goto top level folder
    cd ../../
    ./bin/singa-run.sh -conf examples/cifar10/job.conf


현재 실행중인 작업의 리스트를 보려면

    ./bin/singa-console.sh list

    JOB ID     | NUM PROCS
    ---------- | -----------
    24         | 1

작업을 종료하려면

    ./bin/singa-console.sh kill JOB_ID


로그 및 작업 정보는 */tmp/singa-log* 폴더에 저장됩니다.
*conf/singa.conf* 파일의 `log-dir`에서 변경 가능합니다.


#### 비동기 병렬 트레이닝

    # job.conf
    ...
    cluster {
      nworker_groups : 2
      nworkers_per_procs : 2
      workspace : "examples/cifar10/"
    }

여러 worker 그룹을 실행함으로써 [비동기 트레이닝](architecture.html)을 할 수 있습니다.
예를 들어, *job.conf* 을 위와 같이 변경합니다.
기본적으로 하나의 worker 그룹이 하나의 worker를 갖도록 설정되어 있습니다.
위의 설정은 하나의 프로세스에 2개의 worker가 설정되어 있기 때문에 2개의 worker 그룹이 동일한 프로세스로 실행됩니다.
결과 인메모리 [Downpour](frameworks.html) 트레이닝 프레임워크로 실행됩니다.

사용자는 데이터의 분산을 신경 쓸 필요는 없습니다.
랜덤 오프셋에 따라 각 worker 그룹에 데이터가 보내집니다.
각 worker는 다른 데이터 파티션을 담당합니다.

    # job.conf
    ...
    neuralnet {
      layer {
        ...
        sharddata_conf {
          random_skip : 5000
        }
      }
      ...
    }

스크립트 실행 :

    ./bin/singa-run.sh -conf examples/cifar10/job.conf

#### 동기화 병렬 트레이닝

    # job.conf
    ...
    cluster {
      nworkers_per_group : 2
      nworkers_per_procs : 2
      workspace : "examples/cifar10/"
    }

하나의 worker 그룹으로 여러 worker를 실행하여 [동기 트레이닝](architecture.html)을 수행 할 수 있습니다.
예를 들어, *job.conf* 파일을 위와 같이 변경합니다.
위의 설정은 하나의 worker 그룹에 2개의 worker가 설정되었습니다.
worker 들은 그룹 내에서 동기화합니다.
이것은 인메모리 [sandblaster](frameworks.html)로 실행됩니다.
모델은 2개의 worker로 분할됩니다. 각 레이어가 2개의 worker로 분산됩니다.
배분 된 레이어는 원본 레이어와 기능은 같지만 특징 인스턴스의 수가 `B / g` 로 됩니다.
여기서 `B`는 미니밧치 인스턴스의 숫자로 `g`는 그룹의 worker 수 입니다.
[다른 스킴](neural-net.html)을 이용한 레이어 (뉴럴네트워크) 파티션 방법도 있습니다.

다른 설정들은 모두 "병렬화 없음"의 경우와 동일합니다.

    ./bin/singa-run.sh -conf examples/cifar10/job.conf

### 클러스터에서의 트레이닝

클러스터 설정을 변경하여 위 트레이닝 프레임워크를 확장합니다.

    nworker_per_procs : 1

모든 프로세스는 하나의 worker 스레드를 생성합니다.
결과 worker 우리는 다른 프로세스 (노드)에서 생성됩니다.
클러스터의 노드를 특정하려면 *SINGA_ROOT/conf/* 의 *hostfile* 의 설​​정이 필요합니다.

e.g.,

    logbase-a01
    logbase-a02

zookeeper location도 설정해야합니다.

e.g.,

    # conf/singa.conf
    zookeeper_host : "logbase-a01"

스크립트의 실행은 "Single 노드 트레이닝"과 동일합니다.

    ./bin/singa-run.sh -conf examples/cifar10/job.conf

## Mesos에서 실행

*working* ...

## 다음

SINGA 의 코드 변경 및 추가에 대한 자세한 내용은 [프로그래밍 가이드](programming-guide.html)를 참조하십시오.
