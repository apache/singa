# 프로그래밍 가이드

---

Figure 1에 그려진 다음과 같은 4가지 Components 를 설정하여 트레이닝을 시작합니다.

  * [NeuralNet](neural-net.html) : 뉴럴네트워크의 구조와 각 "레이어"의 설정을 기술합니다.
  * [TrainOneBatch](train-one-batch.html) : 모델 카테고리에 적합한 알고리즘을 기술합니다.
  * [Updater](updater.html) : server에서 매개 변수를 업데이트하는 방법을 기술합니다.
  * [Cluster Topology](distributed-training.html) : workers와 servers 분산 토폴로지를 기술합니다.

*Basic 유저 가이드* 에서 built-in components 를 써서 트레이닝을 시작하는 방법을 설명합니다. *Advanced 유저 가이드* 에서는 유저가 임플리멘트한 모델, 함수, 알고리듬을 써서 트레이닝을 시작하는 방법을 설병합니다. 트레이닝 데이타는 [process](data.html) 를 참고로 준비를 해주세요.

<img src="../_static/images/overview.png" align="center" width="400px"/>
<span><strong>Figure 1 - SINGA Overview </strong></span>



## Basic 유저 가이드

SINGA 에서 준비된 main 함수를 써서 쉽게 트레이닝을 시작할수 있습니다.
이 경우 [JobProto](../api/classsinga_1_1JobProto.html) 를 위하여 google protocol buffer message 로 씌여진 job configuration 파일을 준비합니다. 그리고 아래의 커맨드라인을 실행합니다.

    ./bin/singa-run.sh -conf <path to job conf> [-resume]

`-resume` 는 저번 [checkpoint](checkpoint.html) 부터 다시 트레이닝을 계속할때 쓰는 인수 입니다.
[MLP](mlp.html) 와 [CNN](cnn.html) 샘플들은 built-in 컴포넌트를 이용하고 있습니다.
Please read the corresponding pages for their job configuration files. The subsequent pages will illustrate the details on each component of the configuration.

## Advanced 유저 가이드

If a user's model contains some user-defined components, e.g.,
[Updater](updater.html), he has to write a main function to
register these components. It is similar to Hadoop's main function. Generally,
the main function should

* SINGA 초기화, e.g., setup logging.

* 유저 컴포넌트의 등록

* job configuration 을 작성하고 SINGA driver 에서 설정

main 함수의 샘플입니다.

    #include "singa.h"
    #include "user.h"  // header for user code

    int main(int argc, char** argv) {
      singa::Driver driver;
      driver.Init(argc, argv);
      bool resume;
      // parse resume option from argv.

      // register user defined layers
      driver.RegisterLayer<FooLayer>(kFooLayer);
      // register user defined updater
      driver.RegisterUpdater<FooUpdater>(kFooUpdater);
      ...
      auto jobConf = driver.job_conf();
      //  update jobConf

      driver.Train(resume, jobConf);
      return 0;
    }

Driver class' `Init` method 는 커맨드라인 인수 `-conf <job conf>` 에서 주어진 job configuration 파일을 읽습니다. 그 파일에는 cluster topology 정보가 기술 되어있고, 유저가 neural net, updater 등을 업데이트 혹은 설정 하기위한 `jobConf`를 리턴합니다.
유저가 Layer, Updater, Worker, Param 등의 subclass를 정의하면, driver 에 등록을 해야합니다.
트레이닝을 시작하기 위하여 job configuration 즉 `jobConf`를 driver.Train 에 넘겨줍니다.

<!--We will provide helper functions to make the configuration easier in the
future, like [keras](https://github.com/fchollet/keras).-->

유저코드를 compile 하고 SINGA library (*.libs/libsinga.so*) 와 링크시켜 실행파일, e.g., *mysinga*, 을 생성합니다. 프로그램은 다음과 같이 실행합니다.

    ./bin/singa-run.sh -conf <path to job conf> -exec <path to mysinga> [other arguments]

[RNN application](rnn.html) 에서 RNN 모델의 트레이닝을 위한 함수의 프로그램 예를 설명합니다.
