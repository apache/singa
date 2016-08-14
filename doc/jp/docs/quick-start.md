# クイック スタート

---

## SINGA セットアップ

SINGAのインストールについては[こちら](installation.html)をご覧ください。

### Zookeeper の実行

SINGAのトレーニングは　[zookeeper](https://zookeeper.apache.org/) を利用します。まずは zookeeper サービスが開始されていることを確認してください。

準備された thirdparty のスクリプトを使って zookeeper をインストールした場合、次のスクリプトを実行してください。

    #goto top level folder
    cd  SINGA_ROOT
    ./bin/zk-service.sh start

(`./bin/zk-service.sh stop` // zookeeper の停止).

デフォルトのポートを使用せずに zookeeper をスタートさせる時は、`conf/singa.conf`を編集してください。

    zookeeper_host: "localhost:YOUR_PORT"

## スタンドアローンモードでの実行

スタンドアローンモードでSINGAを実行するとは、[Mesos](http://mesos.apache.org/) や [YARN](http://hadoop.apache.org/docs/current/hadoop-yarn/hadoop-yarn-site/YARN.html) のようなクラスターマネージャー利用しない場合のことを言います。

### Single ノードでのトレーニング

１つのプロセスがローンチされます。
例として、
[CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html) データセットを利用して
[CNN モデル](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) をトレーニングさせます。
ハイパーパラメーターは、[cuda-convnet](https://code.google.com/p/cuda-convnet/) に基づいて設定されてあります。
詳細は、[CNN サンプル](cnn.html) のページをご覧ください。


#### データと、ジョブ設定

データセットのダウンロードと、Triaing や Test のためのデータシャードの生成は次のように行います。

    cd examples/cifar10/
    cp Makefile.example Makefile
    make download
    make create

Training と Test データセットは、それぞれ *cifar10-train-shard*
と *cifar10-test-shard* フォルダーに作られます。　すべての画像の特徴平均を記述した *image_mean.bin* ファイルも作成されます。

CNN モデルのトレーニングに必要なソースコードはすべてSINGAに組み込まれています。コードを追加する必要はありません。
ジョブ設定ファイル (*job.conf*) を指定して、スクリプト(*../../bin/singa-run.sh*) を実行します。
SINGAのコードを変更、または追加する時は、[プログラミングガイド](programming-guide.html)をご覧ください。

#### 並列化なしのトレーニング

Cluster Topology のデフォルト値は、１つの worker と　１つの server となっています。
データとニューラルネットの並列化はされません。

トレーニングを開始するには次のスクリプトを実行します。

    # goto top level folder
    cd ../../
    ./bin/singa-run.sh -conf examples/cifar10/job.conf


現在、起動中のジョブのリストを表示するには

    ./bin/singa-console.sh list

    JOB ID    |NUM PROCS
    ----------|-----------
    24        |1

ジョブの強制終了をするには

    ./bin/singa-console.sh kill JOB_ID


ログとジョブの情報は */tmp/singa-log* フォルダーに保存されます。
*conf/singa.conf* ファイルの `log-dir`で変更可能です。


#### 非同期、並列トレーニング

    # job.conf
    ...
    cluster {
      nworker_groups: 2
      nworkers_per_procs: 2
      workspace: "examples/cifar10/"
    }

複数の worker グループをローンチすることによって、
In SINGA, [非同期トレーニング](architecture.html) を実行することが出来ます。
例えば、*job.conf* を上記のように変更します。
デフォルトでは、１つの worker グループが１つの worker を持つよう設定されています。
上記の設定では、１つのプロセスに２つの worker が設定されているので、２つの worker グループが同じプロセスとして実行されます。
結果、インメモリ [Downpour](frameworks.html) トレーニングフレームワークとして、実行されます。

ユーザーは、データの分散を気にする必要はありません。
ランダムオフセットに従い、各 worker グループに、データが振り分けられます。
各 worker は異なるデータパーティションを担当します。

    # job.conf
    ...
    neuralnet {
      layer {
        ...
        sharddata_conf {
          random_skip: 5000
        }
      }
      ...
    }

スクリプト実行:

    ./bin/singa-run.sh -conf examples/cifar10/job.conf

#### 同期、並列トレーニング

    # job.conf
    ...
    cluster {
      nworkers_per_group: 2
      nworkers_per_procs: 2
      workspace: "examples/cifar10/"
    }

１つのworkerグループとして複数のworkerをローンチすることで [同期トレーニング](architecture.html)を実行することが出来ます。
例えば、*job.conf* ファイルを上記のように変更します。
上記の設定では、１つの worker グループに２つの worker が設定されました。
worker 達はグループ内で同期します。
これは、インメモリ [sandblaster](frameworks.html) として実行されます。
モデルは２つのworkerに分割されます。各レイヤーが２つのworkerに振り分けられます。
振り分けられたレイヤーはオリジナルのレイヤーと機能は同じですが、特徴インスタンスの数が `B/g` になります。
ここで、`B`はミニバッチのインスタンスの数で、`g`はグループ内の worker の数です。
[別のスキーム](neural-net.html) を利用したレイヤー（ニューラルネットワーク）パーティション方法もあります。

他の設定はすべて「並列化なし」の場合と同じです。

    ./bin/singa-run.sh -conf examples/cifar10/job.conf

### クラスタ上でのトレーニング

クラスター設定を変更して、上記トレーニングフレームワークの拡張を行います。

    nworker_per_procs: 1

すべてのプロセスは１つのworkerスレッドを生成します。
結果、worker 達は異なるプロセス（ノード）内で生成されます。
クラスター内のノードを特定するには、*SINGA_ROOT/conf/* の *hostfile* の設定が必要です。

e.g.,

    logbase-a01
    logbase-a02

zookeeper location も設定する必要があります。

e.g.,

    #conf/singa.conf
    zookeeper_host: "logbase-a01"

スクリプトの実行は「Single ノード トレーニング」と同じです。

    ./bin/singa-run.sh -conf examples/cifar10/job.conf

## Mesos　での実行

*working*...

## 次へ

SINGAのコード変更や追加に関する詳細は、[プログラミングガイド](programming-guide.html) をご覧ください。
