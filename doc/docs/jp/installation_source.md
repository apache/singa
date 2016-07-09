# ソースからのビルド

---

## Dependencies （依存性）

SINGA は Linux プラットフォームで開発されました。

下記のライブラリを必要とします。ご確認ください。

  * glog version 0.3.3

  * google-protobuf version 2.6.0

  * openblas version >= 0.2.10

  * zeromq version >= 3.2

  * czmq version >= 3

  * zookeeper version 3.4.6


オプション

  * lmdb version 0.9.10


すべての Dependencies をインストールするには

    # thirdparty フォルダに移動
    cd thirdparty
    ./install.sh all $PREFIX

$PREFIX がシステムパス (例 /usr/local/) でない場合は、下記の変数を exportコマンドで設定してください。

    export LD_LIBRARY_PATH=$PREFIX/lib:$LD_LIBRARY_PATH
    export CPLUS_INCLUDE_PATH=$PREFIX/include:$CPLUS_INCLUDE_PATH
    export LIBRARY_PATH=$PREFIX/lib:$LIBRARY_PATH
    export PATH=$PREFIX/bin:$PATH

インストールスクリプトの詳細は、このページの下にあります。

## ソースからのビルド

GNU autotools を使用します。まずは GCC (version >= 4.8) のバージョンを確認してください。

SINGA をビルドする２つの方法を用意しました。

  * Git コマンドを使用して [Github](https://github.com/apache/incubator-singa.git) から最新のソースコード（レポジトリ）をクローンし、次のコマンドを実行します。

        $ git clone git@github.com:apache/incubator-singa.git
        $ cd incubator-singa
        $ ./autogen.sh
        $ ./configure
        $ make

  Note: [nusinga](https://github.com/orgs/nusinga) アカウントの singa repository は Apache Incubator project として使用していたものなので、最新ではありません。ご注意ください。

  * Release パッケージをダウンロードし、次のコマンドを実行します。

        $ tar xvf singa-xxx
        $ cd singa-xxx
        $ ./configure
        $ make

    ある機能は外部ライブラリに依存します。
    それらの機能を利用するために、`--enable-<feature>` を付けてコンパイルしてください。
    例えば、lmdb サポートを追加するには

        $ ./configure --enable-lmdb

<!---
Zhongle: please update the code to use the follow command

    $ make test

After compilation, you will find the binary file singatest. Just run it!
More details about configure script can be found by running:

		$ ./configure -h
-->

うまくコンパイルが成功すると、*.libs/* フォルダ内に *libsinga.so* と実行ファイル *singa* が生成されます。

特定の dependent ライブラリが見つからない場合、次のスクリプトを実行してください。
<!---
to be updated after zhongle changes the code to use

    ./install.sh libname \-\-prefix=

-->
    # thirdparty フォルダに移動
    $ cd thirdparty
    $ ./install.sh LIB_NAME PREFIX

インストールパスを指定しない場合、ライブラリは（ソフトウェアが指定した）デフォルトのフォルダにインストールされます。
例えば、デフォルトのフォルダに `zeromq` ライブラリをインストールするには、次のコマンドを

    $ ./install.sh zeromq

別のフォルダ（e.g., PREFIX）にインストールするには、次のコマンドを

    $ ./install.sh zeromq PREFIX

*/usr/local*　フォルダにすべての Dependencies をインストールするには、次のコマンドを

    $ ./install.sh all /usr/local

ライブラリ名（LIB_NAME）は以下のとおりです。

    LIB_NAME          Library
    czmq（注）       　czmq lib
    glog              glog lib
    lmdb              lmdb lib
    OpenBLAS          OpenBLAS lib
    protobuf          Google protobuf
    zeromq            zeromq lib
    zookeeper         Apache zookeeper

(注) `czmq` をインストールする時、`zeromq` のパスを -f オプションで指定する必要があります。
コマンドは次のとおりです。

<!---
to be updated to

    $./install.sh czmq  \-\-prefix=/usr/local \-\-zeromq=/usr/local/zeromq
-->

    $./install.sh czmq  /usr/local -f=/usr/local/zeromq

結果、*/usr/local* フォルダに `czmq` がインストールされます。

### FAQ
* Q1:I get error `./configure --> cannot find blas_segmm() function` even I
have installed OpenBLAS.

  A1: This means the compiler cannot find the `OpenBLAS` library. If you installed
  it to $PREFIX (e.g., /opt/OpenBLAS), then you need to export it as

      $ export LIBRARY_PATH=$PREFIX/lib:$LIBRARY_PATH
      # e.g.,
      $ export LIBRARY_PATH=/opt/OpenBLAS/lib:$LIBRARY_PATH


* Q2: I get error `cblas.h no such file or directory exists`.

  Q2: You need to include the folder of the cblas.h into CPLUS_INCLUDE_PATH,
  e.g.,

      $ export CPLUS_INCLUDE_PATH=$PREFIX/include:$CPLUS_INCLUDE_PATH
      # e.g.,
      $ export CPLUS_INCLUDE_PATH=/opt/OpenBLAS/include:$CPLUS_INCLUDE_PATH
      # then reconfigure and make SINGA
      $ ./configure
      $ make


* Q3:While compiling SINGA, I get error `SSE2 instruction set not enabled`

  A3:You can try following command:

      $ make CFLAGS='-msse2' CXXFLAGS='-msse2'


* Q4:I get `ImportError: cannot import name enum_type_wrapper` from
google.protobuf.internal when I try to import .py files.

  A4:After install google protobuf by `make install`, we should install python
  runtime libraries. Go to protobuf source directory, run:

      $ cd /PROTOBUF/SOURCE/FOLDER
      $ cd python
      $ python setup.py build
      $ python setup.py install

  You may need `sudo` when you try to install python runtime libraries in
  the system folder.


* Q5: I get a linking error caused by gflags.

  A5: SINGA does not depend on gflags. But you may have installed the glog with
  gflags. In that case you can reinstall glog using *thirdparty/install.sh* into
  a another folder and export the LDFLAGS and CPPFLAGS to include that folder.


* Q6: While compiling SINGA and installing `glog` on mac OS X, I get fatal error
`'ext/slist' file not found`

  A6:Please install `glog` individually and try :

      $ make CFLAGS='-stdlib=libstdc++' CXXFLAGS='stdlib=libstdc++'

* Q7: When I start a training job, it reports error related with "ZOO_ERROR...zk retcode=-4...".

  A7: This is because the zookeeper is not started. Please start the zookeeper service

      $ ./bin/zk-service start

  If the error still exists, probably that you do not have java. You can simple
  check it by

      $ java --version

* Q8: When I build OpenBLAS from source, I am told that I need a fortran compiler.

  A8: You can compile OpenBLAS by

      $ make ONLY_CBLAS=1

  or install it using

	    $ sudo apt-get install openblas-dev

  or

	    $ sudo yum install openblas-devel

  It is worth noting that you need root access to run the last two commands.
  Remember to set the environment variables to include the header and library
  paths of OpenBLAS after installation (please refer to the Dependencies section).

* Q9: When I build protocol buffer, it reports that GLIBC++_3.4.20 not found in /usr/lib64/libstdc++.so.6.

  A9: This means the linker found libstdc++.so.6 but that library
  belongs to an older version of GCC than was used to compile and link the
  program. The program depends on code defined in
  the newer libstdc++ that belongs to the newer version of GCC, so the linker
  must be told how to find the newer libstdc++ shared library.
  The simplest way to fix this is to find the correct libstdc++ and export it to
  LD_LIBRARY_PATH. For example, if GLIBC++_3.4.20 is listed in the output of the
  following command,

      $ strings /usr/local/lib64/libstdc++.so.6|grep GLIBC++

  then you just set your environment variable as

      $ export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH
