# 从源程序安装SIGNA

---

## 依赖

SINGA 在Linux平台上开发与测试。安装SINGA需要下拉列依赖库：

  * glog version 0.3.3

  * google-protobuf version 2.6.0

  * openblas version >= 0.2.10

  * zeromq version >= 3.2

  * czmq version >= 3

  * zookeeper version 3.4.6


可选依赖包括：

  * lmdb version 0.9.10


你可以使用下列命令将所有的依赖库安装到$PREFIX文件夹下：

    # make sure you are in the thirdparty folder
    cd thirdparty
    ./install.sh all $PREFIX

如果$PREFIX不是一个系统路径（如：/esr/local/），请在继续安装前使用下述命令导出相关变量：

    export LD_LIBRARY_PATH=$PREFIX/lib:$LD_LIBRARY_PATH
    export CPLUS_INCLUDE_PATH=$PREFIX/include:$CPLUS_INCLUDE_PATH
    export LIBRARY_PATH=$PREFIX/lib:$LIBRARY_PATH
    export PATH=$PREFIX/bin:$PATH

关于使用这个脚本的细节后文会详细介绍。

## 从源程序安装SINGA

SINGA 使用 GNU autotools 构建，需要GCC (version >= 4.8)。
有两种方式安装SINGA。

  * 如果你想使用最近的代码，请执行以下命令从 [Github](https://github.com/apache/incubator-singa.git) 上克隆：

        $ git clone git@github.com:apache/incubator-singa.git
        $ cd incubator-singa
        $ ./autogen.sh
        $ ./configure
        $ make

  注意: 由于我们的疏忽，在加入Apache Incubator项目后，[nusinga](https://github.com/orgs/nusinga) 帐号下的SINGA库（repo）并没有删除，但它早已没有更新，很抱歉给大家带来的不便。

  * 如果你下载了发布包，请按以下命令安装：

        $ tar xvf singa-xxx
        $ cd singa-xxx
        $ ./configure
        $ make

    SINGA的部分特性依赖于外部库，这些特性可以使用`--enable-<feature>`编译。
    比如，按准跟支持lmdb的SINGA，可以运行下面的命令：

        $ ./configure --enable-lmdb

<!---
Zhongle: please update the code to use the follow command

    $ make test

After compilation, you will find the binary file singatest. Just run it!
More details about configure script can be found by running:

		$ ./configure -h
-->

SINGA编译成功后， *libsinga.so* 和可执行文件 *singa* 会生成在 *.libs/* 文件夹下。

如果缺失（或没有检测到）某些依赖库，可使用下面的脚本下载和安装：

<!---
to be updated after zhongle changes the code to use

    ./install.sh libname \-\-prefix=

-->
    # must goto thirdparty folder
    $ cd thirdparty
    $ ./install.sh LIB_NAME PREFIX

如果没有指定安装路径，这些库会被安装在这些软件默认的安装路径下。比如，如果想在默认系统文件夹下安装`zeromq`，请执行以下命令：

    $ ./install.sh zeromq

或者，如果想安装到其他目录：

    $ ./install.sh zeromq PREFIX

也可以将所有的依赖库安装到 */usr/local* 目录:

    $ ./install.sh all /usr/local

下表展示了各依赖库的第一个参数：

    LIB_NAME  LIBRARIE
    czmq*                 czmq lib
    glog                  glog lib
    lmdb                  lmdb lib
    OpenBLAS              OpenBLAS lib
    protobuf              Google protobuf
    zeromq                zeromq lib
    zookeeper             Apache zookeeper

*: 因为 `czmq` 依赖于 `zeromq`，下述脚本多提供一个参数，说明 `zeromq` 的位置。
`czmq` 的安装命令是：

<!---
to be updated to

    $./install.sh czmq  \-\-prefix=/usr/local \-\-zeromq=/usr/local/zeromq
-->

    $./install.sh czmq  /usr/local -f=/usr/local/zeromq

执行后，`czmq` 会被安装在 */usr/local*，上述最后一个路径指明了 zeromq 的路径。

### 常见问题
* Q1: 即使安装了 OpenBLAS，仍遇见 `./configure --> cannot find blas_segmm() function` 错误。

  A1: 该错误是指编译器找不着`OpenBLAS`，如果你安装在 $PREFIX (如, /opt/OpenBLAS)，你需要将路径导出，如下所示

      $ export LIBRARY_PATH=$PREFIX/lib:$LIBRARY_PATH
      # e.g.,
      $ export LIBRARY_PATH=/opt/OpenBLAS/lib:$LIBRARY_PATH


* Q2: 碰见错误`cblas.h no such file or directory exists`。

  Q2: 你需要将 cblas.h 所在文件夹包含到 CPLUS_INCLUDE_PATH 中，如：

      $ export CPLUS_INCLUDE_PATH=$PREFIX/include:$CPLUS_INCLUDE_PATH
      # e.g.,
      $ export CPLUS_INCLUDE_PATH=/opt/OpenBLAS/include:$CPLUS_INCLUDE_PATH
      # then reconfigure and make SINGA
      $ ./configure
      $ make


* Q3: 编译SINGA时，碰见错误`SSE2 instruction set not enabled`。

  A3: 你可以尝试以下命令:

      $ make CFLAGS='-msse2' CXXFLAGS='-msse2'


* Q4: 当我试着import .py文件时，从google.protobuf.internal 得到错误`ImportError: cannot import name enum_type_wrapper`。

  A4: 通过 `make install` 安装google protobuf后, 我们应该安装python运行时库。在protobuf源文件夹下运行：

      $ cd /PROTOBUF/SOURCE/FOLDER
      $ cd python
      $ python setup.py build
      $ python setup.py install

  如果你要在系统文件夹中安装python的运行时库，可能要用`sudo`。


* Q5: 遇见由gflags导致的链接错误。

  A5: SINGA不依赖gflags，但你可能在安装glog时安装了gflags。这种情况下你需要用 *thirdparty/install.sh* 重新将glog安装到另一文件夹，并将该文件夹路径导出到LDFLAGS 和 CPPFLAGS 中。


* Q6: 在mac OS X上编译SINGA和安装 `glog` 时，遇到了致命错误 `'ext/slist' file not found`

  A6: 请单独安装`glog`，再尝试以下命令:

      $ make CFLAGS='-stdlib=libstdc++' CXXFLAGS='stdlib=libstdc++'

* Q7: 当我启动一个训练作业时，程序报错为 "ZOO_ERROR...zk retcode=-4..."。

  A7: 这是因为 zookeeper 没有启动，请启动 zookeeper 服务。

      $ ./bin/zk-service start

  如果仍有这个错误，可能是没有java，你可以用下述命令查看

      $ java --version

* Q8: 当我从源文件安装 OpenBLAS 时，被告知需要一个 fortran 编译器。

  A8: 按如下命令编译 OpenBLAS：

      $ make ONLY_CBLAS=1

  或者用apt-get安装

	    $ sudo apt-get install openblas-dev

  或者

	    $ sudo yum install openblas-devel

  后两个命令需要 root 权限，注意OpenBLAS安装后设置环境变量包含头文件和库的路径（参照 依赖 小节）

* Q9: 当我安装 protocol buffer 时，被告知 GLIBC++_3.4.20 not found in /usr/lib64/libstdc++.so.6.

  A9: 这说明链接器找到了 libstdc++.so.6，但是这个文件比用于编译和链接程序的GCC版本老。程序要求属于新版本GCC的libstdc++，所以必须告诉链接器怎么找到新版本的额libstdc++共享库。最简单的解决方法是找到正确的 libstdc++，并把它导出到 LD_LIBRARY_PATH 中。如, 如果GLIBC++_3.4.20 被下面的命令列出，

      $ strings /usr/local/lib64/libstdc++.so.6|grep GLIBC++

  你只需这样设置你的环境变量：

      $ export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

* Q10: 当我在编译glog时，提示如下错误"src/logging_unittest.cc:83:20: error: ‘gflags’ is not a namespace-name"

  A10: 可能是你已经安装的gflags版本，其命名空间不是gflags，而是其他的，比如是'google'。因此glog不能找到 'gflags' 命名空间。
  
  编译glog不需要gflags，你可以修改 configure.ac 文件，忽略 gflags。

  1. cd to glog src directory
  2. 修改 configure.ac 第125行，改为 "AC_CHECK_LIB(gflags, main, ac_cv_have_libgflags=0, ac_cv_have_libgflags=0)"
  3. autoreconf 
 
  然后，请重新编译glog。
