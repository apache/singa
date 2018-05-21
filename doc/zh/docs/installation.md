# 安装

## 从Conda安装

Conda是Python，CPP和其他软件包的软件包管理员。

目前，SINGA有适用于Linux和MacOSX的conda软件包（Python 2.7和Python 3.6）。
建议使用[Miniconda3](https://conda.io/miniconda.html)与SINGA一起使用。安装完miniconda后，执行以下命令之一来安装SINGA。

1. CPU版本

        conda install -c nusdbsystem singa-cpu

2. 由CUDA和cuDNN支持的GPU版本

        conda install -c nusdbsystem singa-gpu

  等同于

		conda install -c nusdbsystem singa=1.1.1=py36_cuda9.0-cudnn7.1.2

  在执行上述命令之前，必须先安装CUDA 9.0。 其他CUDA版本的Singa软件包也可用。 以下说明列出了所有可用的Singa软件包。

		conda search -c nusdbsystem singa

  如果下面指令没有出现错误提示，说明SINGA已经安装成功。

		python -c "from singa import tensor"


## 从源码安装

源文件可以被下载为[tar.gz 文件](https://dist.apache.org/repos/dist/dev/incubator/singa/)，或者为一个git仓库：

		$ git clone https://github.com/apache/incubator-singa.git
		$ cd incubator-singa/

### 由conda创建SINGA

Conda-build是一款构建工具，可以安装anaconda云中的相关库并执行构建脚本。 生成的软件包可以上传到anaconda云中供他人下载和安装。

安装conda-build (安装miniconda后)

		conda install conda-build

创建CPU版本的SINGA

		export BUILD_STR=cpu
		conda build tool/conda/singa/ --python3.6 (or --python2.7)

上面的指令已在Ubuntu 16.04和Mac OSX上测试过。更多信息请参考[Travis-CI页面](https://travis-ci.org/apache/incubator-singa)。

创建GPU版本的SINGA

		export BUILD_STR=cudax.y-cudnna.b.c (e.g. cuda9.0-cudnn7.1.2)
		export CUDNN_PATH=<path to cudnn folder>
		conda build tool/conda/singa/ --python3.6 (or --python2.7)

这些基于GPU平台的指令已经在Ubuntu 16.04 (cuDNN >= 7和CUDA >= 9)上测试过。
[Nvidia的docker镜像](https://hub.docker.com/r/nvidia/cuda/)提供了cuDNN和CUDA的创建环境。

屏幕上将显示生成的包文件的位置。
请参阅[conda安装](https://conda.io/docs/commands/conda-install.html)
对从本地文件安装软件包的说明。

### 使用本地工具在Ubuntu上创建SINGA

编译和运行SINGA需要以下库。
有关在Ubuntu 16.04上安装它们的说明，
请参阅SINGA [Dockerfiles](https://github.com/apache/incubator-singa/blob/master/tool/docker/)。

* cmake (>=2.8)
* gcc (>=4.8.1) or Clang
* google protobuf (>=2.5)
* blas (tested with openblas >=0.2.10)
* swig(>=3.0.10) for compiling PySINGA
* numpy(>=1.11.0) for compiling PySINGA


1. 在incubator-singa目录下创建一个`build`文件夹并进入其中
2. 运行 `cmake [options] ..`
  默认情况下除了`USE_PYTHON`，其他所有可选项都是OFF

    * `USE_MODULES=ON`, 当protobuf和blas没有被安装时使用
    * `USE_CUDA=ON`, 当CUDA和cuDNN可用时使用
    * `USE_PYTHON=ON`, 用于编译PySINGA
    * `USE_PYTHON3=ON`, 用于支持Python 3编译 (默认的是Python 2)
    * `USE_OPENCL=ON`, 用于支持OpenCL编译
    * `PACKAGE=ON`, 用于创建Debian包
    * `ENABLE_TEST`，用于编译单元测试用例

3. 编译代码， 如： `make`
4. 进入python文件夹
5. 运行 `pip install .`或者 `pip install -e .`。第二个指令创建符号链接而不是将文件复制到python站点包文件夹中。

当USE_PYTHON=ON时，第4步和第5步用于安装PySINGA。

在通过ENABLE_TEST=ON编译好SINGA后，你可以运行单元测试

		$ ./bin/test_singa

你可以看到所有测试用例和测试结果。
如果SINGA通过所有测试，那么你已经成功安装了SINGA。

### 在Windows上编译SINGA

基于Python支持的Windows上的安装说明可以在[这里](install_win.html)找到。

### 更多编译选择

#### USE_MODULES

如果protobuf和openblas没有安装，在编译SINGA时需要如下处理：

		$ In SINGA ROOT folder
		$ mkdir build
		$ cd build
		$ cmake -DUSE_MODULES=ON ..
		$ make

cmake会下载OpenBlas和Protobuf（2.6.1）并同SINGA一起编译。

你可以使用`cmake ..`来配置编译操作指令。 如果一些依赖库没有被安装在默认路径下，你需要导出相应的环境变量：

		export CMAKE_INCLUDE_PATH=<path to the header file folder>
		export CMAKE_LIBRARY_PATH=<path to the lib file folder>

#### USE_PYTHON

类似于编译CPP代码， PySINGA可以被这么编译：

		$ cmake -DUSE_PYTHON=ON ..
		$ make
		$ cd python
		$ pip install .

#### USE_CUDA

用户被推荐安装CUDA和[cuDNN](https://developer.nvidia.com/cudnn)以在GPU上运行SINGA时获得更好的性能。

SINGA已经在CUDA 9和cuDNN 7上测试过。 如果cuDNN在非系统目录下解压，如/home/bob/local/cudnn/, 下面的指令需要被执行以让cmake和运行时能找到它：

		$ export CMAKE_INCLUDE_PATH=/home/bob/local/cudnn/include:$CMAKE_INCLUDE_PATH
		$ export CMAKE_LIBRARY_PATH=/home/bob/local/cudnn/lib64:$CMAKE_LIBRARY_PATH
		$ export LD_LIBRARY_PATH=/home/bob/local/cudnn/lib64:$LD_LIBRARY_PATH

cmake对CUDA和cuDNN的操作选项应该被开启：

		# 依赖库已经被安装
		$ cmake -DUSE_CUDA=ON ..
		$ make

#### USE_OPENCL

SINGA用opencl-header和viennacl（1.7.1版本或更新）以获得OpenCL支持。 它们可由以下指令安装：

		# 在Ubuntu 16.04上
		$ sudo apt-get install opencl-headers, libviennacl-dev
		# 在Fedora上
		$ sudo yum install opencl-headers, viennacl

另外，你将需要OpenCL Installable Client Driver (ICD)以在你的平台上运行OpenCL。

* 对于AMD和Nvidia GPU, 驱动包还需要包含正确的OpenCL ICD。
* 对于英特尔CPUs和/或GPUs, 可以从[英特尔官网](https://software.intel.com/en-us/articles/opencl-drivers)上获取到。 注意， 官网上提供的驱动仅支持近期的CPUs和GPUs。
* 对于更老的英特尔CPUs，你可以选用`beignet-opencl-icd`包。

注意，在CPU上运行OpenCL目前是不推荐的，因为很慢。 内存传输是以秒的级别（CPU上为1000 ms，而GPU上为1毫秒）。

更多关于OpenCL环境配置的信息可以从[这里](https://wiki.tiker.net/OpenCLHowTo)获得。

如果ViennaCL包版本低于1.7.1，你将需要从源码编译：

从[git仓库](https://github.com/viennacl/viennacl-dev)clone，checkout到`release-1.7.1`标签。记得把仓库的路径加到环境变量`PATH`中，并且创建库到`LD_LIBRARY_PATH`。

编译基于OpenCL支持的SINGA (测试与SINGA 1.1)：

		$ cmake -DUSE_OPENCL=ON ..
		$ make

#### PACKAGE

此设置用于创建Debian包。 设置PACKAGE=ON并用以下命令创建包：

		$ cmake -DPACKAGE=ON
		$ make package


## FAQ

* Q: 在使用由wheel安装的PySINGA('import singa')时，出现错误。

    A: 请查看`python -c "from singa import _singa_wrap"`详细错误提示。 这有时是由依赖库造成的，比如，有多个版本的protobuf，cudnn缺失，numpy版本不匹配。 下面的步骤详述了不同的案例：
    1. 检查cudnn，cuda和gcc版本，推荐使用cudnn5，cuda7.5和gcc4.8/4.9。 如果gcc是5.0版本， 需要降低版本。 如果cudnn确实或者与wheel版本不匹配，你可以将正确的cudnn版本下载到~/local/cudnn/ 并且

            $ echo "export LD_LIBRARY_PATH=/home/<yourname>/local/cudnn/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc

    2. 如果是protobuf的问题，需要下载最新的[编译过protobuf和openblas的whl文件](https://issues.apache.org/jira/browse/SINGA-255)。 或者，你可以从源码安装protobuf到指定文件夹，比如：~/local/；解压tar文件，然后执行

            $ ./configure --prefix=/home/<yourname>local
            $ make && make install
            $ echo "export LD_LIBRARY_PATH=/home/<yourname>/local/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
            $ source ~/.bashrc

    3. 如果找不到其他python库，你可以用pip或conda创建python虚拟环境。

    4. 如果不是以上原因造成的，进入`_singa_wrap.so`所在文件夹，执行

            $ python
            >> import importlib
            >> importlib.import_module('_singa_wrap')

      检查错误信息。 比如，如果numpy版本不匹配，错误信息将会是

            RuntimeError: module compiled against API version 0xb but this version of numpy is 0xa

      接着，你需要提升numpy版本。


* Q: 运行`cmake ..`报错，找不到依赖库。

    A: 如果你没有安装相应库，就去安装它们。如果你把这些库安装在非系统默认的路径下，如/usr/local，你可以将正确路径导出到环境变量中：

        $ export CMAKE_INCLUDE_PATH=<path to your header file folder>
        $ export CMAKE_LIBRARY_PATH=<path to your lib file folder>


* Q: `make`报错，如连接阶段

    A: 如果你的库文件在非系统默认路径下，你需要导出相应的变量

        $ export LIBRARY_PATH=<path to your lib file folder>
        $ export LD_LIBRARY_PATH=<path to your lib file folder>


* Q: 头文件错误，比如：'cblas.h no such file or directory exists'

    A: 你需要把cblas.h的路径加入到CPLUS_INCLUDE_PATH，如

        $ export CPLUS_INCLUDE_PATH=/opt/OpenBLAS/include:$CPLUS_INCLUDE_PATH

* Q:编译SINGA时，我发现了错误`SSE2 instruction set not enabled`

    A:你可以尝试如下命令

        $ make CFLAGS='-msse2' CXXFLAGS='-msse2'

* Q:当我试图导入.py文件时，我得到错误提示`ImportError: cannot import name enum_type_wrapper`。

    A: 你需要安装绑定到python的protobuf，可以由如下命令安装

        $ sudo apt-get install protobuf

    或者从源码安装

        $ cd /PROTOBUF/SOURCE/FOLDER
        $ cd python
        $ python setup.py build
        $ python setup.py install

* Q: 当我从源码创建OpenBLAS时，被告知需要Fortran编译器。

    A: 你可以用如下命令编译OpenBLAS

        $ make ONLY_CBLAS=1

    或者

        $ sudo apt-get install libopenblas-dev

* Q: 当我创建protocol buffer时，出现错误提示`GLIBC++_3.4.20 not found in /usr/lib64/libstdc++.so.6`。

    A: 这说明连接器找到了libstdc++.so.6，但是这个库属于一个更老版本的GCC编译器。 要编译的程序依赖于定义在新版本GCC下的libstdc++库，所以连接器必须被告知如何找到新版的可共享的libstdc++库。 最简单的处理方法是找到正确的libstdc++库，导出到LD_LIBRARY_PATH变量。 比如，如果GLIBC++_3.4.20被列在如下命令的输出中

        $ strings /usr/local/lib64/libstdc++.so.6|grep GLIBC++

    之后，你只需要设置环境变量

        $ export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

* Q: 当我创建glog时报错，"src/logging_unittest.cc:83:20: error: 'gflags' is not a namespace-name"。

    A: 这可能是你装了一个不同命名空间的gflags，比如"google"，所以glog找不到'gflags'命名空间。 gflags不是创建glog必须的， 所以你可以修改configure.ac文件以忽略gflags。

        1. cd to glog src directory
        2. change line 125 of configure.ac  to "AC_CHECK_LIB(gflags, main, ac_cv_have_libgflags=0, ac_cv_have_libgflags=0)"
        3. autoreconf

    之后，你可以重新创建glog。

* Q: 当使用虚拟环境时，每次我运行pip install都会重新安装numpy。 然而，在`import numpy`时，numpy可能并没有被使用。

    A: 这可能是因为在使用虚拟环境时，`PYTHONPATH`被设置成了空以防止与虚拟环境中的路径发生冲突。

* Q: 当从源码编译PySINGA时，会因为缺失<numpy/objectarray.h>而出现编译错误。

    A: 请安装numpy并且通过如下命令导出numpy头文件

        $ export CPLUS_INCLUDE_PATH=`python -c "import numpy; print numpy.get_include()"`:$CPLUS_INCLUDE_PATH

* Q: 当在Mac OS X下运行PySINGA时，我得到了错误信息"Fatal Python error: PyThreadState_Get: no current thread Abort trap: 6"。

    A: 这个错误很典型地出现在当你系统中存在多个版本的python并且你是通过pip安装
SINGA的（这个问题可以通过由conda安装SINGA来解决）， 比如：一个来自于OS，一个通过Homebrew安装。 和SINGA连接的Python必须和Python解析器是同个版本。你可以通过which python来查看python解析器版本，并通过otool -L <path to _singa_wrap.so>检查和PySINGA连接的Python版本。 为了解决这个问题， 需要用正确的Python版本来编译SINGA。 特别地，如果你从源码创建的PySINGA，当唤起[cmake](http://stackoverflow.com/questions/15291500/i-have-2-versions-of-python-installed-but-cmake-is-using-older-version-how-do)时你需要指定安装路径

        $ cmake -DPYTHON_LIBRARY=`python-config --prefix`/lib/libpython2.7.dylib -DPYTHON_INCLUDE_DIR=`python-config --prefix`/include/python2.7/ ..

    如果从二进制文件安装PySINGA，比如debian或者wheel，那么你需要改变python解析器，如重置变量$PATH，把正确的Python路径加在最前面。

