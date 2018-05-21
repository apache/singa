.. Licensed to the Apache Software Foundation (ASF) under one
   or more contributor license agreements.  See the NOTICE file
   distributed with this work for additional information
   regarding copyright ownership.  The ASF licenses this file
   to you under the Apache License, Version 2.0 (the
   "License"); you may not use this file except in compliance
   with the License.  You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing,
   software distributed under the License is distributed on an
   "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
   KIND, either express or implied.  See the License for the
   specific language governing permissions and limitations
   under the License.


Installing SINGA on macOS 13.10
===============================

Requirements
------------

* homebrew被用来安装需要的库。尝试如下指令：

.. code-block:: bash

	brew update

如果你的系统中没有homebrew或者你升级了之前的操作系统，你可能会看到错误信息，请参考FAQ。

* 安装创建SINGA需要的软件:

.. code-block:: bash

	brew tap homebrew/science
	brew tap homebrew/python

	brew install opebblas
	brew install protobuf
	brew install swig

	brew install git
	brew install cmake

	brew install python
	brew install opencv
	brew install glog lmdb

# 如果在cmake中使用USE_MODULES选项将会需要下面的操作：

.. code-block:: bash

	brew install automake
	brew install wget

* 准备编译器 

为了让编译器（和cmake）知道openblas路径，

.. code-block:: bash

	export CMAKE_INCLUDE_PATH=/usr/local/opt/openblas/include:$CMAKE_INCLUDE_PATH
	export CMAKE_LIBRARY_PATH=/usr/local/opt/openblas/lib:$CMAKE_LIBRARY_PATH


让运行时知道openblas路径，

.. code-block:: bash

	export LD_LIBRARY_PATH=/usr/local/opt/openblas/library:$LD_LIBRARY_PATH

将numpy头文件路径加入编译器标记中，例如：

.. code-block:: bash

	export CXXFLAGS="-I /usr/local/lib/python2.7/site-packages/numpy/core/include $CXXFLAGS"

* 获取源代码并编译它：

.. code-block:: bash

	git clone https://github.com/apache/incubator-singa.git

	cd incubator-singa
	mkdir build
	cd build

	cmake ..
	make

* 可选的： 创建虚拟环境：

.. code-block:: bash

	virtualenv ~/venv
	source ~/venv/bin/activate

* 安装python模块

.. code-block:: bash
	
	cd python
	pip install .

如果从下面指令没有得到错误信息，则说明SINGA已成功安装。

.. code-block:: bash

    python -c "from singa import tensor"

* 运行Jupyter notebook

.. code-block:: bash

	pip install matplotlib

	cd ../../doc/en/docs/notebook
	jupyter notebook

视频教程
--------------

接下来的步骤请参考视频:

.. |video| image:: https://img.youtube.com/vi/T8xGTH9vCBs/0.jpg
   :scale: 100%
   :align: middle
   :target: https://www.youtube.com/watch?v=T8xGTH9vCBs

+---------+
| |video| |
+---------+

FAQ
---

* 如何安装或更新homebrew:

.. code-block:: bash
	
	/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

* protobuf报错. 

尝试重写链接:

.. code-block:: bash

	brew link --overwrite protobuf
