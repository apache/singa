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


在Windows上创建SINGA
=========================

从Microsoft Windows源代码构建SINGA的过程包括四个部分：安装依赖关系，构建SINGA源代码，（可选）安装python模块和（可选）运行单元测试。

1. 安装依赖库
-----------------------

你可以创建一个文件夹来安装依赖库。

依赖库有下面这些：

* 编译器和IDE
	* Visual Studio. 社区版是免费的，可用于构建SINGA。https://www.visualstudio.com/
* CMake
	* 可以从 http://cmake.org/ 下载
	* 确保cmake可执行路径在系统路径中，或者在调用cmake时使用完整路径。
* SWIG
	* 可以从 http://swig.org/ 下载
	* 确保cmake可执行路径在系统路径中，或者在调用swig时使用完整路径。使用最近的版本，比如 3.0.12.

* Protocol Buffers
	* 下载一个合适的版本，比如： 2.6.1: https://github.com/google/protobuf/releases/tag/v2.6.1 .	
	* 下载protobuf-2.6.1.zip和protoc-2.6.1-win32.zip . 
	* 在依赖库文件夹下解压它们。将protoc可执行路径加入系统路径中，或者在调用它是使用完整路径。
	* 在vsproject文件夹中找到Visual Studio solution并打开。
	* 更改创建环境为Release和x64。
	* 创建libprotobuf项目。 
* Openblas
	* 下载一个合适的版本，比如： 0.2.20: http://www.openblas.net 
	* 在依赖文件夹下提取源程序。
	* 如果你没有安装Perl，下载一个perl环境比如 Strawberry Perl (http://strawberryperl.com/)
	* 通过在源文件夹下运行下面指令来创建Visual Studio的解决方案：

	.. code-block:: bash

		cmake -G "Visual Studio 15 2017 Win64" 

	* 打开Visual Studio并更改创建环境为Release和x64。
	* 创建libopenblas项目。

* Google glog
	* 下载一个合适的版本，比如： 0.3.5： https://github.com/google/glog/releases
	* 在依赖文件夹下提取源程序。
	* 打开Visual Studio。
	* 更改创建环境为Release和x64。
	* 创建libglog项目。

2. 创建SINGA
---------------------

* 下载SINGA源代码
* 编译protobuf文件:
	* 进入 src/proto 文件夹

.. code-block:: bash
	
		mkdir python_out
		protoc.exe *.proto --python_out python_out

* 生成支持C++和Python的swig界面:
	进入 src/api

.. code-block:: bash
	
		swig -python -c++ singa.i
		
* 生成支持SINGA的Visual Studio:
	进入SINGA源代码所在的根文件夹

.. code-block:: bash	

	mkdir build
	cd build
	
* 调用cmake并将路径加到系统路径中，类似于如下的例子：

.. code-block:: bash
	
	cmake -G "Visual Studio 15 2017 Win64" ^
	  -DGLOG_INCLUDE_DIR="D:/WinSinga/dependencies/glog-0.3.5/src/windows" ^
	  -DGLOG_LIBRARIES="D:/WinSinga/dependencies/glog-0.3.5/x64/Release" ^
	  -DCBLAS_INCLUDE_DIR="D:/WinSinga/dependencies/openblas-0.2.20/lapack-netlib/CBLAS/include" ^
	  -DCBLAS_LIBRARIES="D:/WinSinga/dependencies/openblas-0.2.20/lib/RELEASE" ^
	  -DProtobuf_INCLUDE_DIR="D:/WinSinga/dependencies/protobuf-2.6.1/src" ^
	  -DProtobuf_LIBRARIES="D:/WinSinga/dependencies/protobuf-2.6.1/vsprojects/x64/Release" ^
	  -DProtobuf_PROTOC_EXECUTABLE="D:/WinSinga/dependencies/protoc-2.6.1-win32/protoc.exe" ^
	  ..

* 在Visual Studio中打开生成好的解决方案
* 更改创建环境为Release和x64。
* 将src/api中的singa_wrap.cxx文件加入singa_objects项目。
* 在singa_objects项目中，打开Additional Include Directories。
* 添加Python include path。
* 添加numpy include path。
* 添加protobuf include path。
* 在singa_objects项目的预处理器定义中，添加USE_GLOG。
* 创建singa_objects项目。
	
* 在singa项目中:
	* 将singa_wrap.obj添加到对象库
	* 将目标名称更改为_singa_wrap
	* 将目标扩展名更改为.pyd
	* 将配置类型更改为动态库（.dll）
	* 转到其他库目录并添加到Python，openblas，protobuf和glog库的路径
	* 转到附加依赖关系并添加libopenblas.lib，libglog.lib和libprotobuf.lib
	
* 创建singa项目
	
	
3. 安装Python模块
------------------------

* 在build/python/setup.py中，将Change _singa_wrap.so改为_singa_wrap.pyd  
* 拷贝src/proto/python_out中的文件到build/python/singa/proto

* （可选择的）创建并激活一个虚拟环境：

.. code-block:: bash

	mkdir SingaEnv
	virtualenv SingaEnv
	SingaEnv\Scripts\activate
	
* 进入build/python文件夹并运行：

.. code-block:: bash

	python setup.py install

* 通过将_singa_wrap.pyd，libglog.dll和libopenblas.dll添加到路径或通过将它们复制到python站点包中的singa包文件夹中，使它们可用。
	
* 通过下面指令验证SINGA已安装：

.. code-block:: bash

	python -c "from singa import tensor"

你可以在这里看到一个关于创建过程的视频教程：
	

.. |video| image:: https://img.youtube.com/vi/cteER7WeiGk/0.jpg
   :scale: 100%
   :align: middle
   :target: https://www.youtube.com/watch?v=cteER7WeiGk

+---------+
| |video| |
+---------+

	
4. 运行单元测试
-----------------

* 在test文件夹下，生成Visual Studio的解决方案:

.. code-block:: bash

	cmake -G "Visual Studio 15 2017 Win64"

* 在Visual Studio中打开生成好的解决方案。

* 更改创建环境为Release和x64。

* 创建glog项目。

* 在test_singa项目中:
	
    * 将USE_GLOG添加到预处理器定义中。
    * 在其他包含目录中，添加上述步骤2中使用的GLOG_INCLUDE_DIR，CBLAS_INCLUDE_DIR和Protobuf_INCLUDE_DIR的路径。 添加也构建和建立/包含文件夹。
    * 转到其他库目录并添加到openblas，protobuf和glog库的路径。 也可以添加build / src / singa_objects.dir / Release。
    * 转到附加依赖项并添加libopenblas.lib，libglog.lib和libprotobuf.lib。 修复两个库的名称：gtest.lib和singa_objects.lib。

* 创建test_singa项目。

* 通过把它们加入到系统路径或拷贝到test/release文件夹下使得libglog.dll和libopenblas.dll可被获取到。

* 单元测试有如下运行方式：

	* 从命令行:
	
		.. code-block:: bash
	
			test_singa.exe

	* 从Visual Studio:
		* 右键单击test_singa项目并选择“设为启动项目”
		* 从“调试”菜单中选择“无需调试即可开始”

你可以在这里看到一个关于运行单元测试的视频教程：
	

.. |video| image:: https://img.youtube.com/vi/393gPtzMN1k/0.jpg
   :scale: 100%
   :align: middle
   :target: https://www.youtube.com/watch?v=393gPtzMN1k

+---------+
| |video| |
+---------+

	
5. 创建基于CUDA的GPU支持
------------------------------

在本节中，我们将扩展前面的步骤以启用GPU。

5.1 安装依赖库
------------------------

除了1部分的依赖库，我们还将需要如下库：

* CUDA 
	
	从https://developer.nvidia.com/cuda-downloads下载合适的版本，比如9.1。确保安装Visual Studio集成模块。

* cuDNN

	从https://developer.nvidia.com/cudnn下载合适的版本，比如7.1。 

* cnmem: 

	* 从https://github.com/NVIDIA/cnmem下载最新版本
	* 创建Visual Studio的解决方案:
	
		.. code-block:: bash
	
			cmake -G "Visual Studio 15 2017 Win64"
		
	* 在Visual Studio中打开生成的解决方案。
	* 将创建设置更改为Release和x64。
	* 创建cnmem项目。
	

5.2 创建SINGA
----------------------

* 调用cmake并在系统中添加类似以下示例的路径：

	.. code-block:: bash
	
    		cmake -G "Visual Studio 15 2017 Win64" ^
			  -DGLOG_INCLUDE_DIR="D:/WinSinga/dependencies/glog-0.3.5/src/windows" ^
			  -DGLOG_LIBRARIES="D:/WinSinga/dependencies/glog-0.3.5/x64/Release" ^
			  -DCBLAS_INCLUDE_DIR="D:/WinSinga/dependencies/openblas-0.2.20/lapack-netlib/CBLAS/include" ^
			  -DCBLAS_LIBRARIES="D:/WinSinga/dependencies/openblas-0.2.20/lib/RELEASE" ^
			  -DProtobuf_INCLUDE_DIR="D:/WinSinga/dependencies/protobuf-2.6.1/src" ^
			  -DProtobuf_LIBRARIES="D:\WinSinga/dependencies/protobuf-2.6.1/vsprojects/x64/Release" ^
			  -DProtobuf_PROTOC_EXECUTABLE="D:/WinSinga/dependencies/protoc-2.6.1-win32/protoc.exe" ^
			  -DCUDNN_INCLUDE_DIR=D:\WinSinga\dependencies\cudnn-9.1-windows10-x64-v7.1\cuda\include ^
			  -DCUDNN_LIBRARIES=D:\WinSinga\dependencies\cudnn-9.1-windows10-x64-v7.1\cuda\lib\x64 ^
			  -DSWIG_DIR=D:\WinSinga\dependencies\swigwin-3.0.12 ^
			  -DSWIG_EXECUTABLE=D:\WinSinga\dependencies\swigwin-3.0.12\swig.exe ^
			  -DUSE_CUDA=YES ^
			  -DCUDNN_VERSION=7 ^
			  ..
  

* 为C++和Python生成swig接口：
	进入 src/api

	.. code-block:: bash
	
		swig -python -c++ singa.i

* 在Visual Studio中打开生成的解决方案

* 将版本设置更改为Release和x64

5.2.1 创建singa_objects
----------------------------

* 将src/api中的singa_wrap.cxx文件添加到singa_objects项目中。
* 在singa_objects项目中，打开其他包含目录。
* 添加Python包含路径。
* 添加numpy包含路径。
* 添加protobuf包含路径。
* 为CUDA，cuDNN和cnmem添加包含路径。
* 在singa_objects项目的预处理器定义中，添加USE_GLOG，USE_CUDA和USE_CUDNN。删除DISABLE_WARNINGS。
* 建立singa_objects项目。
	
5.2.2 创建singa-kernel
---------------------------	

* 创建一个类型为“CUDA 9.1 Runtime”的新Visual Studio程序。 给它起一个名字，比如singa-kernel。
* 该项目带有一个名为kernel.cu的初始文件。 从项目中删除此文件。
* 添加此文件：src / core / tensor / math_kernel.cu
* 在项目设置中：

	* 将Platfrom工具集设置为“Visual Studio 2015（v140）”
	* 将配置类型设置为“静态库（.lib）”
	* 在包含目录中，添加build/include

* 创建singa-kernel项目


5.2.3 创建singa
--------------------
	
* 在singa项目中:

	* 将singa_wrap.obj添加到对象库。
	* 将目标名称更改为_singa_wrap。
	* 将目标扩展名更改为.pyd。
	* 将配置类型更改为动态库（.dll）。
	* 转到其他库目录并添加到Python，openblas，protobuf和glog库的路径。
	* 还将库路径添加到singa-kernel，cnmem，cuda和cudnn。
	* 转到附加依赖关系并添加libopenblas.lib，libglog.lib和libprotobuf.lib。
	* 添加：singa-kernel.lib，cnmem.lib，cudnn.lib，cuda.lib，cublas.lib，curand.lib和cudart.lib。
	
* 创建singa项目

5.3. 安装Python模块
--------------------------

* 在build/python/setup.py中，将Change _singa_wrap.so改为_singa_wrap.pyd  
* 拷贝src/proto/python_out中的文件到build/python/singa/proto

* （可选择的）创建并激活一个虚拟环境：

.. code-block:: bash

	mkdir SingaEnv
	virtualenv SingaEnv
	SingaEnv\Scripts\activate
	
* 进入build/python文件夹并运行：

.. code-block:: bash

	python setup.py install

* 将_singa_wrap.pyd，libglog.dll，libopenblas.dll，cnmem.dll，CUDA运行时（例如cudart64_91.dll）和cuDNN（例如cudnn64_7.dll）添加到路径或通过将它们复制到singa包文件夹 python网站包。
	
* 通过下面指令验证SINGA已安装：

.. code-block:: bash

	python -c "from singa import device; dev = device.create_cuda_gpu()"

关于这部分的视频教程可以在下面找到：
	

.. |video| image:: https://img.youtube.com/vi/YasKVjRtuDs/0.jpg
   :scale: 100%
   :align: middle
   :target: https://www.youtube.com/watch?v=YasKVjRtuDs

+---------+
| |video| |
+---------+

5.4. 运行单元测试
-----------------

* 在test文件夹下，生成Visual Studio的解决方案:

.. code-block:: bash

	cmake -G "Visual Studio 15 2017 Win64"

* 在Visual Studio中打开生成好的解决方案。

* 更改创建环境为Release和x64。

* 创建glog项目。

* 在test_singa项目中:
	
    * 将USE_GLOG添加到预处理器定义中。
    * 在其他包含目录中，添加上述步骤2中使用的GLOG_INCLUDE_DIR，CBLAS_INCLUDE_DIR和Protobuf_INCLUDE_DIR的路径。 添加也构建和建立/包含文件夹。
    * 转到其他库目录并添加到openblas，protobuf和glog库的路径。 也可以添加build / src / singa_objects.dir / Release。
    * 转到附加依赖项并添加libopenblas.lib，libglog.lib和libprotobuf.lib。 修复两个库的名称：gtest.lib和singa_objects.lib。

* 创建test_singa项目。

* 通过把它们加入到系统路径或拷贝到test/release文件夹下使得libglog.dll和libopenblas.dll可被获取到。

* 单元测试有如下运行方式：

	* 从命令行:
	
		.. code-block:: bash
	
			test_singa.exe

	* 从Visual Studio:
		* 右键单击test_singa项目并选择“设为启动项目”
		* 从“调试”菜单中选择“无需调试即可开始”

你可以在这里看到一个关于运行单元测试的视频教程：
	

.. |video| image:: https://img.youtube.com/vi/YOjwtrvTPn4/0.jpg
   :scale: 100%
   :align: middle
   :target: https://www.youtube.com/watch?v=YOjwtrvTPn4

+---------+
| |video| |
+---------+
