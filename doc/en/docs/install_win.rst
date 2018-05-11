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


Building SINGA on Windows
=========================

The process of building SINGA from source on Microsoft Windows has three parts: install dependencies, build SINGA source, and (optionally) install the python module.

1. Install Dependencies
----------------------

You may create a folder for building the dependencies.

The dependencies are:

* Compiler and IDE
	* Visual Studio. The community edition is free and can be used to build SINGA. https://www.visualstudio.com/
* CMake
	* Can be downloaded from http://cmake.org/ 
	* Make sure the path to cmake executable is in the system path, or use full path when calling cmake.
* SWIG
	* Can be downloaded from http://swig.org/ 
	* Make sure the path to swig executable is in the system path, or use full path when calling swig. Use a recent version such as 3.0.12.

* Protocol Buffers
	* Download a suitable version such as 2.6.1: https://github.com/google/protobuf/releases/tag/v2.6.1 .	
	* Download both protobuf-2.6.1.zip and protoc-2.6.1-win32.zip . 
	* Extract both of them in dependecies folder. Add the path to protoc executable to the system path, or use full path when calling it.
	* Open the Visual Studio solution which can be found in vsproject folder.
	* Change the build settings to Release and x64.
	* build libprotobuf project. 
* Openblas
	* Download a suitable source version such as 0.2.20 from http://www.openblas.net 
	* Extract the source in the dependecies folder.
	* If you don't have Perl installed, download a perl environment such as Strawberry Perl (http://strawberryperl.com/)
	* Build the Visual Studio solution by running this command in the source folder:

	.. code-block:: bash

		cmake -G "Visual Studio 15 2017 Win64" 

	* Open the Visual Studio solution and change the build settings to Release and x64.
	* Build libopenblas project

* Google glog
	* Download a suitable version such as 0.3.5 from https://github.com/google/glog/releases
	* Extract the source in the dependencies folder.
	* Open the Visual Studio solution.
	* Change the build settings to Release and x64.
	* Build libglog project

2. Build SINGA source
---------------------

* Download SINGA source code
* Compile the protobuf files:
	* Goto src/proto folder

.. code-block:: bash
	
		mkdir python_out
		protoc.exe *.proto --python_out python_out

* Generate swig interfaces for C++ and Python:
	Goto src/api

.. code-block:: bash
	
		swig -python -c++ singa.i
		
* generate Visual Studio solution for SINGA:
	Goto SINGA source code root folder

.. code-block:: bash	

	mkdir build
	cd build
	
* Call cmake and add the paths in your system similar to the following example:

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

* Open the generated solution in Visual Studio
* Change the build settings to Release and x64
* Add the singa_wrap.cxx file from src/api to the singa_objects project
* In the singa_objects project, open Additional Include Directories.
* Add Python include path
* Add numpy include path
* Add protobuf include path
* In the preprocessor definitions of the singa_objects project, add USE_GLOG
* Build singa_objects project
	
* In singa project:
	* add singa_wrap.obj to Object Libraries
	* change target name to _singa_wrap
	* change target extension to .pyd
	* change configuration type to Dynamic Library (.dll)
	* goto Additional Library Directories and add the path to python, openblas, protobuf and glog libraries
	* goto Additional Dependencies and add libopenblas.lib, libglog.lib and libprotobuf.lib
	
* build singa project
	
	
3. Install Python module
------------------------

* Change _singa_wrap.so to _singa_wrap.pyd in build/python/setup.py 
* Copy the files in src/proto/python_out to build/python/singa/proto

* Optionally create and activate a virtual environment:

.. code-block:: bash

	mkdir SingaEnv
	virtualenv SingaEnv
	SingaEnv\Scripts\activate
	
* goto build/python folder and run:

.. code-block:: bash

	python setup.py install

* Make _singa_wrap.pyd, libglog.dll and libopenblas.dll available by adding them to the path or by copying them to singa package folder in the python site-packages 
	
* Verify that SINGA is installed by running:

.. code-block:: bash

	python -c "from singa import tensor"

A video tutorial for the build process can be found here:
	

.. |video| image:: https://img.youtube.com/vi/cteER7WeiGk/0.jpg
   :scale: 100%
   :align: middle
   :target: https://www.youtube.com/watch?v=cteER7WeiGk

+---------+
| |video| |
+---------+

	
	
	
	
	
	
	
	
	
	
