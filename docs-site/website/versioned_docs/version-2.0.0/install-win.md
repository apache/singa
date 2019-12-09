---
id: version-2.0.0-install-win
title: Build SINGA on Windows
original_id: install-win
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

The process of building SINGA from source on Microsoft Windows has four parts: install dependencies, build SINGA source, (optionally) install the python module and (optionally) run the unit tests.

## Install Dependencies

You may create a folder for building the dependencies.

The dependencies are:

- Compiler and IDE
  - Visual Studio. The community edition is free and can be used to build SINGA. https://www.visualstudio.com/
- CMake
  - Can be downloaded from http://cmake.org/
  - Make sure the path to cmake executable is in the system path, or use full path when calling cmake.
- SWIG

  - Can be downloaded from http://swig.org/
  - Make sure the path to swig executable is in the system path, or use full path when calling swig. Use a recent version such as 3.0.12.

- Protocol Buffers
  - Download a suitable version such as 2.6.1: https://github.com/google/protobuf/releases/tag/v2.6.1 .
  - Download both protobuf-2.6.1.zip and protoc-2.6.1-win32.zip .
  - Extract both of them in dependencies folder. Add the path to protoc executable to the system path, or use full path when calling it.
  - Open the Visual Studio solution which can be found in vsproject folder.
  - Change the build settings to Release and x64.
  - build libprotobuf project.
- Openblas

  - Download a suitable source version such as 0.2.20 from http://www.openblas.net
  - Extract the source in the dependencies folder.
  - If you don't have Perl installed, download a perl environment such as Strawberry Perl (http://strawberryperl.com/)
  - Build the Visual Studio solution by running this command in the source folder:

  ```bash
  cmake -G "Visual Studio 15 2017 Win64"
  ```

  - Open the Visual Studio solution and change the build settings to Release and x64.
  - Build libopenblas project

- Google glog
  - Download a suitable version such as 0.3.5 from https://github.com/google/glog/releases
  - Extract the source in the dependencies folder.
  - Open the Visual Studio solution.
  - Change the build settings to Release and x64.
  - Build libglog project

## Build SINGA source

- Download SINGA source code
- Compile the protobuf files:

  - Goto src/proto folder

  ```shell
  mkdir python_out
  protoc.exe *.proto --python_out python_out
  ```

- Generate swig interfaces for C++ and Python: Goto src/api

  ```shell
  swig -python -c++ singa.i
  ```

- generate Visual Studio solution for SINGA: Goto SINGA source code root folder

  ```shell
  mkdir build
  cd build
  ```

- Call cmake and add the paths in your system similar to the following example:

  ```shell
  cmake -G "Visual Studio 15 2017 Win64" ^
    -DGLOG_INCLUDE_DIR="D:/WinSinga/dependencies/glog-0.3.5/src/windows" ^
    -DGLOG_LIBRARIES="D:/WinSinga/dependencies/glog-0.3.5/x64/Release" ^
    -DCBLAS_INCLUDE_DIR="D:/WinSinga/dependencies/openblas-0.2.20/lapack-netlib/CBLAS/include" ^
    -DCBLAS_LIBRARIES="D:/WinSinga/dependencies/openblas-0.2.20/lib/RELEASE" ^
    -DProtobuf_INCLUDE_DIR="D:/WinSinga/dependencies/protobuf-2.6.1/src" ^
    -DProtobuf_LIBRARIES="D:/WinSinga/dependencies/protobuf-2.6.1/vsprojects/x64/Release" ^
    -DProtobuf_PROTOC_EXECUTABLE="D:/WinSinga/dependencies/protoc-2.6.1-win32/protoc.exe" ^
    ..
  ```

- Open the generated solution in Visual Studio
- Change the build settings to Release and x64
- Add the singa_wrap.cxx file from src/api to the singa_objects project
- In the singa_objects project, open Additional Include Directories.
- Add Python include path
- Add numpy include path
- Add protobuf include path
- In the preprocessor definitions of the singa_objects project, add USE_GLOG
- Build singa_objects project

- In singa project:

  - add singa_wrap.obj to Object Libraries
  - change target name to \_singa_wrap
  - change target extension to .pyd
  - change configuration type to Dynamic Library (.dll)
  - goto Additional Library Directories and add the path to python, openblas, protobuf and glog libraries
  - goto Additional Dependencies and add libopenblas.lib, libglog.lib and libprotobuf.lib

- build singa project

## Install Python module

- Change `_singa_wrap.so` to `_singa_wrap.pyd` in build/python/setup.py
- Copy the files in `src/proto/python_out` to `build/python/singa/proto`

- Optionally create and activate a virtual environment:

  ```shell
  mkdir SingaEnv
  virtualenv SingaEnv
  SingaEnv\Scripts\activate
  ```

- goto build/python folder and run:

  ```shell
  python setup.py install
  ```

- Make \_singa_wrap.pyd, libglog.dll and libopenblas.dll available by adding them to the path or by copying them to singa package folder in the python site-packages

- Verify that SINGA is installed by running:

  ```shell
  python -c "from singa import tensor"
  ```

A video tutorial for the build process can be found here:

[![youtube video](https://img.youtube.com/vi/cteER7WeiGk/0.jpg)](https://www.youtube.com/watch?v=cteER7WeiGk)

## Run Unit Tests

- In the test folder, generate the Visual Studio solution:

  ```shell
  cmake -G "Visual Studio 15 2017 Win64"
  ```

- Open the generated solution in Visual Studio.

- Change the build settings to Release and x64.

- Build glog project.

- In test_singa project:

  - Add USE_GLOG to the Preprocessor Definitions.
  - In Additional Include Directories, add path of GLOG_INCLUDE_DIR, CBLAS_INCLUDE_DIR and Protobuf_INCLUDE_DIR which were used in step 2 above. Add also build and build/include folders.
  - Goto Additional Library Directories and add the path to openblas, protobuf and glog libraries. Add also build/src/singa_objects.dir/Release.
  - Goto Additional Dependencies and add libopenblas.lib, libglog.lib and libprotobuf.lib. Fix the names of the two libraries: gtest.lib and singa_objects.lib.

- Build test_singa project.

- Make libglog.dll and libopenblas.dll available by adding them to the path or by copying them to test/release folder

- The unit tests can be executed

  - From the command line:

  ```shell
  test_singa.exe
  ```

  - From Visual Studio:
    - right click on the test_singa project and choose 'Set as StartUp Project'.
    - from the Debug menu, choose 'Start Without Debugging'

A video tutorial for running the unit tests can be found here:

[![youtube video](https://img.youtube.com/vi/393gPtzMN1k/0.jpg)](https://www.youtube.com/watch?v=393gPtzMN1k)

## Build GPU support with CUDA

In this section, we will extend the previous steps to enable GPU.

### Install Dependencies

In addition to the dependencies in section 1 above, we will need the following:

- CUDA

  Download a suitable version such as 9.1 from https://developer.nvidia.com/cuda-downloads . Make sure to install the Visual Studio integration module.

- cuDNN

  Download a suitable version such as 7.1 from https://developer.nvidia.com/cudnn

- cnmem:

  - Download the latest version from https://github.com/NVIDIA/cnmem
  - Build the Visual Studio solution:

  ```shell
  cmake -G "Visual Studio 15 2017 Win64"
  ```

  - Open the generated solution in Visual Studio.
  - Change the build settings to Release and x64.
  - Build the cnmem project.

### Build SINGA source

- Call cmake and add the paths in your system similar to the following example:
  ```shell
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
  ```

* Generate swig interfaces for C++ and Python: Goto src/api

  ```shell
  swig -python -c++ singa.i
  ```

* Open the generated solution in Visual Studio

* Change the build settings to Release and x64

#### Building singa_objects

- Add the singa_wrap.cxx file from src/api to the singa_objects project
- In the singa_objects project, open Additional Include Directories.
- Add Python include path
- Add numpy include path
- Add protobuf include path
- Add include path for CUDA, cuDNN and cnmem
- In the preprocessor definitions of the singa_objects project, add USE_GLOG, USE_CUDA and USE_CUDNN. Remove DISABLE_WARNINGS.
- Build singa_objects project

#### Building singa-kernel

- Create a new Visual Studio project of type "CUDA 9.1 Runtime". Give it a name such as singa-kernel.
- The project comes with an initial file called kernel.cu. Remove this file from the project.
- Add this file: src/core/tensor/math_kernel.cu
- In the project settings:

  - Set Platform Toolset to "Visual Studio 2015 (v140)"
  - Set Configuration Type to " Static Library (.lib)"
  - In the Include Directories, add build/include.

- Build singa-kernel project

#### Building singa

- In singa project:

  - add singa_wrap.obj to Object Libraries
  - change target name to \_singa_wrap
  - change target extension to .pyd
  - change configuration type to Dynamic Library (.dll)
  - goto Additional Library Directories and add the path to python, openblas, protobuf and glog libraries
  - Add also the library path to singa-kernel, cnmem, cuda and cudnn.
  - goto Additional Dependencies and add libopenblas.lib, libglog.lib and libprotobuf.lib.
  - Add also: singa-kernel.lib, cnmem.lib, cudnn.lib, cuda.lib , cublas.lib, curand.lib and cudart.lib.

- build singa project

### Install Python module

- Change \_singa_wrap.so to \_singa_wrap.pyd in build/python/setup.py
- Copy the files in src/proto/python_out to build/python/singa/proto

- Optionally create and activate a virtual environment:

  ```shell
  mkdir SingaEnv
  virtualenv SingaEnv
  SingaEnv\Scripts\activate
  ```

- goto build/python folder and run:

  ```shell
  python setup.py install
  ```

- Make \_singa_wrap.pyd, libglog.dll, libopenblas.dll, cnmem.dll, CUDA Runtime (e.g. cudart64_91.dll) and cuDNN (e.g. cudnn64_7.dll) available by adding them to the path or by copying them to singa package folder in the python site-packages

- Verify that SINGA is installed by running:

  ```shell
  python -c "from singa import device; dev = device.create_cuda_gpu()"
  ```

A video tutorial for this part can be found here:

[![youtube video](https://img.youtube.com/vi/YasKVjRtuDs/0.jpg)](https://www.youtube.com/watch?v=YasKVjRtuDs)

### Run Unit Tests

- In the test folder, generate the Visual Studio solution:

  ```shell
  cmake -G "Visual Studio 15 2017 Win64"
  ```

- Open the generated solution in Visual Studio, or add the project to the singa solution that was created in step 5.2

- Change the build settings to Release and x64.

- Build glog project.

- In test_singa project:

  - Add USE_GLOG; USE_CUDA; USE_CUDNN to the Preprocessor Definitions.
  - In Additional Include Directories, add path of GLOG_INCLUDE_DIR, CBLAS_INCLUDE_DIR and Protobuf_INCLUDE_DIR which were used in step 5.2 above. Add also build, build/include, CUDA and cuDNN include folders.
  - Goto Additional Library Directories and add the path to openblas, protobuf and glog libraries. Add also build/src/singa_objects.dir/Release, singa-kernel, cnmem, CUDA and cuDNN library paths.
  - Goto Additional Dependencies and add libopenblas.lib; libglog.lib; libprotobuf.lib; cnmem.lib; cudnn.lib; cuda.lib; cublas.lib; curand.lib; cudart.lib; singa-kernel.lib. Fix the names of the two libraries: gtest.lib and singa_objects.lib.

* Build test_singa project.

* Make libglog.dll, libopenblas.dll, cnmem.dll, cudart64_91.dll and cudnn64_7.dll available by adding them to the path or by copying them to test/release folder

* The unit tests can be executed

  - From the command line:

    ```shell
    test_singa.exe
    ```

  - From Visual Studio:
    - right click on the test_singa project and choose 'Set as StartUp Project'.
    - from the Debug menu, choose 'Start Without Debugging'

A video tutorial for running the unit tests can be found here:

[![youtube video](https://img.youtube.com/vi/YOjwtrvTPn4/0.jpg)](https://www.youtube.com/watch?v=YOjwtrvTPn4)
