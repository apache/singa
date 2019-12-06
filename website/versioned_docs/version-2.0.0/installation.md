---
id: version-2.0.0-installation
title: Installation
original_id: installation
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

## From Conda

Conda is a package manager for Python, CPP and other packages.

Currently, SINGA has conda packages for Linux and MacOSX. [Miniconda3](https://conda.io/miniconda.html) is recommended to use with SINGA. After installing miniconda, execute the one of the following commands to install SINGA.

1. CPU only

```shell
$ conda install -c nusdbsystem -c conda-forge singa-cpu
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ntkhi-Z6XTR8WYPXiLwujHd2dOm0772V)

2. GPU with CUDA and cuDNN (CUDA driver >=384.81 is required)

```shell
$ conda install -c nusdbsystem -c conda-forge singa-gpu
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1do_TLJe18IthLOnBOsHCEe-FFPGk1sPJ)

3. Install a specific version of SINGA. The following command lists all the available SINGA packages.

```shell
$ conda search -c nusdbsystem singa

Loading channels: done
# Name                       Version           Build  Channel
singa                      2.1.0.dev        cpu_py36  nusdbsystem
singa                      2.1.0.dev        cpu_py37  nusdbsystem
```

The following command install a specific version of SINGA,

```shell
$ conda install -c nusdbsystem -c conda-forge singa=2.1.0.dev=cpu_py37
```

If there is no error message from

```shell
$ python -c "from singa import tensor"
```

then SINGA is installed successfully.

## Using Docker

Install Docker on your local host machine following the [instructions](https://docs.docker.com/install/). Add your user into the [docker group](https://docs.docker.com/install/linux/linux-postinstall/) to run docker commands without `sudo`.

1. CPU-only.

```shell
$ docker run -it apache/singa:2.0.0-cpu /bin/bash
```

2. With GPU enabled. Install [Nvidia-Docker](https://github.com/NVIDIA/nvidia-docker) after install Docker.

```shell
$ nvidia-docker run -it apache/singa:2.0.0-gpu /bin/bash
```

3. For the complete list of SINGA Docker images (tags), visit the [docker hub site](https://hub.docker.com/r/apache/singa/). For each docker image, the tag is named as

```shell
version-(cpu|gpu)[-devel]
```

| Tag | Description | Example value |
| --- | --- | --- |
| `version` | SINGA version | 'nightly', '2.0.0', '1.2.0' |
| `cpu` | the image cannot run on GPUs | 'cpu' |
| `gpu` | the image can run on Nvidia GPUs | 'gpu', or 'cudax.x-cudnnx.x' e.g., 'cuda10.0-cudnn7.3' |
| `devel` | indicator for development | if absent SINGA Python package is installed for runtime only; if present, the building environment is also created, you can recompile SINGA from source at '/root/singa' |

> Please note that using the nightly built images is not recommended excpet for SINGA development and testing. Using an official release is recommended. Official releases have version numbers such as '2.0.0' and '1.2.0'.

## From source

You can [build and install SINGA](build.md) from the source code using native building tools or conda-build, on local host OS or in a Docker container.

## FAQ

- Q: Error from `from singa import tensor`

  A: Check the detailed error from

  ```shell
  python -c  "from singa import _singa_wrap"
  # go to the folder of _singa_wrap.so
  ldd path to _singa_wrap.so
  python
  >> import importlib
  >> importlib.import_module('_singa_wrap')
  ```

  The folder of `_singa_wrap.so` is like `~/miniconda3/lib/python3.7/site-packages/singa`. Normally, the error is caused by the mismatch or missing of dependent libraries, e.g. cuDNN or protobuf. The solution is to create a new virtual environment and install SINGA in that environment, e.g.,

  ```shell
  conda create -n singa
  conda activate singa
  conda install -c nusdbsystem -c conda-forge singa-cpu
  ```

- Q: When using virtual environment, every time I install SINGA, numpy would be reinstalled. However, the numpy is not used when I run `import numpy`

  A: It could be caused by the `PYTHONPATH` environment variable which should be set to empty when you are using virtual environment to avoid the conflicts with the path of the virtual environment.

- Q: When I run SINGA in Mac OS X, I got the error "Fatal Python error: PyThreadState_Get: no current thread Abort trap: 6"

  A: This error happens typically when you have multiple versions of Python in your system, e.g, the one comes with the OS and the one installed by Homebrew. The Python linked by SINGA must be the same as the Python interpreter. You can check your interpreter by `which python` and check the Python linked by SINGA via `otool -L <path to _singa_wrap.so>`. This problem should be resolved if SINGA is installation via conda.
