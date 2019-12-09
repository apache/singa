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

* homebrew is used to install the requirements. Try:

.. code-block:: bash

	brew update

If you don't have homebrew in your system or if you upgraded from a previous operating system, you may see an error message. See FAQ below.

* installing required software for building SINGA:

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

# These are needed if USE_MODULES option in cmake is used.

.. code-block:: bash

	brew install automake
	brew install wget

* preparing compiler 

To let the compiler (and cmake) know the openblas
path,

.. code-block:: bash

	export CMAKE_INCLUDE_PATH=/usr/local/opt/openblas/include:$CMAKE_INCLUDE_PATH
	export CMAKE_LIBRARY_PATH=/usr/local/opt/openblas/lib:$CMAKE_LIBRARY_PATH

To let the runtime know the openblas path,

.. code-block:: bash

	export LD_LIBRARY_PATH=/usr/local/opt/openblas/library:$LD_LIBRARY_PATH

Add the numpy header path to the compiler flags, for example:

.. code-block:: bash

	export CXXFLAGS="-I /usr/local/lib/python2.7/site-packages/numpy/core/include $CXXFLAGS"

* Get the source code and build it:

.. code-block:: bash

	git clone https://github.com/apache/singa.git

	cd singa
	mkdir build
	cd build

	cmake ..
	make

* Optional: create virtual enviromnet:

.. code-block:: bash

	virtualenv ~/venv
	source ~/venv/bin/activate

* Install the python module

.. code-block:: bash
	
	cd python
	pip install .

If there is no error message from

.. code-block:: bash

    python -c "from singa import tensor"

then SINGA is installed successfully.

* Run Jupyter notebook

.. code-block:: bash

	pip install matplotlib

	cd ../../doc/en/docs/notebook
	jupyter notebook

Video Tutorial
--------------

See these steps in the following video:

.. |video| image:: https://img.youtube.com/vi/T8xGTH9vCBs/0.jpg
   :scale: 100%
   :align: middle
   :target: https://www.youtube.com/watch?v=T8xGTH9vCBs

+---------+
| |video| |
+---------+

FAQ
---

* How to install or update homebrew:

.. code-block:: bash
	
	/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

* There is an error with protobuf. 

Try overwriting the links:

.. code-block:: bash

	brew link --overwrite protobuf
