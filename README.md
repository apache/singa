##Apache SINGA

Distributed deep learning system

[Project Page](http://singa.incubator.apache.org)



###Dependencies

SINGA is developed and tested on Linux platforms with the following external libraries.

  * gflags version 2.1.1, use the default setting for namespace (i.e., gflags).

  * glog version 0.3.3.

  * gtest version 1.7.0.

  * google-protobuf version 2.6.0.

  * openblas version >= 0.2.10.

  * opencv version 2.4.9.

  * zeromq version >= 3.2

  * czmq version >= 3

  * zookeeper version 3.4.6

Tips:
For libraries like openblas, opencv, older versions may also work, because we do not use any newly added features.


###Building SINGA From Sources

The build system of SINGA is based on GNU autotools. To build singa, you need gcc version >= 4.8.2. To support visualization, you need python2.7.x.
The common steps to build SINGA can be:

	1.Extract source files;
	2.Run configure script to generate makefiles;
	3.Build and install SINGA.

On Unix-like systems with GNU Make as build tool, these build steps can be summarized by the following sequence of commands executed in a shell.

	$ cd SINGA/FOLDER
	$ ./configure
	$ make
	$ make install

After executing above commands, SINGA library will be installed in the system default directory.
If you want to specify your own installation directory, use the following command instead.

	$ ./configure --prefix=/YOUR/OWN/FOLDER

Also, you can create your own build directory and run those commands. For example, please try:

	$ cd SINGA/FOLDER
	$ mkdir build && cd build
	$ ../configure
	$ make	
	$ make install

The result of configure script will indicate you whether there exist dependency missings in your system.
If you do not install the dependencies, you can run the following commands.
To download & install the thirdparty dependencies:

	$ cd thirdparty
	$ ./install.sh MISSING_LIBRARY_NAME1 YOUR_INSTALL_PATH1 MISSING_LIBRARY_NAME2 YOUR_INSTALL_PATH2 ...

If you do not specify the installation path, the library will be installed in default folder.
For example, if you want to build zeromq library in system folder and gflags in /usr/local, just run:

	$ ./install.sh zeromq gflags /usr/local

Another example can be to install all dependencies in /usr/local directory:

	$ ./install.sh all /usr/local

Here is a table showing the first arguments:

	MISSING_LIBRARY_NAME	LIBRARIES
	cmake					cmake tools
	czmq*					czmq lib
	gflags					gflags lib
	glog					glog lib
	lmdb					lmdb lib
	OpenBLAS				OpenBLAS lib
	opencv					OpenCV
	protobuf				Google protobuf
	zeromq					zeromq lib
	zookeeper				Apache zookeeper

*: Since czmq depends on zeromq, the script offers you one more argument with "-f" option  to indicate zeromq location.
The installation commands of czmq can be:

	$ ./install.sh czmq  /usr/local -f /usr/local/zeromq

After the execution, czmq will be installed in /usr/local while zeromq is installed in /usr/local/zeromq.

### FAQ

Q1:While compiling Singa and installing glog on max OS X, I get fatal error "'ext/slist' file not found"

A1:You may install glog individually and try command :

	$ make CFLAGS='-stdlib=libstdc++' CXXFLAGS='stdlib=libstdc++'
#


Q2:While compiling Singa, I get error "SSE2 instruction set not enabled"

A2:You can try following command:
	
	$ make CFLAGS='-msse2' CXXFLAGS='-msse2'
#


Q3:I get error "./configure --> cannot find blas_segmm() function" even I run "install.sh OpenBLAS".

A3:Since OpenBLAS library is installed in /opt folder by default or /other/folder by your preference, you may edit your environment settings. You need add its default installation directories before linking, just run:

	$ export LDFLAGS=-L/opt
	
Or as an alternative option, you can also edit LIBRARY_PATH to figure it out.

#

Q4:I get ImportError from google.protobuf.internal when I try to import .py files. (ImportError: cannot import name enum_type_wrapper)

A4:After install google protobuf by "make install", we should install python runtime libraries. Go to protobuf source directory, run:

	$ cd /PROTOBUF/SOURCE/FOLDER
	$ cd python
	$ python setup.py build
	$ python setup.py install

You may need "sudo" when you try to install python runtime libraries in system folder.

#
