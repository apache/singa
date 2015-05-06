Apache SINGA
=====

Distributed deep learning system

[Project Page](http://singa.incubator.apache.org)



Dependencies
=====

SINGA is developed and tested on Linux platforms with the following external libraries.

  * gflags version 2.1.1, use the default setting for namespace (i.e., gflags).

  * glog version 0.3.3.

  * gtest version 1.7.0.

  * google-protobuf version 2.6.0.

  * openblas version >= 0.2.10.

  * opencv version 2.4.9.

  * zeromq version >= 3.2

  * czmq version >= 3

Tips:
For libraries like openblas, opencv, older versions may also work, because we do not use any newly added features.



BUILDING SINGA SOFTWARE FROM SOURCES
=====
Build steps
-----
The build system of SINGA is based on GNU autotools.
The common steps to build SINGA can be:

	1.Extract source files;
	2.Run configure script to generate makefiles;
	3.Build and install SINGA.

On Unix-like systems with GNU Make as build tool, these build steps can be summarized by the following sequence of commands executed in a shell.

	$ cd SINGA/FOLDER
	$ ./configure
	$ make
	$ make install

After executing above commands, SINGA library will be installed in the default directory.
If you want to specify your own installation directory, use the following command instead.

	$ ./configure --prefix=/YOUR/OWN/FOLDER

The result of configure script will indicate you whether there exist dependency missings in your system.
If you do not install the dependencies, you can run the following commands.
To download the thirdparty dependencies:

	$ ./script/download.sh

After downloading, to install the thirdparty dependencies:

	$ ./thirdparty/install-dependencies.sh MISSING_LIBRARY_NAME1 YOUR_INSTALL_PATH1 MISSING_LIBRARY_NAME2 YOUR_INSTALL_PATH2

If you do not specify the installation path, the lirary will be installed in default folder.
For example, if you want to build zeromq library in system folder and gflags in /usr/local, just run:
	
	$ ./thirdparty/install-dependencies.sh zeromq gflags /usr/local

Another example can be to install all dependencies in /usr/local directory:

	$ ./thirdparty/install-dependencies.sh all /usr/local

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

*: Since czmq depends on zeromq, the script offers you one more argument to indicate zeromq location.
The installation commands of czmq can be:
		$ ./thirdparty/install-dependencies.sh czmq  /usr/local /usr/local/zeromq
After the execution, czmq will be installed in /usr/local while zeromq is installed in /usr/local/zeromq.
