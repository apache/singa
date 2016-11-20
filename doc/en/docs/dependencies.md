# Dependent library installation

## Windows

This section is used to compile and install the dependent libraries under
windows system from source codes. The following instructions ONLY work for Visual Studio 2015 as
previous VS does not support [C++11 features](https://msdn.microsoft.com/en-us/library/hh567368.aspx) well (including generic lambdas, auto, non-static
data member initializers). If you intend to generate a 32-bit/64-bit singa solution, please configure all the
VS projects for the dependent libraries as 32-bit/64-bit. This can be done by
"Configuration Manager" in VS 2015 or use corresponding generator in cmake. When compiling the following libraries, you
may get system-specific warnings/errors. Please fix them according to the
prompts by VS.

### Google Logging
The glog library is an optional library for singa project. But it is currently necessary for Window compilation.
Since the latest release version of glog will encounter error C2084 on sprintf function
under VS2015, we test the compilation and installation using the master branch from [github](https://github.com/google/glog).

Step 1: Download and decompress the source code. Or use `git clone
https://github.com/google/glog` to get the code.

Step 2: Open "glog.sln" file under project folder. You will get a conversion
dialog and please finish it by the prompts. Compile all the projects in the solution after
proper configuration, especially "libglog" and "libglog_static" projects.

Step 3: Copy all the header files and the entire directory named "glog" under
"src\windows\" folder into the installation include folder (or system folder).
Copy all the generated library files into the installation library folder (or
system folder).

Step 4: Done.


### Google protobuf

Tested on version 2.6.1:

Step 1: Download and decompress the source code.

Step 2: Open "protobuf.sln" file under "vsprojects" folder. You will get a conversion
dialog and please finish it by the prompts. Compile all the projects in the solution after proper
configuration. Especially "libprotobuf", "libprotobuf-lite", "libprotoc" and
"protoc" projects.

Step 3: Run "extract_includes.bat" script under "vsprojects" folder, you will
get a new "include" folder with all the headers.

Step 4: Copy the library files, such as "libprotobuf.lib",
"libprotobuf-lite.lib", "libprotoc.lib", etc., into your installation library folder (or
system folder). Copy the binary file "protoc" into your installation binary
folder (or system folder). Copy all the headers and folders in "include" folder into your
installation include folder (or system folder).

Step 5: Done.

### CBLAS

There are ready-to-use binary packages online
([link](https://sourceforge.net/projects/openblas/files/)). However, we still install
OpenBLAS with version 0.2.18 as test:

Step 1: Download and decompress the source code.

Step 2: Start a cmd window under the OpenBLAS folder then run the following
commands to generate the solution:

    $ md build $$ cd build
    $ cmake -G "Visual Studio 14" ..

Or run `cmake -G "Visual Studio 14 Win64"` as you wish.

Step 3: Install Perl into your system and put perl.exe on your path. Open "OpenBlas.sln" and build the solution, especially "libopenblas"
project.

Step 4: Copy the library files under "build\lib" folder and all header files
under OpenBLAS folder into installation library and include folders (or system
folders).

Step 5: Done.


## FAQ

1. Error C2375 'snprintf': redefinition; different linkage

    Add “HAVE_SNPRINTF” to “C/C++ - Preprocessor - Preprocessor definitions”

2. Error due to hash map

    Add "_SILENCE_STDEXT_HASH_DEPRECATION_WARNINGS" to Preprocessor Definitions.


