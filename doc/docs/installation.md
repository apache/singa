#Building SINGA from source
To install SINGA, please clone the newest code from [Github](https://github.com/apache/incubator-singa) and execute the following commands,

    
    $ git clone https://github.com/apache/incubator-singa.git
    $ cd incubator-singa/
    # switch to dev branch
    $ git checkout dev
  
A SINGA uses CNMem as a submodule in lib/, execute the following commands to initialize and update the submodules for SINGA, 

    $ git submodule init
    $ git submodule update

Then in SINGA_ROOT, execute the following commands for compiling SINGA,

    $ mkdir build
    $ cd build/
    # generate Makefile for compilation
    $ cmake ..
    # compile SINGA
    $ make

After compiling SINGA, in order to test all components of SINGA, please execute the following commands,
    
    $ cd bin/
    $ ./test_singa

You can see all the testing cases with testing results. If SINGA has passes all tests, then you have successfully installed SINGA. Please proceed to enjoy SINGA!