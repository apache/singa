# Package SINGA using conda-build

[conda-build](https://conda.io/docs/user-guide/tasks/build-packages/index.html) is a packaging tool like apt-get, which is associated with [anaconda cloud](https://anaconda.org/) for package management for both python and cpp libraries.


## Environment variables

Depending on the building evironment, export the corresponding build string.

	# for singa with gpu, e.g. cuda8.0-cudnn7.0.5
    export BUILD_STR=cudax.y-cudnna.b.c

    # for singa running only on cpu
    export BUILD_STR=cpu


To package SINGA with CUDA and CUDNN, 

    export CUDNN_PATH=<path to cudnn folder>

this folder should include a subfolder `include/cudnn.h` for the header file, and another subfolder `lib64` for the shared libraries. The BUILD_STR and CUDNN_PATH must be consistent. For example, if CUDNN_PATH is set, then BUILD_STR must be like cudax.y-cudnna.b.c.

## Instruction

After exporting the environment variables, execute the following command to compile SINGA and package it

    conda-build .

You will see the package path from the screen output.