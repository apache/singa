#/**
# * Copyright 2015 The Apache Software Foundation
# *
# * Licensed to the Apache Software Foundation (ASF) under one
# * or more contributor license agreements.  See the NOTICE file
# * distributed with this work for additional information
# * regarding copyright ownership.  The ASF licenses this file
# * to you under the Apache License, Version 2.0 (the
# * "License"); you may not use this file except in compliance
# * with the License.  You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */
#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ(2.59)
AC_INIT(singa, 0.1.0, dev@singa.incubator.apache.org)
AC_CONFIG_AUX_DIR(config)
AC_CONFIG_MACRO_DIR(config)
AC_CONFIG_SRCDIR([src/utils/common.cc])
AC_CONFIG_HEADER([config.h])
AM_INIT_AUTOMAKE([subdir-objects foreign])
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])
ACLOCAL_AMFLAGS = -I m4 --install

# Checks for programs.
AC_PROG_CXX([g++])
AC_PROG_CC
AC_DISABLE_STATIC
AC_PROG_LIBTOOL

# Setup custom CUDA paths
AC_ARG_ENABLE(cuda,
  [AS_HELP_STRING(--enable-cuda,enable CUDA support)],
    cudaval="yes",
    cudaval="no")
AM_CONDITIONAL(DCUDA, [test "$cudaval" = "yes"])

AC_ARG_WITH([cuda],
   [AS_HELP_STRING(
        [--with-cuda=PATH],
        [prefix where CUDA is installed])],
   [cuda_prefix=$withval],
   [cuda_prefix="/usr/local/cuda"])
if test "$cuda_prefix" == "yes"; then
    if test "$withval" == "yes"; then
        cuda_prefix="/usr/local/cuda"
    fi
fi

if test x"$cudaval" = x"yes"; then
    AC_MSG_CHECKING([nvcc in $cuda_prefix/bin])
    if test -x "$cuda_prefix/bin/nvcc"; then
        AC_MSG_RESULT([found])
        AC_DEFINE_UNQUOTED([NVCC_PATH], ["$cuda_prefix/bin/nvcc"], [Path to nvcc binary])
    else
        AC_MSG_RESULT([not found!])
        AC_MSG_FAILURE([nvcc was not found in $cuda_prefix/bin])
    fi
    CUDA_CFLAGS="-I$cuda_prefix/include"
    CUDA_LDFLAGS="-L$cuda_prefix/lib64 -L$cuda_prefix/lib"
    CUDA_LIBS="-lcublas -lcudart -lcurand"
    LIBS="$LIBS $CUDA_LIBS"
    LDFLAGS="$LDFLAGS $CUDA_LDFLAGS -L./"
    LIBTOOL='LD_LIBRARY_PATH=$(PWD) $(SHELL) $(top_builddir)/libtool'
    NVCC="nvcc"
  DEBUG="-DUSE_GPU"
    AC_DEFINE(DCUDA,[1],[Defined if CUDA should be used])
  AC_CHECK_LIB([cublas], [main], [], [
     AC_MSG_ERROR([unable to find cuda library])
    ])
  AC_CHECK_LIB([cudart], [main], [], [
      AC_MSG_ERROR([unable to find cudart library])
    ])
  AC_CHECK_LIB([curand], [main], [], [
      AC_MSG_ERROR([unable to find curand library])
    ])
else
    CUDA_CFLAGS=""
    CUDA_LDFLAGS=""
    CUDA_LIBS=""
    NVCC=""
    DEBUG="-DCPU_ONLY"
fi
AC_SUBST(LIBTOOL)
AC_SUBST(NVCC)
AC_SUBST(CUDA_LDFLAGS)
AC_SUBST(CUDA_LIBS)
AC_SUBST(CUDA_CFLAGS)

# Setup custom CUDNN paths
AC_ARG_ENABLE([cudnn],
    [AS_HELP_STRING(--enable-cudnn,enable CUDNN support)],
    [enable_cudnn=yes], [enable_cudnn=no])
AM_CONDITIONAL(DCUDNN, [test "$enable_cudnn" = "yes"])

AC_ARG_WITH([cudnn],
    [AS_HELP_STRING([--with-cudnn=PATH], [prefix where CUDNN is installed])],
    [cudnn_prefix=$withval], [cudnn_prefix="/usr/local/cuda"])
if test "$cudnn_prefix" == "yes"; then
    if test "$withval" == "yes"; then
        cudnn_prefix="/usr/local/cuda"
    fi
fi

if test x"$enable_cudnn" == x"yes"; then
    CUDNN_CFLAGS="-I$cudnn_prefix/include"
    CUDNN_LDFLAGS="-L$cudnn_prefix/lib64 -L$cudnn_prefix/lib"
    CUDNN_LIBS="-lcudnn"
    LIBS="$LIBS $CUDNN_LIBS"
    LDFLAGS="$LDFLAGS $CUDNN_LDFLAGS"
    DEBUG+=" -DUSE_CUDNN"
    AC_DEFINE(DCUDNN,[1],[Defined if CUDNN should be used])
    AC_CHECK_LIB([cudnn], [main], [], [
        AC_MSG_ERROR([unable to find cudnn library])
        ])
else
    CUDNN_CFLAGS=""
    CUDNN_LDFLAGS=""
    CUDNN_LIBS=""
fi

AC_SUBST(CUDNN_CFLAGS)
AC_SUBST(CUDNN_LDFLAGS)
AC_SUBST(CUDNN_LIBS)

# Checks for libraries.
AC_SEARCH_LIBS([cblas_sgemm], [openblas], [], [
  AC_MSG_ERROR([unable to find cblas_sgemm() function])
  ])
AC_SEARCH_LIBS([zmq_ctx_new], [zmq], [], [
  AC_MSG_ERROR([unable to find zmq_ctx_new() function])
    ])
AC_SEARCH_LIBS([zmsg_new], [czmq], [], [
  AC_MSG_ERROR([unable to find zmsg_new() function])
  ])
AC_CHECK_LIB([glog], [main], [], [
  AC_MSG_ERROR([unable to find glog library])
  ])
AC_CHECK_LIB([protobuf], [main], [], [
  AC_MSG_ERROR([unable to find protobuf library])
  ])

# Setup custom lmdb paths 
AC_ARG_ENABLE(lmdb,
     AS_HELP_STRING([--enable-lmdb],[enable debug option]),
   [enable_lmdb=yes],[enable_lmdb=no])
AM_CONDITIONAL(LMDB, test "$enable_lmdb" = yes)
if test x"$enable_lmdb" = x"yes"; then
  AC_SEARCH_LIBS([mdb_env_create], [lmdb], [], [
    AC_MSG_ERROR([unable to find mdb_env_create() function])
    ])
  AC_DEFINE(LMDB, 1, [Enable Option layer])
fi

PROGS=''
LTLIBS=''
# Setup custom libhdfs paths 
AC_ARG_ENABLE(hdfs,
  AS_HELP_STRING([--enable-hdfs],[enable hdfs support]),
  [enable_hdfs=yes],[enable_hdfs=no])
AM_CONDITIONAL(DHDFS, test "$enable_hdfs" = yes)

AC_ARG_WITH([libhdfs],
    [AS_HELP_STRING([--with-libhdfs=PATH], [prefix where libhdfs is installed])],
    [hdfs_prefix=$withval], [hdfs_prefix="/usr/local"])
if test "$hdfs_prefix" == "yes"; then
    if test "$withval" == "yes"; then
        cudnn_prefix="/usr/local"
    fi
fi

if test x"$enable_hdfs" != x"no"; then
  HDFS_CFLAGS="-I$hdfs_prefix/include"
  HDFS_LDFLAGS="-L$hdfs_prefix/lib"
  HDFS_LIBS="-lhdfs3"
  LIBS="$LIBS $HDFS_LIBS"
  LDFLAGS="$LDFLAGS $HDFS_LDFLAGS"
  DEBUG+=" -DUSE_HDFS"
  AC_DEFINE(DHDFS,[1],[Defined if HDFS should be used])
  AC_CHECK_LIB([hdfs3], [main], [], [
      AC_MSG_ERROR([unable to find hdfs3 library])
      ])
else
  HDFS_CFLAGS=""
  HDFS_LDFLAGS=""
  HDFS_LIBS=""
fi

AC_SUBST(HDFS_CFLAGS)
AC_SUBST(HDFS_LDFLAGS)
AC_SUBST(HDFS_LIBS)

# Setup for test args 
AC_ARG_ENABLE(test,
  AS_HELP_STRING([--enable-test],[enable singa test]),
  [enable_test=yes],[enable_test=no])
AM_CONDITIONAL(SINGATEST, test "$enable_test" = yes)
if test x"$enable_test" != x"no"; then
  PROGS+='singatest test '
  LTLIBS+='libgtest.la '
else
  PROGS+=''
  LTLIBS+=''
fi

# Setup for debug args
AC_ARG_ENABLE(debug,
  AS_HELP_STRING([--enable-debug],[enable debug mode]),
  [enable_debug=yes],[enable_debug=no])
AM_CONDITIONAL(SINGADEBUG, [test "$enable_debug" = yes])
if test x"$enable_debug" == x"yes"; then
  DEBUG+=' -g'
else
  DEBUG+=' -O2'
fi

# Setup for python args
AC_ARG_ENABLE(python,
    [AS_HELP_STRING([--enable-python],[enable python binding])],
    [enable_python=yes],[enable_python=no])
AM_CONDITIONAL([PY], [test "enable_python" = "yes"])
if test x"$enable_python" != x"no"; then
    AC_CHECK_PROG(PY_CHECK,python,yes)
    if test x"$PY_CHECK" != x"yes"; then
        AC_MSG_ERROR([Cannot find command "python". Please intall before make.])
    else
        PYLIBS="-lpython`python -V 2>&1 | awk '{print substr($2,1,3)}'`"
    fi
fi
AC_SUBST(PYLIBS)

# Setup for python paths
AC_ARG_WITH([python],
    [AS_HELP_STRING([--with-python=PATH],[prefix where python is installed])],
    [python_prefix=$withval],[python_prefix="/usr/include/python`python -V 2>&1 | awk '{print substr($2,1,3)}'`"])
if test "$python_prefix" == "yes"; then
    if test "$withval" == "yes"; then
        python_prefix="/usr/include/python`python -V 2>&1 | awk '{print substr($2,1,3)}'`"
    fi
fi


if test x"$enable_python" != x"no"; then
    AC_MSG_CHECKING([Python.h in $python_prefix])
    if test -f "$python_prefix/Python.h"; then
        AC_MSG_RESULT([Python.h found])
        AC_DEFINE_UNQUOTED([PYTHON_PATH], ["$python_prefix"], [Path to python binary])
    else
        AC_MSG_RESULT([Python.h not found!])
        AC_MSG_FAILURE([Python.h was not found in $python_prefix])
    fi
    #AC_DEFINE(PY,[1],[Defined if PY should be used])
    PY_PROGS='_driver.la '
    PYFLAGS="-I$python_prefix "
else
    PY_PROGS=''
    PYFLAGS=''
fi

AC_SUBST([PROGS])
AC_SUBST([LTLIBS])
AC_SUBST([DEBUG])
AC_SUBST([PYFLAGS])
AC_SUBST([PY_PROGS])

# Setup for opencv libs
#AC_CHECK_LIB([opencv_imgproc], [main], [], [
#  AC_MSG_ERROR([unable to find opencv_imgproc lib])
#  ])
#AC_CHECK_LIB([opencv_highgui], [main], [], [
#  AC_MSG_ERROR([unable to find opencv_highgui lib])
#  ])
#AC_CHECK_LIB([opencv_core], [main], [], [
#  AC_MSG_ERROR([unable to find opencv_core lib])
#  ])
AC_CHECK_LIB([zookeeper_mt], [main], [], [
  AC_MSG_ERROR([unable to find zookeeper])
  ])

# Checks for header files.
AC_HEADER_STDC
AC_CHECK_HEADERS([fcntl.h malloc.h stdlib.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_CONST
AC_C_INLINE
AC_TYPE_SIZE_T
AC_C_VOLATILE

# Checks for library functions.
AC_FUNC_MALLOC
AC_FUNC_STAT
AC_CHECK_FUNCS([gethostname memset mkdir pow sqrt])
AC_OUTPUT(Makefile)
