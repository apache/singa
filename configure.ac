#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ(2.59)
AC_INIT(singa, 0.1, dev@singa.incubator.apache.org)
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
AC_PROG_LIBTOOL

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

AC_CHECK_LIB([opencv_imgproc], [main], [], [
  AC_MSG_ERROR([unable to find opencv_imgproc lib])
  ])
AC_CHECK_LIB([opencv_highgui], [main], [], [
  AC_MSG_ERROR([unable to find opencv_highgui lib])
  ])
AC_CHECK_LIB([opencv_core], [main], [], [
  AC_MSG_ERROR([unable to find opencv_core lib])
  ])
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
