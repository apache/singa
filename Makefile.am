ACLOCAL_AMFLAGS = -I config
AUTOMAKE_OPTIONS = foreign subdir-objects

#AM_CPPFLAGS = -I$(top_srcdir)/src

MSHADOW_FLAGS = -DMSHADOW_USE_CUDA=0 -DMSHADOW_USE_CBLAS=1 -DMSHADOW_USE_MKL=0
DEFAULT_FLAGS = -Wall -pthread -fPIC -std=c++11 -Wno-unknown-pragmas \
              $(MSHADOW_FLAGS) -DCPU_ONLY=1 -funroll-loops -DTHREADED

CFLAGS = $(DEBUG)
CXXFLAGS = $(DEBUG)
AC_CXXFLAGS = $(DEBUG)

INCLUDES = -I$(top_srcdir)/include

PROTOS := $(top_srcdir)/src/proto/singa.proto \
          $(top_srcdir)/src/proto/job.proto \
          $(top_srcdir)/src/proto/common.proto
PROTO_SRCS := src/proto/singa.pb.cc \
              src/proto/job.pb.cc \
              src/proto/common.pb.cc
PROTO_HDRS := include/proto/singa.pb.h \
              include/proto/job.pb.h \
              include/proto/common.pb.h
PROTO_PYS := tool/pb2/singa_pb2.py \
             tool/pb2/job_pb2.py \
             tool/pb2/common_pb2.py

SINGA_SRCS := src/driver.cc \
              src/utils/cluster.cc \
              src/utils/cluster_rt.cc \
              src/utils/graph.cc \
              src/utils/common.cc \
              src/utils/param.cc \
              src/utils/updater.cc \
              src/utils/data_shard.cc \
              src/utils/blob.cc \
              src/server.cc \
              src/worker.cc \
              src/stub.cc \
              src/neuralnet/layer.cc \
              src/neuralnet/connection_layer.cc \
              src/neuralnet/input_layer.cc \
              src/neuralnet/loss_layer.cc \
              src/neuralnet/neuron_layer.cc \
              src/neuralnet/output_layer.cc \
              src/neuralnet/neuralnet.cc \
              src/comm/socket.cc \
              src/comm/msg.cc

SINGA_HDRS := include/singa.h \
              include/utils/cluster.h \
              include/utils/cluster_rt.h \
              include/utils/param.h \
              include/utils/common.h \
              include/utils/factory.h \
              include/utils/data_shard.h \
              include/utils/singleton.h \
              include/utils/graph.h \
              include/utils/blob.h \
              include/utils/updater.h \
              include/utils/tinydir.h \
              include/server.h \
              include/worker.h \
              include/stub.h \
              include/neuralnet/layer.h \
              include/neuralnet/connection_layer.h \
              include/neuralnet/input_layer.h \
              include/neuralnet/loss_layer.h \
              include/neuralnet/neuron_layer.h \
              include/neuralnet/output_layer.h \
              include/neuralnet/neuralnet.h \
              include/mshadow/tensor_expr.h \
              include/mshadow/tensor_container.h \
              include/mshadow/tensor_expr_ext.h \
              include/mshadow/tensor.h \
              include/mshadow/tensor_io.h \
              include/mshadow/cxxnet_op.h \
              include/mshadow/tensor_base.h \
              include/mshadow/tensor_random.h \
              include/comm/msg.h \
              include/comm/socket.h

GTEST_SRCS := include/gtest/gtest-all.cc
GTEST_HRDS := include/gtest/gtest.h
TEST_SRCS := include/gtest/gtest_main.cc \
			 src/test/test_cluster.cc \
             src/test/test_common.cc \
			 src/test/test_msg.cc \
			 src/test/test_neuralnet.cc \
			 src/test/test_paramslicer.cc \
			 src/test/test_shard.cc

#EXTRA_PROGRAMS = $(PROGS)
EXTRA_PROGRAMS = singatest
#EXTRA_LTLIBRARIES = $(LTLIBS)
EXTRA_LTLIBRARIES = libgtest.la

lib_LTLIBRARIES = libsinga.la $(LTLIBS)
bin_PROGRAMS = singa singatool $(PROGS)

#lib_LTLIBRARIES = libsinga.la
libsinga_la_SOURCES = $(PROTO_HDRS) $(PROTO_SRCS) $(SINGA_HDRS) $(SINGA_SRCS)
libsinga_la_CXXFLAGS = $(DEFAULT_FLAGS) -msse3 -fpermissive
if LMDB
libsinga_la_CXXFLAGS += -DUSE_LMDB
endif
libsinga_la_LDFLAGS = -I./include


#bin_PROGRAMS = singa
singa_SOURCES = src/main.cc
singa_CXXFLAGS = $(DEFAULT_FLAGS) -MMD
singa_LDFLAGS = -I./include \
                -lsinga \
                -lglog  \
                -lprotobuf \
                -lrt \
                -lopenblas \
                -lzmq \
                -lczmq \
                -lzookeeper_mt
if LMDB
singa_LDFLAGS += -llmdb
endif

#bin_PROGRAMS += singatool
singatool_SOURCES = src/utils/tool.cc
singatool_CXXFLAGS = -Wall -pthread -fPIC -std=c++11 -MMD -Wno-unknown-pragmas \
                     -funroll-loops -DTHREADED
singatool_LDFLAGS = -I./include \
                    -lsinga \
                    -lglog  \
                    -lprotobuf \
                    -lzookeeper_mt

#lib_LTLIBRARIES += libgtest.la
libgtest_la_SOURCES = $(GTEST_HDRS) $(GTEST_SRCS)
libgtest_la_CXXFLAGS = $(DEFAULT_FLAGS) -msse3 -fpermissive
if LMDB
libgtest_la_CXXFLAGS += -DUSE_LMDB
endif
libgtest_la_LDFLAGS = -I./include

#bin_PROGRAMS += singatest

singatest_SOURCES = $(GTEST_HDRS) $(TEST_SRCS)
singatest_CXXFLAGS = $(DEFAULT_FLAGS)
singatest_LDFLAGS = -I./include \
                -lsinga \
                -lglog  \
                -lprotobuf \
                -lrt \
                -lopenblas \
                -lzmq \
                -lczmq \
                -lzookeeper_mt \
				-lgtest
if LMDB
singatest_LDFLAGS += -llmdb
endif

clean-local:
	rm -rf $(PROTO_SRCS) $(PROTO_HDRS)
	rm -rf $(PROTO_PYS)


$(PROTO_HDRS) $(PROTO_SRCS): $(PROTOS)
	protoc --proto_path=$(top_srcdir)/src/proto --cpp_out=$(top_srcdir)/src/proto $(PROTOS)
	mkdir -p $(top_srcdir)/tool/pb2/
	touch $(top_srcdir)/tool/pb2/__init__.py
	protoc --proto_path=$(top_srcdir)/src/proto --python_out=$(top_srcdir)/tool/pb2 $(PROTOS)
	mkdir -p $(top_srcdir)/include/proto/
	cp $(top_srcdir)/src/proto/*.pb.h $(top_srcdir)/include/proto/
	@echo
