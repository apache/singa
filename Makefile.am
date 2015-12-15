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
              src/server.cc \
              src/worker.cc \
              src/stub.cc \
              src/neuralnet/layer.cc \
              src/neuralnet/connection_layer/bridge.cc \
              src/neuralnet/connection_layer/concate.cc \
              src/neuralnet/connection_layer/slice.cc \
              src/neuralnet/connection_layer/split.cc \
              src/neuralnet/input_layer/csv.cc \
              src/neuralnet/input_layer/image_preprocess.cc \
              src/neuralnet/input_layer/prefetch.cc \
              src/neuralnet/input_layer/record.cc \
              src/neuralnet/input_layer/deprecated.cc \
              src/neuralnet/input_layer/store.cc \
              src/neuralnet/output_layer/csv.cc \
              src/neuralnet/output_layer/record.cc \
              src/neuralnet/loss_layer/euclidean.cc \
              src/neuralnet/loss_layer/softmax.cc \
              src/neuralnet/neuron_layer/argsort.cc \
              src/neuralnet/neuron_layer/convolution.cc \
              src/neuralnet/neuron_layer/dropout.cc \
              src/neuralnet/neuron_layer/inner_product.cc \
              src/neuralnet/neuron_layer/lrn.cc \
              src/neuralnet/neuron_layer/pooling.cc \
              src/neuralnet/neuron_layer/rbm.cc \
              src/neuralnet/neuron_layer/gru.cc \
              src/neuralnet/neuron_layer/relu.cc \
              src/neuralnet/neuron_layer/sigmoid.cc \
              src/neuralnet/neuron_layer/stanh.cc \
              src/neuralnet/neuron_layer/softmax.cc \
              src/neuralnet/neuralnet.cc \
              src/comm/socket.cc \
              src/comm/msg.cc \
              src/io/kvfile.cc \
              src/io/kvfile_store.cc \
              src/io/textfile_store.cc \
              src/io/store.cc \
              src/utils/cluster.cc \
              src/utils/cluster_rt.cc \
              src/utils/graph.cc \
              src/utils/common.cc \
              src/utils/param.cc \
              src/utils/updater.cc \
              src/utils/blob.cc \
              src/utils/image_transform.cc

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
              include/utils/tokenizer.h \
              include/utils/image_transform.h \
              include/server.h \
              include/worker.h \
              include/stub.h \
              include/neuralnet/layer.h \
              include/neuralnet/output_layer/csv.h \
              include/neuralnet/output_layer/record.h \
              include/neuralnet/neuralnet.h \
              include/singa/comm/msg.h \
              include/singa/comm/socket.h \
              include/singa/io/store.h \
              include/singa/io/kvfile.h \
              include/singa/io/kvfile_store.h \
              include/singa/io/textfile_store.h \
              include/mshadow/tensor_expr.h \
              include/mshadow/tensor_container.h \
              include/mshadow/tensor_expr_ext.h \
              include/mshadow/tensor.h \
              include/mshadow/tensor_io.h \
              include/mshadow/cxxnet_op.h \
              include/mshadow/tensor_base.h \
              include/mshadow/tensor_random.h

GTEST_SRCS := include/gtest/gtest-all.cc
GTEST_HRDS := include/gtest/gtest.h
TEST_SRCS := include/gtest/gtest_main.cc \
						 src/test/test_cluster.cc \
						 src/test/test_common.cc \
						 src/test/test_msg.cc \
						 src/test/test_math.cc \
						 src/test/test_neuralnet.cc \
						 src/test/test_paramslicer.cc \
						 src/test/test_kvfile.cc \
						 src/test/test_store.cc \
						 src/test/test_record_input_layer.cc \
						 src/test/test_csv_input_layer.cc \
             					 src/test/test_gru_layer.cc \
						 src/test/test_unrolling.cc

#EXTRA_PROGRAMS = $(PROGS)
EXTRA_PROGRAMS = singatest test
#EXTRA_LTLIBRARIES = $(LTLIBS)
EXTRA_LTLIBRARIES = libgtest.la

lib_LTLIBRARIES = libsinga.la $(LTLIBS)
bin_PROGRAMS = singa singatool $(PROGS)

#lib_LTLIBRARIES = libsinga.la
libsinga_la_SOURCES = $(PROTO_SRCS) $(SINGA_SRCS)
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
                #-lrt \
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

#bin_PROGRAMS += test

singatest_SOURCES = $(GTEST_HDRS) $(TEST_SRCS)
singatest_CXXFLAGS = $(DEFAULT_FLAGS)
singatest_LDADD = ./libgtest.la
singatest_LDFLAGS = -I./include \
                -lsinga \
                -lglog  \
                -lprotobuf \
                #-lrt \
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
	rm -rf rat_check
	rm -rf tool/pb2

rat:
	@if test ! -z '$(shell command -v java 2>/dev/null)'; then \
		if test ! -z '$(shell echo $$RAT_PATH)'; then \
			make distclean;\
			java -jar $(RAT_PATH) -E rat-excludes -d . > rat_check; \
		else \
			echo "RAT_PATH is not set to correct jar file. Apache RAT can be downloaded at http://creadur.apache.org/rat/download_rat.cgi"; \
		fi \
	else \
		echo "java is not found"; \
	fi

test: singatest
	@./singatest

$(PROTO_HDRS) $(PROTO_SRCS): $(PROTOS)
	protoc --proto_path=$(top_srcdir)/src/proto --cpp_out=$(top_srcdir)/src/proto $(PROTOS)
	mkdir -p $(top_srcdir)/tool/pb2/
	touch $(top_srcdir)/tool/pb2/__init__.py
	protoc --proto_path=$(top_srcdir)/src/proto --python_out=$(top_srcdir)/tool/pb2 $(PROTOS)
	mkdir -p $(top_srcdir)/include/singa/proto/
	cp $(top_srcdir)/src/proto/*.pb.h $(top_srcdir)/include/singa/proto/
	@echo
