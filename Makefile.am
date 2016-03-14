ACLOCAL_AMFLAGS = -I config
AUTOMAKE_OPTIONS = foreign subdir-objects

#AM_CPPFLAGS = -I$(top_srcdir)/src
#AM_LDFLAGS = $(LD_FLAGS) 

MSHADOW_FLAGS = -DMSHADOW_USE_CUDA=0 -DMSHADOW_USE_CBLAS=1 -DMSHADOW_USE_MKL=0
DEFAULT_FLAGS = -Wall -pthread -fPIC -std=c++11 -Wno-unknown-pragmas \
              $(MSHADOW_FLAGS) -funroll-loops -DTHREADED

CFLAGS = $(DEBUG)
CXXFLAGS = $(DEBUG)
#AC_CXXFLAGS = $(DEBUG)

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
PROTO_PYS := tool/python/pb2/singa_pb2.py \
             tool/python/pb2/job_pb2.py \
             tool/python/pb2/common_pb2.py

CUDA_SRCS := src/utils/math_kernel.cu
CUDA_OBJS := src/utils/math_kernel.o
CUDA_HDRS := include/singa/utils/math_kernel.h

CUDNN_SRCS := src/neuralnet/loss_layer/cudnn_softmaxloss.cc \
			  src/neuralnet/neuron_layer/cudnn_softmax.cc \
			  src/neuralnet/neuron_layer/cudnn_pooling.cc \
			  src/neuralnet/neuron_layer/cudnn_activation.cc \
			  src/neuralnet/neuron_layer/cudnn_lrn.cc \
			  src/neuralnet/neuron_layer/cudnn_convolution.cc

PY_SRCS := tool/python/singa/driver_wrap.cxx \
		   src/driver.cc

HDFS_SRCS := src/io/hdfsfile.cc \
			 src/io/hdfsfile_store.cc
HDFS_HDRS := include/singa/io/hdfsfile.h \
			 include/singa/io/hdfsfile_store.h

SINGA_SRCS := src/driver.cc \
              src/server.cc \
              src/worker.cc \
              src/stub.cc \
              src/neuralnet/layer.cc \
              src/neuralnet/connection_layer/bridge.cc \
              src/neuralnet/connection_layer/concate.cc \
              src/neuralnet/connection_layer/slice.cc \
              src/neuralnet/connection_layer/split.cc \
              src/neuralnet/connection_layer/rnn_dummy.cc \
              src/neuralnet/input_layer/char_rnn.cc \
              src/neuralnet/input_layer/onehot.cc \
              src/neuralnet/input_layer/csv.cc \
              src/neuralnet/input_layer/image_preprocess.cc \
              src/neuralnet/input_layer/prefetch.cc \
              src/neuralnet/input_layer/record.cc \
              src/neuralnet/input_layer/deprecated.cc \
              src/neuralnet/input_layer/store.cc \
              src/neuralnet/input_layer/rnn_label.cc \
              src/neuralnet/output_layer/accuracy.cc \
              src/neuralnet/output_layer/argsort.cc \
              src/neuralnet/output_layer/csv.cc \
              src/neuralnet/output_layer/record.cc \
              src/neuralnet/output_layer/char_rnn.cc \
              src/neuralnet/loss_layer/euclidean.cc \
              src/neuralnet/loss_layer/softmax.cc \
              src/neuralnet/neuron_layer/activation.cc \
              src/neuralnet/neuron_layer/convolution.cc \
              src/neuralnet/neuron_layer/dropout.cc \
              src/neuralnet/neuron_layer/dummy.cc \
              src/neuralnet/neuron_layer/embedding.cc \
              src/neuralnet/neuron_layer/inner_product.cc \
              src/neuralnet/neuron_layer/lrn.cc \
              src/neuralnet/neuron_layer/pooling.cc \
              src/neuralnet/neuron_layer/rbm.cc \
              src/neuralnet/neuron_layer/gru.cc \
              src/neuralnet/neuron_layer/relu.cc \
              src/neuralnet/neuron_layer/sigmoid.cc \
              src/neuralnet/neuron_layer/softmax.cc \
              src/neuralnet/neuron_layer/stanh.cc \
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
			  include/singa/utils/math_blob.h \
			  include/singa/utils/math_addr.h \
			  include/singa/utils/cluster.h \
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
              include/neuralnet/output_layer.h \
              include/neuralnet/input_layer.h \
              include/neuralnet/loss_layer.h \
              include/neuralnet/neuron_layer.h \
              include/neuralnet/connection_layer.h \
              include/neuralnet/neuralnet.h \
              include/singa/comm/msg.h \
              include/singa/comm/socket.h \
              include/singa/io/store.h \
              include/singa/io/kvfile.h \
              include/singa/io/kvfile_store.h \
              include/singa/io/textfile_store.h \
              include/mshadow/cxxnet_op.h \
              include/mshadow/tensor_expr.h \
              include/mshadow/tensor_container.h \
              include/mshadow/tensor_expr_ext.h \
              include/mshadow/tensor.h \
              include/mshadow/tensor_io.h \
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
						 src/test/test_connection_layers.cc \
						 src/test/test_record_input_layer.cc \
						 src/test/test_csv_input_layer.cc \
             					 src/test/test_gru_layer.cc \
						 src/test/test_unrolling.cc

#EXTRA_PROGRAMS = $(PROGS)
EXTRA_PROGRAMS = singatest test
#EXTRA_LTLIBRARIES = $(LTLIBS)
EXTRA_LTLIBRARIES = libgtest.la _driver.la

lib_LTLIBRARIES = libsinga.la $(LTLIBS)
bin_PROGRAMS = singa singatool $(PROGS)
pydir = $(CURDIR)/tool/python/singa/
py_LTLIBRARIES = $(PY_PROGS)

#lib_LTLIBRARIES = libsinga.la
libsinga_la_SOURCES = $(PROTO_SRCS) $(SINGA_SRCS)
libsinga_la_CXXFLAGS = $(DEFAULT_FLAGS) -msse3 -fpermissive -I$(top_srcdir)/include
libsinga_la_LDFLAGS =
if LMDB
libsinga_la_CXXFLAGS += -DUSE_LMDB
endif
if DCUDA
libsinga_la_SOURCES += $(CUDA_SRCS) $(CUDA_HDRS)
libsinga_la_CXXFLAGS += $(CUDA_CFLAGS)
libsinga_la_LDFLAGS += $(CUDA_LDFLAGS) $(CUDA_LIBS) -L./ -lsingagpu -Wl,-rpath=.
libsinga_la_LIBADD = libsingagpu.so
endif

if DCUDNN
libsinga_la_SOURCES += $(CUDNN_SRCS)
libsinga_la_CXXFLAGS += $(CUDNN_CFLAGS)
libsinga_la_LDFLAGS += $(CUDNN_LDFLAGS) $(CUDNN_LIBS)
endif

if DHDFS
libsinga_la_SOURCES += $(HDFS_SRCS)
libsinga_la_CXXFLAGS += $(HDFS_CFLAGS)
libsinga_la_LDFLAGS += $(HDFS_LDFLAGS) $(HDFS_LIBS)
endif

#bin_PROGRAMS = singa
singa_SOURCES = src/main.cc
singa_CXXFLAGS = $(DEFAULT_FLAGS) -MMD -I$(top_srcdir)/include
singa_LDFLAGS = -lsinga \
                -lglog  \
                -lprotobuf \
                -lopenblas \
                -lzmq \
                -lczmq \
                -lzookeeper_mt
if LMDB
singa_LDFLAGS += -llmdb
endif

if DCUDA
singa_SOURCES += $(CUDA_SRCS) $(CUDA_HDRS)
singa_CXXFLAGS += $(CUDA_CFLAGS)
singa_LDFLAGS += $(CUDA_LDFLAGS) $(CUDA_LIBS)
endif

if DCUDNN
singa_SOURCES += $(CUDNN_SRCS)
singa_CXXFLAGS += $(CUDNN_CFLAGS)
singa_LDFLAGS += $(CUDNN_LDFLAGS) $(CUDNN_LIBS)
endif

if DHDFS
singa_SOURCES += $(HDFS_SRCS)
singa_CXXFLAGS += $(HDFS_CFLAGS)
singa_LDFLAGS += $(HDFS_LDFLAGS) $(HDFS_LIBS)
endif
#bin_PROGRAMS += singatool
singatool_SOURCES = src/utils/tool.cc #$(CUDA_SRCS) $(CUDA_HDRS) $(CUDNN_SRCS)
singatool_CXXFLAGS = -Wall -pthread -fPIC -std=c++11 -MMD -Wno-unknown-pragmas \
					 -funroll-loops -DTHREADED -I$(top_srcdir)/include $(DEFAULT_FLAGS)
singatool_LDFLAGS = -lsinga \
                    -lglog  \
                    -lprotobuf \
                    -lzookeeper_mt

if DHDFS
singatool_SOURCES += $(HDFS_SRCS)
singatool_CXXFLAGS += $(HDFS_CFLAGS)
singatool_LDFLAGS += $(HDFS_LDFLAGS) $(HDFS_LIBS)
endif
#lib_LTLIBRARIES += libgtest.la
libgtest_la_SOURCES = $(GTEST_HDRS) $(GTEST_SRCS)
libgtest_la_CXXFLAGS = $(DEFAULT_FLAGS) -msse3 -fpermissive -I$(top_srcdir)/include
if LMDB
libgtest_la_CXXFLAGS += -DUSE_LMDB
endif
#libgtest_la_LDFLAGS = -I$(top_srcdir)/include

#bin_PROGRAMS += test

singatest_SOURCES = $(GTEST_HDRS) $(TEST_SRCS)
singatest_CXXFLAGS = $(DEFAULT_FLAGS) -I$(top_srcdir)/include
singatest_LDADD = ./libgtest.la
singatest_LDFLAGS = -lsinga \
                -lglog  \
                -lprotobuf \
                -lopenblas \
                -lzmq \
                -lczmq \
                -lzookeeper_mt \
			    -lgtest
if LMDB
singatest_LDFLAGS += -llmdb
endif

if DCUDA
singatest_SOURCES += $(CUDA_SRCS) $(CUDA_HDRS)
singatest_CXXFLAGS += $(CUDA_CFLAGS)
singatest_LDFLAGS += $(CUDA_LDFLAGS) $(CUDA_LIBS)
endif

if DCUDNN
singatest_SOURCES += $(CUDNN_SRCS)
singatest_CXXFLAGS += $(CUDNN_CFLAGS)
singatest_LDFLAGS += $(CUDNN_LDFLAGS) $(CUDNN_LIBS)
endif

_driver_la_SOURCES = $(PY_SRCS)
_driver_la_CXXFLAGS = $(DEFAULT_FLAGS) $(MSHADOW_FLAGS) -I$(top_srcdir)/include $(PYFLAGS)
_driver_la_LDFLAGS = -lsinga -module -shared $(PYLIBS) -avoid-version -rpath $(pydir)

if DCUDA
_driver_la_CXXFLAGS += $(CUDA_CFLAGS)
_driver_la_LDFLAGS += $(CUDA_LDFLAGS) $(CUDA_LIBS)
endif

if DCUDNN
_driver_la_CXXFLAGS += $(CUDNN_CFLAGS)
_driver_la_LDFLAGS += $(CUDNN_LDFLAGS) $(CUDNN_LIBS)
endif

clean-local:
	rm -rf $(PROTO_SRCS) $(PROTO_HDRS)
	rm -rf $(PROTO_PYS)
	rm -rf neuralnet/neuron_layer/*.o
	rm -rf src/utils/math_kernel.o
	rm -rf rat_check
	rm -rf tool/python/pb2
	rm -rf libsingagpu.so

# Add scrips for py driver installation
all-local:
	@if [ -f ".libs/_driver.so" ]; then \
		echo "Copy libs for python wrapper"; \
		cp -f .libs/_driver.so tool/python/singa/; \
		touch tool/python/singa/__init__.py; \
	fi

# For rat check
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

.cu.o: .cu
	$(NVCC) $(MSHADOW_FLAGS) --shared -Xcompiler -fPIC $(CUDA_CFLAGS) $(CUDA_LDFLAGS) $(CUDA_LIBS) -I$(top_srcdir)/include -std=c++11 -G -c -o $@ $<

# Generate gpu libs for singa
libsingagpu.so: $(CUDA_OBJS)
	$(NVCC) -o libsingagpu.so -shared -Xcompiler -fPIC $(CUDA_OBJS)  $(CUDA_CFLAGS) $(CUDA_LDFLAGS) $(CUDA_LIBS) -I$(top_srcdir)/include -std=c++11 -G

# Create python class files
install-pyLTLIBRARIES: $(py_LTLIBRARIES)
	touch tool/python/singa/__init__.py
	cp -f .libs/_driver.so tool/python/singa/

uninstall-pyLTLIBRARIES:
	rm -f tool/python/singa/__init__.py
	rm -f tool/python/singa/_driver.so

# For autorun singatest
test: singatest
	@./singatest

$(PROTO_HDRS) $(PROTO_SRCS): $(PROTOS)
	protoc --proto_path=$(top_srcdir)/src/proto --cpp_out=$(top_srcdir)/src/proto $(PROTOS)
	mkdir -p $(top_srcdir)/tool/python/pb2/
	touch $(top_srcdir)/tool/python/pb2/__init__.py
	protoc --proto_path=$(top_srcdir)/src/proto --python_out=$(top_srcdir)/tool/python/pb2 $(PROTOS)
	mkdir -p $(top_srcdir)/include/singa/proto/
	cp $(top_srcdir)/src/proto/*.pb.h $(top_srcdir)/include/singa/proto/
	@echo
