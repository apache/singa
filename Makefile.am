ACLOCAL_AMFLAGS = -I config
AUTOMAKE_OPTIONS = foreign subdir-objects

AM_CPPFLAGS = -I$(top_srcdir)/src

MSHADOW_FLAGS = -DMSHADOW_USE_CUDA=0 -DMSHADOW_USE_CBLAS=1 -DMSHADOW_USE_MKL=0

AC_CXXFLAGS = -O3 -Wall -pthread -fPIC -std=c++11 -Wno-unknown-pragmas \
              $(MSHADOW_FLAGS) -DCPU_ONLY=1 -funroll-loops

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

SINGA_SRCS := src/utils/cluster.cc \
              src/utils/cluster_rt.cc \
              src/utils/graph.cc \
              src/utils/common.cc \
              src/utils/param.cc \
              src/utils/updater.cc \
              src/utils/data_shard.cc \
              src/utils/blob.cc \
              src/trainer/server.cc \
              src/trainer/worker.cc \
              src/trainer/trainer.cc \
              src/neuralnet/base_layer.cc \
              src/neuralnet/neuralnet.cc \
              src/neuralnet/optional_layer.cc \
              src/neuralnet/layer.cc \
              src/communication/socket.cc \
              src/communication/msg.cc

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
              include/trainer/server.h \
              include/trainer/worker.h \
              include/trainer/trainer.h \
              include/neuralnet/base_layer.h \
              include/neuralnet/layer.h \
              include/neuralnet/optional_layer.h \
              include/neuralnet/neuralnet.h \
              include/mshadow/tensor_expr.h \
              include/mshadow/tensor_container.h \
              include/mshadow/tensor_expr_ext.h \
              include/mshadow/tensor.h \
              include/mshadow/tensor_io.h \
              include/mshadow/cxxnet_op.h \
              include/mshadow/tensor_base.h \
              include/mshadow/tensor_random.h \
              include/communication/msg.h \
              include/communication/socket.h


lib_LTLIBRARIES = libsinga.la
libsinga_la_SOURCES = $(PROTO_HDRS) $(PROTO_SRCS) $(SINGA_HDRS) $(SINGA_SRCS)
libsinga_la_CXXFLAGS = -O3 -Wall -pthread -fPIC -std=c++11 -Wno-unknown-pragmas \
                       $(MSHADOW_FLAGS) -DCPU_ONLY=1 -funroll-loops -gdwarf-2 \
                       -gstrict-dwarf -Woverloaded-virtual -DTHREADED -fpermissive
libsinga_la_LDFLAGS = -I./include

bin_PROGRAMS = singa
singa_SOURCES = src/main.cc
singa_CXXFLAGS = -O3 -Wall -pthread -fPIC -std=c++11 -MMD -Wno-unknown-pragmas \
                 $(MSHADOW_FLAGS) -DCPU_ONLY=1 -funroll-loops -DTHREADED
singa_LDFLAGS = -I./include \
                -lsinga \
                -lglog  \
                -lgflags \
                -lprotobuf \
                -lrt \
                -lopencv_highgui \
                -lopencv_imgproc \
                -lopencv_core \
                -lopenblas \
                -lzmq \
                -lczmq \
                -llmdb \
                -lzookeeper_mt

bin_PROGRAMS += singatool
singatool_SOURCES = src/utils/tool.cc
singatool_CXXFLAGS = -O3 -Wall -pthread -fPIC -std=c++11 -MMD -Wno-unknown-pragmas \
                     -funroll-loops -DTHREADED
singatool_LDFLAGS = -I./include \
                    -lsinga \
                    -lglog  \
                    -lgflags \
                    -lprotobuf \
                    -lzookeeper_mt

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
