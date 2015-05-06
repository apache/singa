CLOCAL_AMFLAGS = -I config 

AM_CPPFLAGS = -I$(top_srcdir)src

MSHADOW_FLAGS = -DMSHADOW_USE_CUDA=0 -DMSHADOW_USE_CBLAS=1 -DMSHADOW_USE_MKL=0
AC_CXXFLAGS = -O3 -Wall -pthread -fPIC -std=c++11 -Wno-unknown-pragmas \
    $(MSHADOW_FLAGS) -DCPU_ONLY=1 -funroll-loops 

INCLUDES=-I$(top_srcdir)/include

PROTOS := src/proto/cluster.proto src/proto/model.proto
PROTO_SRCS := src/proto/cluster.pb.cc src/proto/model.pb.cc
PROTO_HDRS := src/proto/cluster.pb.h  src/proto/model.pb.h
PROTO_OBJS := src/proto/cluster.pb.o src/proto/model.pb.o

SINGA_SRCS := src/utils/cluster.cc \
			  src/utils/graph.cc \
			  src/utils/common.cc \
			  src/utils/param.cc \
			  src/utils/updater.cc \
			  src/utils/data_shard.cc \
			  src/utils/blob.cc \
			  src/trainer/server.cc \
			  src/trainer/worker.cc \
			  src/trainer/pm_server.cc \
			  src/trainer/trainer.cc \
			  src/trainer/pm_worker.cc \
			  src/neuralnet/base_layer.cc \
			  src/neuralnet/neuralnet.cc \
			  src/neuralnet/layer.cc \
			  src/communication/socket.cc \
			  src/communication/msg.cc 
SINGA_HDRS := include/utils/cluster.h \
			  include/utils/param.h \
			  include/utils/common.h \
			  include/utils/factory.h \
			  include/utils/data_shard.h \
			  include/utils/singleton.h \
			  include/utils/graph.h \
			  include/utils/blob.h \
			  include/utils/updater.h \
			  include/trainer/server.h \
			  include/trainer/worker.h \
			  include/trainer/pm_worker.h \
			  include/trainer/trainer.h \
			  include/trainer/pm_server.h \
			  include/neuralnet/base_layer.h \
			  include/neuralnet/layer.h \
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

SINGA_OBJS := src/proto/cluster.pb.o \
			  src/proto/model.pb.o \
			  src/utils/cluster.o \
			  src/utils/graph.o \
			  src/utils/common.o \
			  src/utils/param.o \
			  src/utils/updater.o \
			  src/utils/data_shard.o \
			  src/utils/blob.o \
			  src/trainer/server.o \
			  src/trainer/worker.o \
			  src/trainer/pm_server.o \
			  src/trainer/trainer.o \
			  src/trainer/pm_worker.o \
			  src/neuralnet/base_layer.o \
			  src/neuralnet/neuralnet.o \
			  src/neuralnet/layer.o \
			  src/communication/socket.o \
			  src/communication/msg.o 

lib_LTLIBRARIES = libsinga.la
libsinga_la_SOURCES = $(PROTO_HDRS) $(PROTO_SRCS) $(SINGA_SRCS) $(SINGA_HDRS)
libsinga_la_CXXFLAGS = -O3 -Wall -pthread -fPIC -std=c++11 -Wno-unknown-pragmas \
    $(MSHADOW_FLAGS) -DCPU_ONLY=1 -funroll-loops -gdwarf-2 -gstrict-dwarf -Woverloaded-virtual
libsinga_la_LDFLAGS = -I./include 

bin_PROGRAMS=singa 
singa_SOURCES=src/main.cc
singa_CXXFLAGS = -O3 -Wall -pthread -fPIC -std=c++11 -MMD -Wno-unknown-pragmas \
    $(MSHADOW_FLAGS) -DCPU_ONLY=1 -funroll-loops
singa_LDFLAGS =	-I./include \
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
				-llmdb

clean-local:
	rm -rf $(PROTO_SRCS) $(PROTO_HDRS) 


$(PROTO_HDRS) $(PROTO_SRCS): $(PROTOS)
	protoc --proto_path=src/proto --cpp_out=src/proto $(PROTOS)
	mkdir -p include/proto/
	cp src/proto/*.pb.h include/proto/
	@echo
