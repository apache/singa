CXX_FLAGS=-I ../include -std=c++11 -I /usr/local/include/hdfs -I ../../include
LD_FLAGS=-lmesos -lsinga -lhdfs3
EXE=scheduler
OBJS=singa_scheduler.o scheduler.pb.o
PROTOS=scheduler.proto
PROTO_HDRS=scheduler.pb.h
PROTO_SRCS=scheduler.pb.cc

CXX=g++

all: $(PROTO_HDRS) $(EXE)

$(PROTO_SRCS) $(PROTO_HDRS): $(PROTOS)
	protoc --cpp_out=. $(PROTOS)

$(EXE): $(OBJS)
	$(CXX) -o $@ $(OBJS) $(LD_FLAGS)
	rm -rf *.o

%.o: %.cc
	$(CXX) $(CXX_FLAGS) -c $<     

clean: 
	rm -rf *.o $(EXE) $(PROTO_HDRS) $(PROTO_SRCS)

