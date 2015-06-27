#CXX_FLAGS=-I $(HOME)/local/include -I $(HOME)/local/include/mesos
LD_FLAGS=-lmesos -lgflags -lglog
EXE=singa_scheduler
OBJS=SingaFramework.o
CXX=g++

all: $(EXE)

$(EXE): $(OBJS)
	$(CXX) -o $@ $(OBJS) $(LD_FLAGS)
	rm -rf *.o

%.o: %.cc
	$(CXX) $(CXX_FLAGS) -c $< 		
