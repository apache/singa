#include <gflags/gflags.h>
#include <glog/logging.h>
#include "trainer/trainer.h"

int main(int argc, char** argv) {
	FLAGS_logtostderr=1;
	google::InitGoogleLogging(argv[0]);
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	singa::ModelProto model_proto;
	singa::ReadProtoFromTextFile("examples/mnist/model.conf", &model_proto);
	singa::SGDTrainer trainer;
	trainer.Init(model_proto.trainer());
	singa::Net net;

	net.Init(model_proto.net());
	trainer.Run(&net);

	return 0;
}
