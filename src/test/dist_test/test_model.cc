// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-08-02 14:13
#include <glog/logging.h>
#include <gflags/gflags.h>


#include "model/sgd_trainer.h"
#include "model/net.h"
#include "proto/model.pb.h"
#include "utils/proto_helper.h"

DEFINE_int32(v, 1, "vlog");

int main(int argc, char** argv) {
  FLAGS_logtostderr=1;
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  lapis::ModelProto model_proto;
  lapis::ReadProtoFromTextFile("examples/imagenet12/model.conf", &model_proto);
  lapis::SGDTrainer trainer;
  trainer.Init(model_proto.trainer());
  lapis::Net net;
  net.Init(model_proto.net());
  trainer.Run(&net);
}
