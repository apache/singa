#ifndef SINGA_SINGA_H_
#define SINGA_SINGA_H_
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <cblas.h>

#include "utils/common.h"
#include "proto/job.pb.h"
#include "proto/singa.pb.h"

#include "utils/param.h"
#include "utils/singleton.h"
#include "utils/factory.h"

#include "neuralnet/neuralnet.h"
#include "trainer/trainer.h"
#include "communication/socket.h"

DEFINE_string(singa_conf, "conf/singa.conf", "Global config file");

namespace singa {
void SubmitJob(int job, bool resume, const JobProto& jobConf) {
  SingaProto singaConf;
  ReadProtoFromTextFile(FLAGS_singa_conf.c_str(), &singaConf);
  if (singaConf.has_log_dir())
    SetupLog(singaConf.log_dir(),
        std::to_string(job) + "-" + jobConf.name());
  if (jobConf.num_openblas_threads() != 1)
    LOG(WARNING) << "openblas is set with " << jobConf.num_openblas_threads()
      << " threads";
  openblas_set_num_threads(jobConf.num_openblas_threads());
  JobProto proto;
  proto.CopyFrom(jobConf);
  proto.set_id(job);
  Trainer trainer;
  trainer.Start(resume, singaConf, &proto);
}
}  // namespace singa
#endif  //  SINGA_SINGA_H_

