#ifndef SINGA_SINGA_H_
#define SINGA_SINGA_H_
#include <gflags/gflags.h>
#include <glog/logging.h>

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
        std::to_string(job) + "-" + jobConf.model().name());
  Trainer trainer;
  trainer.Start(job, resume, jobConf, singaConf);
}
} /* singa */
#endif  //  SINGA_SINGA_H_

