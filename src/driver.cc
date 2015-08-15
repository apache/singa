#include "singa.h"

namespace singa {

/**
 * the job and singa_conf arguments are passed by the singa script which is
 * transparent to users
 */
DEFINE_int32(job, -1, "Unique job ID generated from singa-run.sh");
DEFINE_string(singa_conf, "conf/singa.conf", "Global config file");

void Driver::Init(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  job_id_ = FLAGS_job;

  // register layers
  RegisterLayer<BridgeDstLayer>(kBridgeDst);
  RegisterLayer<BridgeSrcLayer>(kBridgeSrc);
  RegisterLayer<ConvolutionLayer>(kConvolution);
  RegisterLayer<ConcateLayer>(kConcate);
  RegisterLayer<DropoutLayer>(kDropout);
  RegisterLayer<InnerProductLayer>(kInnerProduct);
  RegisterLayer<LabelLayer>(kLabel);
  RegisterLayer<LRNLayer>(kLRN);
  RegisterLayer<MnistLayer>(kMnist);
  RegisterLayer<PrefetchLayer>(kPrefetch);
  RegisterLayer<PoolingLayer>(kPooling);
  RegisterLayer<RGBImageLayer>(kRGBImage);
  RegisterLayer<ReLULayer>(kReLU);
  RegisterLayer<ShardDataLayer>(kShardData);
  RegisterLayer<SliceLayer>(kSlice);
  RegisterLayer<SoftmaxLossLayer>(kSoftmaxLoss);
  RegisterLayer<SplitLayer>(kSplit);
  RegisterLayer<TanhLayer>(kTanh);
  RegisterLayer<RBMVisLayer>(kRBMVis);
  RegisterLayer<RBMHidLayer>(kRBMHid);
#ifdef USE_LMDB
  RegisterLayer<LMDBDataLayer>(kLMDBData);
#endif

  // register updater
  RegisterUpdater<AdaGradUpdater>(kAdaGrad);
  RegisterUpdater<NesterovUpdater>(kNesterov);
  // TODO(wangwei) RegisterUpdater<kRMSPropUpdater>(kRMSProp);
  RegisterUpdater<SGDUpdater>(kSGD);

  // register worker
  RegisterWorker<BPWorker>(kBP);
  RegisterWorker<CDWorker>(kCD);

  // register param
  RegisterParam<Param>(0);
}

template<typename T>
int Driver::RegisterLayer(int type) {
  auto factory = Singleton<Factory<singa::Layer>>::Instance();
  factory->Register(type, CreateInstance(T, Layer));
  return 1;
}

template<typename T>
int Driver::RegisterParam(int type) {
  auto factory = Singleton<Factory<singa::Param>>::Instance();
  factory->Register(type, CreateInstance(T, Param));
  return 1;
}

template<typename T>
int Driver::RegisterUpdater(int type) {
  auto factory = Singleton<Factory<singa::Updater>>::Instance();
  factory->Register(type, CreateInstance(T, Updater));
  return 1;
}

template<typename T>
int Driver::RegisterWorker(int type) {
  auto factory = Singleton<Factory<singa::Worker>>::Instance();
  factory->Register(type, CreateInstance(T, Worker));
  return 1;
}

void Driver::Submit(bool resume, const JobProto& jobConf) {
  SingaProto singaConf;
  ReadProtoFromTextFile(FLAGS_singa_conf.c_str(), &singaConf);
  if (singaConf.has_log_dir())
    SetupLog(singaConf.log_dir(), std::to_string(FLAGS_job)
        + "-" + jobConf.name());
  if (jobConf.num_openblas_threads() != 1)
    LOG(WARNING) << "openblas with "
      << jobConf.num_openblas_threads() << " threads";
  openblas_set_num_threads(jobConf.num_openblas_threads());

  JobProto job;
  job.CopyFrom(jobConf);
  job.set_id(job_id_);
  Trainer trainer;
  trainer.Start(resume, singaConf, &job);
}

}  // namespace singa
