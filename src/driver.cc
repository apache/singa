#include "singa.h"

namespace singa {

void Driver::Init(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);

  //  unique job ID generated from singa-run.sh, passed in as "-singa_job <id>"
  int arg_pos = ArgPos(argc, argv, "-singa_job");
  job_id_ = (arg_pos != -1) ? atoi(argv[arg_pos+1]) : -1;

  //  global signa conf passed by singa-run.sh as "-singa_conf <path>"
  arg_pos = ArgPos(argc, argv, "-singa_conf");
  if (arg_pos != -1)
    ReadProtoFromTextFile(argv[arg_pos+1], &singa_conf_);
  else
    ReadProtoFromTextFile("conf/singa.conf", &singa_conf_);

  //  job conf passed by users as "-conf <path>"
  arg_pos = ArgPos(argc, argv, "-conf");
  CHECK_NE(arg_pos, -1);
  ReadProtoFromTextFile(argv[arg_pos+1], &job_conf_);

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
  if (singa_conf_.has_log_dir())
    SetupLog(singa_conf_.log_dir(), std::to_string(job_id_)
             + "-" + jobConf.name());
  if (jobConf.num_openblas_threads() != 1)
    LOG(WARNING) << "openblas with "
      << jobConf.num_openblas_threads() << " threads";
  openblas_set_num_threads(jobConf.num_openblas_threads());

  JobProto job;
  job.CopyFrom(jobConf);
  job.set_id(job_id_);
  Trainer trainer;
  trainer.Start(resume, singa_conf_, &job);
}

}  // namespace singa
