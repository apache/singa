#include <glog/logging.h>
#include <thread>
#include <chrono>
#include <thread>
#include "utils/singleton.h"
#include "utils/cluster.h"
#include "utils/factory.h"
#include "trainer/worker.h"

namespace singa {
using std::thread;

Worker* Worker::Create(const JobProto& proto) {
  auto factory = Singleton<Factory<singa::Worker>>::Instance();
  Worker* worker = nullptr;
  if (proto.has_user_alg())
    worker = factory->Create(proto.user_alg());
  else
    worker = factory->Create(proto.alg());
  return worker;
}
void Worker::Init(int thread_id, int grp_id, int id) {
  thread_id_ = thread_id;
  grp_id_ = grp_id;
  id_ = id;
  layer_dealer_ = dealer_ = nullptr;
  updater_ = nullptr;
}

void Worker::Setup(
    const JobProto& job, shared_ptr<NeuralNet> train_net,
    shared_ptr<NeuralNet> valid_net, shared_ptr<NeuralNet> test_net) {
  job_conf_.CopyFrom(job);
  train_net_ = train_net;
  validation_net_ = valid_net;
  test_net_ = test_net;
}

Worker::~Worker() {
  if (layer_dealer_)
    delete layer_dealer_;
  if (dealer_)
    delete dealer_;
}

void Worker::InitLocalParams() {
  // for each server grp, its first subscriber worker grp does the param init
  if (grp_id_ % Cluster::Get()->nworker_groups_per_server_group() == 0) {
    // extract params that should be initialized by this worker
    // must gen a name for each param if the user doesn't config it
    std::unordered_map<string, Param*> name2param;
    for (auto layer: train_net_->layers()){
      if (layer->partition_id() == id_) {
        for (auto param : layer->GetParams()) {
          // only owners fill the memory of parameter values.
          if(param->owner() == param->id()) {
            CHECK(name2param.find(param->name()) == name2param.end());
            name2param[param->name()] = param;
          }
        }
      }
    }
    // load from checkpoints. get param blob based on param name.
    // the param from previous checkpoint files will be overwritten by
    // the param with the same name in later checkpoint files.
    for (const auto checkpoint : job_conf_.checkpoint_path()) {
      LOG(INFO) << "Load from checkpoint file " << checkpoint;
      BlobProtos bps;
      ReadProtoFromBinaryFile(checkpoint.c_str(), &bps);
      for (int i = 0; i < bps.name_size(); i++) {
        if (name2param.find(bps.name(i)) != name2param.end()) {
          name2param.at(bps.name(i))->FromProto(bps.blob(i));
          //  if load from pre-training params, reset version to start step
          if(job_conf_.reset_param_version())
            name2param.at(bps.name(i))->set_version(job_conf_.step());
          else  // if resume training, use the same version as last checkpoint
            name2param.at(bps.name(i))->set_version(bps.version(i));
        }
      }
    }
    // init other params who do not have checkpoint version
    for (auto entry : name2param)
      if (entry.second->version() < 0) {
        entry.second->InitValues(job_conf_.step());
        if (!job_conf_.reset_param_version())
          LOG(ERROR) << "better reset version of params from checkpoints "
            << "to the same as other newly initialized params!";
      }

    Metric perf;
    // warmup training before put params to servers
    for (; step_ < job_conf_.warmup_steps(); step_++)
      TrainOneBatch(step_, &perf);
    for (auto layer : train_net_->layers()) {
      if (layer->partition_id() == id_)
        for (auto param : layer->GetParams())
          if (param->owner() == param->id())
            Put(param, param->version());
    }
  }
  // wait owners in the same procs init params, then no get requests sent
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  for (auto layer : train_net_->layers()) {
    if (layer->partition_id() == id_)
      for (auto param : layer->GetParams())
        Get(param, job_conf_.warmup_steps());
  }
}

void Worker::Checkpoint(int step, shared_ptr<NeuralNet> net) {
  if (grp_id_ == 0) {
    BlobProtos bps;
    for (auto layer: net->layers()){
      if (layer->partition_id() == id_) {
        for (auto param : layer->GetParams()) {
          // only owners fill the memory of parameter values.
          if(param->owner() == param->id()) {
            auto *blob = bps.add_blob();
            param->ToProto(blob);
            bps.add_version(param->version());
            bps.add_name(param->name());
          }
        }
      }
    }
    char buf[256];
    snprintf(buf, sizeof(buf), "%s/step%d-worker%d.bin",
         Cluster::Get()->checkpoint_folder().c_str(), step, id_);
    LOG(INFO) << "checkpoint to " << buf;
    WriteProtoToBinaryFile(bps, buf);
  }
}

void ConnectStub(int grp, int id, Dealer* dealer, EntityType entity) {
  dealer->Connect(kInprocRouterEndpoint);
  Msg* ping = new Msg(Addr(grp, id, entity), Addr(-1, -1, kStub));
  ping->set_type(kConnect);
  dealer->Send(&ping);
}

void Worker::Run() {
  LOG(ERROR) << "Worker (group = " << grp_id_ <<", id = " << id_ << ") start";
  auto cluster = Cluster::Get();
  if (updater_==nullptr) {
    int svr_grp = grp_id_ / cluster->nworker_groups_per_server_group();
    CHECK(cluster->runtime()->JoinSGroup(grp_id_, id_, svr_grp));
  }
  dealer_ = new Dealer(2*thread_id_);
  ConnectStub(grp_id_, id_, dealer_, kWorkerParam);
  for (auto layer : train_net_->layers()) {
    if (layer->partition_id() == id_) {
      if (layer->is_bridgelayer()) {
        layer_dealer_ = new Dealer(2*thread_id_+1);
        ConnectStub(grp_id_, id_, layer_dealer_, kWorkerLayer);
        break;
      }
    }
  }

  step_ = job_conf_.step();
  InitLocalParams();
  Metric perf;
  while (!StopNow(step_)) {
    if (ValidateNow(step_)) {
      //LOG(ERROR)<<"Validation at step "<<step;
      CollectAll(validation_net_, step_);
      Test(job_conf_.valid_steps(), kValidation, validation_net_);
    }
    if (TestNow(step_)) {
      //LOG(ERROR)<<"Test at step "<<step;
      CollectAll(test_net_, step_);
      Test(job_conf_.test_steps(), kTest, test_net_);
    }

    if (CheckpointNow(step_)) {
      CollectAll(train_net_, step_);
      Checkpoint(step_, train_net_);
      job_conf_.set_step(step_);
    }
    TrainOneBatch(step_, &perf);
    // LOG(ERROR) << "Train " << step_;
    if (DisplayNow(step_)) {
      Report("Train", perf);
      perf.Reset();
    }
    step_++;
  }

  // save the model
  Checkpoint(step_, train_net_);

  // clean up
  if(updater_ == nullptr) {
    int svr_grp = grp_id_ / cluster->nworker_groups_per_server_group();
    cluster->runtime()->LeaveSGroup(grp_id_, id_, svr_grp);
  }
  // notify the stub on worker stop
  Msg* msg=new Msg(Addr(grp_id_, id_, kWorkerParam), Addr(-1,-1, kStub));
  msg->set_type(kStop);
  dealer_->Send(&msg);  // use param dealer to send the stop msg

  LOG(ERROR) << "Worker (group = " <<grp_id_ << ", id = " << id_ << ") stops";
}



int Worker::Put(Param* param, int step) {
  Msg* msg=new Msg(Addr(grp_id_, id_, kWorkerParam), Addr(-1, -1, kStub));
  msg->set_trgt(ParamTrgt(param->owner(), 0), step);
  msg->set_type(kPut);
  dealer_->Send(&msg);
  return 1;
}

int Worker::Get(Param* param, int step) {
  if (param->version() >= step)
    return 1;
  Msg* msg=new Msg(Addr(grp_id_, id_, kWorkerParam), Addr(-1, -1, kStub));
  msg->set_trgt(ParamTrgt(param->owner(), 0), step);
  msg->set_type(kGet);
  dealer_->Send(&msg);
  return 1;
}

int Worker::Update(Param* param, int step) {
  param->set_local_version(param->version());
  if (updater_) {
    updater_->Update(step, param);
    param->set_version(param->version() + 1);
  } else {
    Msg* msg=new Msg(Addr(grp_id_, id_, kWorkerParam), Addr(-1, -1, kStub));
    msg->set_trgt(ParamTrgt(param->owner(), 0), step);
    msg->set_type(kUpdate);
    dealer_->Send(&msg);
  }
  return 1;
}

int Worker::CollectAll(shared_ptr<NeuralNet> net, int step) {
  auto& layers = net->layers();
  for (auto& layer : layers){
    if (layer->partition_id() == id_)
      for (Param* p: layer->GetParams()) {
        Collect(p, step);
      }
  }
  return 1;
}
int Worker::Collect(Param* param, int step) {
  while (param->version() <= param->local_version())
    std::this_thread::sleep_for(std::chrono::milliseconds(kCollectSleepTime));
  return 1;
}
void Worker::Report(const string& prefix, const Metric & perf) {
  Msg* msg = new Msg(Addr(grp_id_, id_, kWorkerParam), Addr(-1, -1, kStub));
  msg->set_trgt(0, step_);
  msg->set_type(kMetric);
  const string disp = perf.ToString();
  msg->AddFormatFrame("s", prefix.c_str());
  msg->AddFrame(disp.c_str(), disp.length());
  dealer_->Send(&msg);
}

void Worker::ReceiveBlobs(
    bool data, bool grad, BridgeLayer* layer, shared_ptr<NeuralNet> net) {
  while (!layer->ready()) {
    auto msg = layer_dealer_->Receive();
    CHECK_EQ(AddrGrp(msg->src()), grp_id_);
    string name(static_cast<char*>(msg->FrameData()), msg->FrameSize());
    auto receive_layer = net->name2layer(name);
    CHECK(receive_layer->is_bridgelayer());
    auto data = receive_layer->mutable_data(nullptr);
    msg->NextFrame();
    memcpy(data->mutable_cpu_data(), msg->FrameData(), msg->FrameSize());
    static_cast<BridgeLayer*>(receive_layer)->set_ready(true);
    delete msg;
  }
}

void Worker::SendBlobs(
    bool data, bool grad, BridgeLayer* layer, shared_ptr<NeuralNet> net) {
  auto dst=layer->dstlayers().at(0);
  Msg *msg=new Msg();
  msg->set_src(Addr(grp_id_, id_, kWorkerLayer));
  msg->set_dst(Addr(grp_id_, dst->partition_id(), kWorkerLayer));
  msg->AddFrame(dst->name().c_str(), dst->name().length());
  auto const & blob=layer->data(nullptr);
  msg->AddFrame(blob.cpu_data(), blob.count()*sizeof(float));
  layer_dealer_->Send(&msg);
}

void Worker::Test(int nsteps, Phase phase, shared_ptr<NeuralNet> net) {
  Metric perf;
  for (int step = 0; step < nsteps; step++)
    TestOneBatch(step, phase, net, &perf);
  if (phase == kValidation)
    Report("Validation", perf);
  else if (phase == kTest)
    Report("Test", perf);
}

bool Worker::DisplayNow(int step) const {
  return (job_conf_.disp_freq() > 0
      && step >= job_conf_.disp_after()
      && ((step - job_conf_.disp_after())
        % job_conf_.disp_freq() == 0));
}

bool Worker::DisplayDebugInfo(int step) const {
  return DisplayNow(step) && job_conf_.debug() && grp_id_ == 0;
}
bool Worker::StopNow(int step) const {
  return step >= job_conf_.train_steps();
}
bool Worker::CheckpointNow(int step) const {
  return (grp_id_ == 0
      && job_conf_.checkpoint_freq() > 0
      && step >= job_conf_.checkpoint_after()
      && ((step - job_conf_.checkpoint_after())
        % job_conf_.checkpoint_freq() == 0));
}
bool Worker::TestNow(const int step) const {
  return (grp_id_ == 0
      && job_conf_.test_freq() > 0
      && job_conf_.test_steps() > 0
      && step >= job_conf_.test_after()
      && ((step - job_conf_.test_after())
        % job_conf_.test_freq() == 0));
}
bool Worker::ValidateNow(const int step) const {
  return (grp_id_ == 0
      && job_conf_.valid_freq() > 0
      && job_conf_.valid_steps() > 0
      && step >= job_conf_.valid_after()
      && ((step - job_conf_.valid_after())
        % job_conf_.valid_freq() == 0));
}


/****************************BPWorker**********************************/
void BPWorker::Init(int thread_id, int group_id, int worker_id) {
  Worker::Init(thread_id, group_id, worker_id);
}

void BPWorker::Forward(
    int step, Phase phase, shared_ptr<NeuralNet> net, Metric* perf) {
  for (auto& layer : net->layers()) {
    if (layer->partition_id() == id_) {
      if (layer->is_bridgedstlayer())  // recv data from other workers
        ReceiveBlobs(true, false, static_cast<BridgeLayer*>(layer), net);
      if (phase == kTrain) {
        for (Param* p : layer->GetParams()) {  // wait until param is updated
          Collect(p, step);
        }
      }
      layer->ComputeFeature(phase, perf);
      if (layer->is_bridgesrclayer())  // send data to other workers
        SendBlobs(true, false, static_cast<BridgeLayer*>(layer), net);
      if (DisplayDebugInfo(step))
        LOG(INFO) << layer->DebugString(step, kForward);
    }
  }
}

void BPWorker::Backward(int step, shared_ptr<NeuralNet> net) {
  auto& layers=net->layers();
  for (auto it = layers.rbegin(); it != layers.rend(); it++){
    Layer* layer = *it;
    if (layer->partition_id() == id_) {
      if(layer->is_bridgesrclayer()) {
        // ReceiveBlobs(false, true, layer, net);
      }
      layer->ComputeGradient(kTrain);
      if (DisplayDebugInfo(step))
        LOG(INFO) << layer->DebugString(step, kBackward);
      for (Param* p : layer->GetParams())
        Update(p, step);
      if (layer->is_bridgedstlayer()) {
        // SendBlobs(false, true, layer);
      }
    }
  }
}

void BPWorker::TrainOneBatch(int step, Metric* perf) {
  Forward(step, kTrain, train_net_, perf);
  Backward(step, train_net_);
}

void BPWorker::TestOneBatch(int step, Phase phase,
    shared_ptr<NeuralNet> net, Metric* perf) {
  Forward(step, phase, net, perf);
}

/****************************CDWorker**********************************/
void CDWorker::Init(int thread_id, int group_id, int worker_id) {
  Worker::Init(thread_id, group_id, worker_id);
}

void CDWorker::PositivePhase(int step,
     shared_ptr<NeuralNet> net, Metric* perf) {
  auto& layers = net->layers();
  for (auto& layer : layers) {
      // clock_t s=clock();
    layer->ComputeFeature(kPositive, perf);
  }
}

void CDWorker::NegativePhase(int step,
     shared_ptr<NeuralNet> net, Metric* perf) {
// for negative phase, gibbs sampling only concerns RBM bottom and top layer
  auto& layers = net->layers();
  for (int i = 0; i < job_conf_.cd_conf().pcd_k(); i++) {
    for (auto& layer : layers) {
      if (layer->is_vislayer() || layer->is_hidlayer())
        layer->ComputeFeature(kNegative, perf);
    }
  }
}

void CDWorker::GradientPhase(int step, shared_ptr<NeuralNet> net) {
  auto& layers = net->layers();
  for (auto& layer : layers) {
      layer->ComputeGradient(kTrain);
      for (Param* p : layer->GetParams()) {
        Update(p, step);
      }
  }
}

void CDWorker::LossPhase(int step, shared_ptr<NeuralNet> net, Metric* perf) {
  auto& layers = net->layers();
  for (auto& layer : layers) {
    if (layer->is_hidlayer())
      layer->ComputeFeature(kLoss, perf);
  }
  for (auto& layer : layers) {
    if (layer->is_vislayer())
      layer->ComputeLoss(perf);
  }
}

void CDWorker::TrainOneBatch(int step, Metric* perf) {
  PositivePhase(step, train_net_, perf);
  NegativePhase(step, train_net_, perf);
  GradientPhase(step, train_net_);
  LossPhase(step, train_net_, perf);
}

void CDWorker::TestOneBatch(int step, Phase phase,
     shared_ptr<NeuralNet> net, Metric* perf) {
  PositivePhase(step, test_net_, perf);
  LossPhase(step, test_net_, perf);
}
}  // namespace singa
