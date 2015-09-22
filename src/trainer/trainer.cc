/************************************************************
*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
* 
*   http://www.apache.org/licenses/LICENSE-2.0
* 
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*
*************************************************************/

#include <thread>
#include <vector>
#include <map>
#include <chrono>
#include <glog/logging.h>
#include "utils/tinydir.h"
#include <unistd.h>
#include "utils/cluster.h"
#include "utils/common.h"
#include "proto/common.pb.h"
#include "trainer/trainer.h"
#include "mshadow/tensor.h"


namespace singa {
using std::vector;
using std::map;
using std::queue;
using namespace std::chrono;
using std::make_shared;

/***********************Trainer****************************/
Trainer::~Trainer() {
  delete router_;
}

const vector<int> SliceParams(const vector<Param*>& params) {
  // for load-balance among servers in a group and among server groups
  int nserver_grps = Cluster::Get()->nserver_groups();
  int nservers_per_grp = Cluster::Get()->nservers_per_group();
  int lcm = LeastCommonMultiple(nserver_grps, nservers_per_grp);

  // collect sizes of unique Params
  std::vector<int> paramsize;
  for (auto param : params)
    if (param->id() == param->owner())
      paramsize.push_back(param->size());
  // slice into lcm pieces to achieve good load-balance for both intra-group
  // partition (among servers in a group) and inter-group partition (each group
  // is assgined a sub-set of slices)
  auto param_slice = Slice(lcm, paramsize);
  // construct map from Param ID to its slices <slice id, len>
  std::unordered_map<int, vector<std::pair<int, int>>> paramid2slices;
  vector<int> slices;
  auto it = param_slice.begin();
  int slice_id = 0;
  for (auto param : params) {
    if (param->id() == param->owner()) {
      for (int len : *it) {
        slices.push_back(len);
        paramid2slices[param->id()].push_back(std::make_pair(slice_id++, len));
      }
      it++;
    }
  }
  // add slice info for every Param
  for (auto param : params)
    for (auto entry : paramid2slices[param->owner()]) {
      param->AddSlice(entry.first, entry.second);
      LOG(INFO) << "param id " << param->id() << " owner=" << param->owner()
        << ": " << entry.first << ", " << entry.second;
    }
  return slices;
}

void Trainer::SetupWorkerServer(
    const JobProto& job_conf,
    const vector<Worker*>& workers,
    const vector<Server*>& servers) {
  auto cluster = Cluster::Get();
  int grp_size = cluster->nworkers_per_group();
  const auto& net_conf = job_conf.neuralnet();
  auto net = NeuralNet::Create(net_conf, kTrain, grp_size);
  // MUST do SliceParam before share param/net with others
  auto slices = SliceParams(net->params());

  std::unordered_map<int, shared_ptr<NeuralNet>> grp_net;
  int first_grp = workers.size() ? workers.at(0)->grp_id() : -1;
  for (auto worker : workers) {
    int grp_id = worker->grp_id();
    int worker_id = worker->id();
    shared_ptr<NeuralNet> test_net = nullptr, valid_net = nullptr;
    if (grp_net.find(grp_id) == grp_net.end()) {
      if (grp_id == first_grp) {
        //  test are performed only by the first group now. TODO update.
        if (first_grp == 0 && job_conf.test_steps() && worker_id == 0) {
          test_net = NeuralNet::Create(net_conf, kTest, 1); // hard code for exp
          test_net->ShareParamsFrom(net);
        }
        //  validation are performed only by the first group. TODO update.
        if (first_grp == 0 && job_conf.valid_steps() && worker_id == 0) {
          valid_net = NeuralNet::Create(net_conf, kValidation, 1);
          valid_net->ShareParamsFrom(net);
        }
        grp_net[grp_id] = net;
      } else {
        grp_net[grp_id] = NeuralNet::Create(net_conf, kTrain, grp_size);
        if(cluster->share_memory())
          grp_net[grp_id]->ShareParamsFrom(net);
      }
      for (auto layer : grp_net[grp_id]->layers()) {
        bool local = layer->partition_id() >= workers.front()->id()
          && layer->partition_id() <= workers.back()->id();
        for (auto param : layer->GetParams()) {
          int hash = Hash(grp_id, param->owner());
          if (worker_shard_.find(hash) == worker_shard_.end())
            worker_shard_[hash] = new ParamEntry();
          worker_shard_[hash]->AddParam(local, param);
        }
      }
    }
    LOG(INFO) << "grp " << worker->grp_id() << ", worker "
      << worker->id() << " net " << grp_net[grp_id].get();
    worker->Setup(job_conf, grp_net[grp_id], valid_net, test_net);
  }

  //  partition among server groups, each group maintains one sub-set for sync
  auto slice2group = PartitionSlices(cluster->nserver_groups(), slices);
  //  partition within one server group, each server updates for one sub-set
  slice2server_ = PartitionSlices(cluster->nservers_per_group(), slices);

  for (auto server : servers)
    server->Setup(job_conf.updater(), slice2group, slice2server_);
}

vector<Server*> Trainer::CreateServers(const JobProto& job) {
  auto cluster = Cluster::Get();
  vector<Server*> servers;
  if (!cluster->has_server())
    return servers;

  int server_procs = cluster->procs_id();
  // if true, server procs (logical) id starts after worker procs
  if (cluster->server_worker_separate())
    server_procs -= cluster->nworker_procs();
  const vector<int> rng = cluster->ExecutorRng(server_procs,
      cluster->nservers_per_group(),
      cluster->nservers_per_procs());
  int gstart = rng[0], gend = rng[1], start = rng[2], end = rng[3];
  for (int gid = gstart; gid < gend; gid++) {
    for (int sid = start; sid < end; sid++) {
      auto server = new Server(gid, sid);
      servers.push_back(server);
    }
  }
  return servers;
}


vector<Worker*> Trainer::CreateWorkers(int nthreads, const JobProto& job) {
  auto cluster=Cluster::Get();
  vector<Worker*> workers;
  if(!cluster->has_worker())
    return workers;
  const vector<int> rng = cluster->ExecutorRng(cluster->procs_id(),
      cluster->nworkers_per_group(),
      cluster->nworkers_per_procs());
  int gstart = rng[0], gend = rng[1], wstart = rng[2], wend = rng[3];
  for (int gid = gstart; gid < gend; gid++) {
    for (int wid = wstart; wid < wend; wid++) {
      auto *worker = Worker::Create(job);
      worker->Init(nthreads++,gid, wid);
      workers.push_back(worker);
    }
  }
  return workers;
}

void Trainer::Resume(JobProto* jobConf) {
  tinydir_dir dir;
  string folder = Cluster::Get()->checkpoint_folder();
  tinydir_open(&dir, folder.c_str());
  int latest_step = 0;
  // there would be multi checkpoint files (from diff workers) for one step
  vector<string> ck_files;
  // iterate all files to get the files for the last checkpoint
  while (dir.has_next) {
    tinydir_file file;
    tinydir_readfile(&dir, &file);
    tinydir_next(&dir);
    char* ch = strstr(file.name, "step");
    if (ch == nullptr) {
      if (file.name[0] != '.')
        LOG(INFO) << "Irregular file in checkpoint folder: " << file.name;
      continue;
    }

    LOG(INFO) << "Add checkpoint file for resume: " << ch;
    int step = atoi(ch+4);
    if (step == latest_step) {
      ck_files.push_back(file.name);
    } else if(step > latest_step) {
      latest_step = step;
      ck_files.clear();
      ck_files.push_back(string(file.name));
    }
  }

  if (latest_step > 0) {
    jobConf->set_step(latest_step);
    if (!jobConf->has_reset_param_version())
      jobConf->set_reset_param_version(false);
    jobConf->clear_checkpoint_path();
    for (auto ck_file : ck_files)
      jobConf->add_checkpoint_path(folder + "/" + ck_file);
  }
  tinydir_close(&dir);
}

void Trainer::Start(bool resume, const SingaProto& singaConf, JobProto* job) {
  // register job to zookeeper at the beginning
  auto cluster = Cluster::Setup(job->id(), singaConf, job->cluster());
  if (resume)
    Resume(job);

  router_ = new Router();
  router_->Bind(kInprocRouterEndpoint);
  const string hostip = cluster->hostip();
  int port = router_->Bind("tcp://" + hostip + ":*");
  // register endpoint to zookeeper
  cluster->Register(getpid(), hostip + ":" + std::to_string(port));

  int nthreads = 1;
  const vector<Worker*> workers = CreateWorkers(nthreads, *job);
  nthreads += workers.size();
  const vector<Server*> servers = CreateServers(*job);
  SetupWorkerServer(*job, workers, servers);

#ifdef USE_MPI
  for (int i = 0; i < nthreads; i++)
    MPIQueues.push_back(make_shared<SafeQueue>());
#endif
  vector<std::thread> threads;
  for(auto server : servers)
    threads.push_back(std::thread(&Server::Run, server));
  for(auto worker : workers)
    threads.push_back(std::thread(&Worker::Run, worker));
  Run(workers, servers);
  for(auto& thread : threads)
    thread.join();
  for(auto server : servers)
    delete server;
  for(auto worker : workers)
    delete worker;
}

void Trainer::Run(
    const vector<Worker*>& workers,
    const vector<Server*>& servers) {
  int nworkers = workers.size(), nservers = servers.size();
  auto cluster = Cluster::Get();
  procs_id_ = cluster->procs_id();
  LOG(INFO) << "Stub in process " << procs_id_ << " starts";

  map<int, Dealer*> inter_dealers;  // for sending msg to other procs

  std::queue<Msg*> msg_queue;
  while (true) {
    Msg* msg = nullptr;
    if (msg_queue.empty()) {
      msg = router_->Receive();
    } else {
      msg = msg_queue.front();
      msg_queue.pop();
    }
    int type = msg->type(), dst = msg->dst(), flag = AddrType(dst);
    if (flag == kStub && (AddrProc(dst) == procs_id_ || AddrGrp(dst) == -1)) {
      //  the following statements are ordered!
      if (type == kConnect) {
        DeleteMsg(&msg);
      } else if (type == kMetric) {
        DisplayMetric(&msg);
      } else if (type == kStop) {
        int src_flag = AddrType(msg->src());
        if (src_flag == kServer) nservers--;
        else if (src_flag == kWorkerParam) nworkers--;
        DeleteMsg(&msg);
        if (nworkers == 0 && nservers == 0) break;
      } else {
        HandleLocalMsg(&msg_queue, &msg);
      }
    } else {
      int dst_procs = AddrProc(dst);
      if (flag != kStub)
        dst_procs = cluster->ProcsIDOf(AddrGrp(dst), AddrID(dst), flag);
      if (dst_procs != procs_id_) {
        if (inter_dealers.find(dst_procs) == inter_dealers.end())
          inter_dealers[dst_procs] = CreateInterProcsDealer(dst_procs);
        inter_dealers[dst_procs]->Send(&msg);
      } else {
        router_->Send(&msg);
      }
    }
  }
  LOG(ERROR) << "Stub in process " << procs_id_ << " stops";
  for (auto& entry : inter_dealers)
    delete entry.second;
}

void Trainer::DisplayMetric(Msg** msg) {
  Msg* msgg = *msg;
  // only display metrics from the first group
  if (AddrGrp(msgg->src()) == 0) {
    int step = msgg->trgt_version();
    char prefix[128];
    msgg->ParseFormatFrame("s", prefix);
    CHECK(msgg->NextFrame());
    const string perf(static_cast<char*>(msgg->FrameData()), msgg->FrameSize());
    Metric cur(perf);
    LOG(ERROR) << prefix << " step-" << step <<", " << cur.ToLogString();
  }
  DeleteMsg(msg);
}

Dealer* Trainer::CreateInterProcsDealer(int dst_procs) {
  // forward to other procs
  auto cluster = Cluster::Get();
  auto dealer = new Dealer();
  while(cluster->endpoint(dst_procs)=="") {
    //kCollectSleepTime));
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    LOG(ERROR)<<"waiting for procs "<< dst_procs<<" to register";
  }
  dealer->Connect("tcp://"+cluster->endpoint(dst_procs));
  return dealer;
}

void Trainer::HandleLocalMsg(queue<Msg*>* msg_queue, Msg** msg) {
  Msg* msgg = *msg;
  int paramid = ParamID(msgg->trgt_val());
  int type = msgg->type();
  int grp;
  ParamEntry *entry = nullptr;
  switch (type) {  // TODO process other requests, e.g. RESTful
    case kUpdate:
      grp = AddrGrp(msgg->src());
      entry = worker_shard_.at(Hash(grp, paramid));
      for(auto update_msg : HandleUpdate(entry, msg))
        msg_queue->push(update_msg);
      break;
    case kRUpdate:
      grp = AddrGrp(msgg->dst());
      entry = worker_shard_.at(Hash(grp, paramid));
      HandleUpdateResponse(entry, msg);
      break;
    case kGet:
      grp = AddrGrp(msgg->src());
      entry = worker_shard_.at(Hash(grp, paramid));
      for(auto get_msg : HandleGet(entry, msg))
        msg_queue->push(get_msg);
      break;
    case kRGet:
      grp = AddrGrp(msgg->dst());
      entry = worker_shard_.at(Hash(grp, paramid));
      HandleGetResponse(entry, msg);
      break;
    case kPut:
      grp = AddrGrp(msgg->src());
      entry = worker_shard_.at(Hash(grp, paramid));
      for(auto put_msg : HandlePut(entry, msg))
        msg_queue->push(put_msg);
      break;
    default:
      LOG(ERROR)<<"Unknow message type:"<<type;
      break;
  }
}

void Trainer::GenMsgs(int type, int version, ParamEntry* entry,
    Msg* msg, vector<Msg*> *ret) {
  int src_grp = AddrGrp(msg->src());
  int dst_grp = src_grp / Cluster::Get()->nworker_groups_per_server_group();
  auto param=entry->shares.at(0);
  for (int idx = 0 ; idx < param->num_slices(); idx++) {
    int slice_id =param->slice_start() + idx;
    int server = slice2server_[slice_id];
    int dst_procs = Cluster::Get()->ProcsIDOf(dst_grp, server, kServer);
    Msg* new_msg = nullptr;
    if (type == kPut) {
      CHECK_GT(entry->num_total, 0);
      new_msg = param->GenPutMsg(dst_procs != procs_id_, idx);
      new_msg->AddFormatFrame("i", entry->num_total);
    } else if (type == kGet) {
      new_msg = param->GenGetMsg(dst_procs != procs_id_, idx);
    } else if (type == kUpdate) {
      new_msg = param->GenUpdateMsg(dst_procs != procs_id_, idx);
      new_msg->AddFormatFrame("i", entry->num_local);
    } else {
      LOG(FATAL) << "Wrong type";
    }
    new_msg->set_trgt(ParamTrgt(param->owner(), slice_id), version);
    new_msg->set_src(Addr(src_grp, procs_id_, kStub));
    new_msg->set_dst(Addr(dst_grp, server, kServer));
    ret->push_back(new_msg);
  }
}

const vector<Msg*> Trainer::HandleGet(ParamEntry* entry, Msg** msg) {
  vector<Msg*> ret;
  int version = (*msg)->trgt_version();
  if (version > entry->next_version) {
    entry->next_version = version;
    GenMsgs(kGet, version, entry, *msg, &ret);
  }
  DeleteMsg(msg);
  return ret;
}

const vector<Msg*> Trainer::HandleUpdate(ParamEntry *entry, Msg** msg) {
  vector<Msg*> ret;
  entry->num_update++;
  if (entry->num_update >= entry->num_local) {
    // average local gradient
    if (entry->num_local > 1) {
      auto it = entry->shares.begin();
      auto shape=mshadow::Shape1((*it)->size());
      mshadow::Tensor<mshadow::cpu,1> sum((*it)->mutable_cpu_grad(), shape);
      for (++it; it != entry->shares.end(); it++) {
        mshadow::Tensor<mshadow::cpu,1> grad((*it)->mutable_cpu_grad(), shape);
        sum += grad;
      }
    }
    int step = (*msg)->trgt_version();
    GenMsgs(kUpdate, step, entry, *msg, &ret);
    entry->num_update = 0;
  }
  DeleteMsg(msg);
  return ret;
}

const vector<Msg*> Trainer::HandlePut(ParamEntry* entry, Msg** msg) {
  vector<Msg*> ret;
  int version = (*msg)->trgt_version();
  GenMsgs(kPut, version, entry, *msg, &ret);
  DeleteMsg(msg);
  return ret;
}

void Trainer::HandleGetResponse(ParamEntry* entry, Msg** msg) {
  int version = (*msg)->trgt_version();
  int sliceid = SliceID((*msg)->trgt_val());
  auto param = entry->shares.at(0);
  if (param->ParseGetResponseMsg(*msg, sliceid-param->slice_start()))
    param->set_version(version);
  DeleteMsg(msg);
}

void Trainer::HandleUpdateResponse(ParamEntry* entry, Msg** msg) {
  int version = (*msg)->trgt_version();
  int sliceid = SliceID((*msg)->trgt_val());
  auto param = entry->shares.at(0);
  if (param->ParseUpdateResponseMsg(*msg, sliceid-param->slice_start()))
    param->set_version(version);
  DeleteMsg(msg);
}
} /* singa */
