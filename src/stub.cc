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

#include "singa/stub.h"

#include <glog/logging.h>
#include <unistd.h>
#include <map>
#include <thread>
#include <set>
#include "singa/proto/common.pb.h"
#include "singa/utils/cluster.h"
#include "singa/utils/common.h"
#include "singa/utils/tinydir.h"
#include "singa/utils/math_blob.h"

namespace singa {

using std::vector;
using std::string;

/***********************Stub****************************/
Stub::~Stub() {
  delete router_;
}
void Stub::Setup() {
  router_ = new Router();
  router_->Bind(kInprocRouterEndpoint);
  auto cluster = Cluster::Get();
  const string hostip = cluster->hostip();
  int port = router_->Bind("tcp://" + hostip + ":*");
  endpoint_ = hostip + ":" + std::to_string(port);
}
/**
 * Get a hash id for a Param object from a group.
 *
 * Simple multiple group_id with a large prime number 997 (assuming there are
 * no more than 997 worker groups) and plus owner param id.
 */
inline int Hash(int grp_id, int param_id) {
  return grp_id * 997 + param_id;
}
const std::unordered_map<int, ParamEntry*>  CreateParamShard(
    const vector<Worker*>& workers) {
  std::unordered_map<int, ParamEntry*> shard;
  // grp id -> net
  std::unordered_map<int, NeuralNet*> grp2net;
  // grp id -> worker id range
  std::unordered_map<int, std::pair<int, int>> grp2workers;
  for (auto worker : workers) {
    int grp = worker->grp_id(), id = worker->id();
    if (grp2net.find(grp) == grp2net.end()) {
      grp2net[grp] = worker->train_net();
      grp2workers[grp] = std::make_pair(id, id + 1);
    } else {
      CHECK_EQ(grp2net[grp], worker->train_net());
      int start = grp2workers[grp].first, end = grp2workers[grp].second;
      if (start > id) start = id;
      if (end < id + 1) end = id + 1;
      grp2workers[grp] = std::make_pair(start, end);
    }
  }

  for (const auto entry : grp2net) {
    int grp = entry.first;
    int wstart = grp2workers[grp].first, wend = grp2workers[grp].second;
    for (auto layer : entry.second->layers()) {
      if (layer->unroll_index() > 0)
        continue;
      int partition = layer->partition_id();
      bool local =  partition >= wstart && partition < wend;
      for (auto param : layer->GetParams()) {
        int hash = Hash(grp, param->owner());
        if (shard.find(hash) == shard.end())
          shard[hash] = new ParamEntry();
        shard[hash]->AddParam(local, param);
      }
    }
  }
  return shard;
}

void Stub::Run(const vector<int>& slice2server,
    const vector<Worker*>& workers, const vector<Server*>& servers) {
  slice2server_ = slice2server;
  int nworkers = workers.size(), nservers = servers.size();
  auto cluster = Cluster::Get();
  int procs_id = cluster->procs_id();
  LOG(INFO) << "Stub in process " << procs_id << " starts";
  auto shard = CreateParamShard(workers);
  std::map<int, Dealer*> inter_dealers;  // for sending msg to other procs
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
    if (flag == kStub && (AddrProc(dst) == procs_id || AddrGrp(dst) == -1)) {
      //  the following statements are ordered!
      if (type == kConnect) {
        DeleteMsg(&msg);
      } else if (type == kStop) {
        int src_flag = AddrType(msg->src());
        if (src_flag == kServer) nservers--;
        else if (src_flag == kWorkerParam) nworkers--;
        DeleteMsg(&msg);
        if (nworkers == 0 && nservers == 0) break;
      } else {
        int grp;
        int paramid = ParamID(msg->trgt_val());
        ParamEntry *entry = nullptr;
        switch (type) {
          case kUpdate:
            grp = AddrGrp(msg->src());
            entry = shard.at(Hash(grp, paramid));
            for (auto update_msg : HandleUpdateRequest(entry, &msg))
              msg_queue.push(update_msg);
            break;
          case kRUpdate:
            grp = AddrGrp(msg->dst());
            entry = shard.at(Hash(grp, paramid));
            HandleUpdateResponse(entry, &msg);
            break;
          case kGet:
            grp = AddrGrp(msg->src());
            entry = shard.at(Hash(grp, paramid));
            for (auto get_msg : HandleGetRequest(entry, &msg))
              msg_queue.push(get_msg);
            break;
          case kRGet:
            grp = AddrGrp(msg->dst());
            entry = shard.at(Hash(grp, paramid));
            HandleGetResponse(entry, &msg);
            break;
          case kPut:
            grp = AddrGrp(msg->src());
            entry = shard.at(Hash(grp, paramid));
            for (auto put_msg : HandlePutRequest(entry, &msg))
              msg_queue.push(put_msg);
            break;
          default:
            LOG(ERROR) << "Unknow message type:" << type;
            break;
        }
      }
    } else {
      int dst_procs = AddrProc(dst);
      if (flag != kStub)
        dst_procs = cluster->ProcsIDOf(AddrGrp(dst), AddrID(dst), flag);
      if (dst_procs != procs_id) {
        if (inter_dealers.find(dst_procs) == inter_dealers.end())
          inter_dealers[dst_procs] = CreateInterProcsDealer(dst_procs);
        inter_dealers[dst_procs]->Send(&msg);
      } else {
        router_->Send(&msg);
      }
    }
  }
  LOG(ERROR) << "Stub in process " << procs_id << " stops";
  for (auto& entry : inter_dealers)
    delete entry.second;
}

Dealer* Stub::CreateInterProcsDealer(int dst_procs) {
  // forward to other procs
  auto cluster = Cluster::Get();
  auto dealer = new Dealer();
  while (cluster->endpoint(dst_procs) == "") {
    // kCollectSleepTime));
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    LOG(ERROR) << "waiting for procs " << dst_procs << " to register";
  }
  dealer->Connect("tcp://"+cluster->endpoint(dst_procs));
  return dealer;
}

void Stub::GenMsgs(int type, int version, ParamEntry* entry, Msg* msg,
                      vector<Msg*> *ret) {
  int procs_id = Cluster::Get()->procs_id();
  int src_grp = AddrGrp(msg->src());
  int dst_grp = src_grp / Cluster::Get()->nworker_groups_per_server_group();
  auto param = entry->shares.at(0);
  for (int idx = 0 ; idx < param->num_slices(); idx++) {
    int slice_id = param->slice_start() + idx;
    int server = slice2server_[slice_id];
    int dst_procs = Cluster::Get()->ProcsIDOf(dst_grp, server, kServer);
    Msg* new_msg = nullptr;
    if (type == kPut) {
      CHECK_GT(entry->num_total, 0);
      new_msg = param->GenPutMsg(dst_procs != procs_id, idx);
      new_msg->AddFormatFrame("i", entry->num_total);
    } else if (type == kGet) {
      new_msg = param->GenGetMsg(dst_procs != procs_id, idx);
    } else if (type == kUpdate) {
      new_msg = param->GenUpdateMsg(dst_procs != procs_id, idx);
      new_msg->AddFormatFrame("i", entry->num_local);
    } else {
      LOG(FATAL) << "Wrong type";
    }
    new_msg->set_trgt(ParamTrgt(param->owner(), slice_id), version);
    new_msg->set_src(Addr(src_grp, procs_id, kStub));
    new_msg->set_dst(Addr(dst_grp, server, kServer));
    ret->push_back(new_msg);
  }
}

const vector<Msg*> Stub::HandleGetRequest(ParamEntry* entry, Msg** msg) {
  vector<Msg*> ret;
  int version = (*msg)->trgt_version();
  if (version > entry->next_version) {
    entry->next_version = version;
    GenMsgs(kGet, version, entry, *msg, &ret);
  }
  DeleteMsg(msg);
  return ret;
}

const vector<Msg*> Stub::HandleUpdateRequest(ParamEntry *entry, Msg** msg) {
  vector<Msg*> ret;
  entry->num_update++;
  if (entry->num_update >= entry->num_local) {
    // average local gradient
    if (entry->num_local > 1) {
      auto it = entry->shares.begin();
      auto sum = it;
      for (++it; it != entry->shares.end(); it++) {
        AXPY(1.0f, (*it)->grad(), (*sum)->mutable_grad());
      }
    }
    int step = (*msg)->trgt_version();
    GenMsgs(kUpdate, step, entry, *msg, &ret);
    entry->num_update = 0;
  }
  DeleteMsg(msg);
  return ret;
}

const vector<Msg*> Stub::HandlePutRequest(ParamEntry* entry, Msg** msg) {
  vector<Msg*> ret;
  int version = (*msg)->trgt_version();
  GenMsgs(kPut, version, entry, *msg, &ret);
  DeleteMsg(msg);
  return ret;
}

void Stub::HandleGetResponse(ParamEntry* entry, Msg** msg) {
  int version = (*msg)->trgt_version();
  int sliceid = SliceID((*msg)->trgt_val());
  auto param = entry->shares.at(0);
  if (param->ParseGetResponseMsg(*msg, sliceid-param->slice_start()))
    for (auto *p : entry->shares)
      p->set_version(version);
  DeleteMsg(msg);
}

void Stub::HandleUpdateResponse(ParamEntry* entry, Msg** msg) {
  int version = (*msg)->trgt_version();
  int sliceid = SliceID((*msg)->trgt_val());
  auto param = entry->shares.at(0);
  if (param->ParseUpdateResponseMsg(*msg, sliceid-param->slice_start()))
    for (auto *p : entry->shares)
      p->set_version(version);
  DeleteMsg(msg);
}
}  // namespace singa
