#include <thread>
#include <chrono>
#include "mshadow/tensor.h"
#include "trainer/server.h"
#include "utils/param.h"
#include "utils/singleton.h"
#include "utils/factory.h"
#include "utils/cluster.h"
#include "proto/common.pb.h"

namespace singa {

using namespace mshadow;
using std::vector;

Server::Server(int thread_id,int group_id, int server_id):
  thread_id_(thread_id),grp_id_(group_id), id_(server_id){
}

void Server::Setup(const UpdaterProto& proto,
    std::unordered_map<int, ParamEntry*>* shard,
    const vector<int>& slice2group) {
  updater_ = Updater::Create(proto);
  shard_ = shard;
  slice2group_ = slice2group;
}

Server::~Server() {
  delete updater_;
}

void Stop(void * running) {
  *static_cast<bool *>(running) = false;
}

void Server::Run() {
  LOG(ERROR) << "Server (group = " << grp_id_ <<", id = " << id_ << ") start";
  auto dealer = new Dealer(2*thread_id_);
  CHECK(dealer->Connect(kInprocRouterEndpoint));
  Msg* ping = new Msg(Addr(grp_id_, id_, kServer), Addr(-1, -1, kStub));
  ping->set_type(kConnect);
  dealer->Send(&ping);

  auto cluster = Cluster::Get();
  bool running = true;
  CHECK(cluster->runtime()->WatchSGroup(grp_id_, id_, Stop, &running));

  int nserver_grps = cluster->nserver_groups();
  vector<Param*> master_params;
  size_t syncEntry=0;
  Poller poll(dealer);
  // start recv loop and process requests
  while (running) {
    auto *sock = poll.Wait(cluster->poll_time());
    if (poll.Terminated()) {
      LOG(ERROR) << "Connection broken!";
      exit(0);
    } else if (sock == nullptr) {
      continue;
    }
    Msg* msg=dealer->Receive();
    if (msg==nullptr) break;
    Msg* response=nullptr;
    int type=msg->type();
    int slice_id = SliceID(msg->trgt_val());
    if (type == kPut) {
      response = HandlePut(&msg);
      if(slice2group_[slice_id] == grp_id_)
        master_params.push_back(shard_->at(slice_id)->shares.at(0));
    } else {
      if (shard_->find(slice_id) == shard_->end()) {
        // delay the processing by re-queue the msg.
        response = msg;
      } else if (type == kSyncReminder) {
        DeleteMsg(&msg);
        if(syncEntry >= master_params.size())
          continue;
        auto param = master_params.at(syncEntry);
        // control the frequency of synchronization
        // currently sync is triggerred only when the slice is updated
        // by local worker or other workers for at least nserver_groups times.
        // TODO may optimize the trigger condition.
        if (abs(param->local_version() - param->version()) >= nserver_grps) {
          for (auto msg : GenSyncMsgs(param))
            dealer->Send(&msg);
          syncEntry = (syncEntry+1) % master_params.size();
        }
      } else {
        switch (type) {
          case kGet:
            response = HandleGet(&msg);
            break;
          case kUpdate:
            for (auto reply : HandleUpdate(&msg))
              dealer->Send(&reply);
            break;
          case kSyncRequest:
            response = HandleSyncRequest(&msg);
            break;
          default:
            LOG(ERROR)<<"Unknown message type "<<type;
            break;
        }
      }
    }
    if (response != nullptr)
      dealer->Send(&response);
  }

  // send stop msg to stub
  Msg* msg = new Msg(Addr(grp_id_, id_, kServer), Addr(-1, -1, kStub));
  msg->set_type(kStop);
  dealer->Send(&msg);
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));

  LOG(ERROR) << "Server (group = " << grp_id_ << ", id = " << id_ << ") stops";
  delete dealer;
}

const vector<Msg*> Server::GenSyncMsgs(Param* param) {
  vector<Msg*> ret;
  // TODO replace the argument (0,0) to sync a chunk instead of a slice
  auto msg = param->GenSyncMsg(0, 0);
  auto cluster = Cluster::Get();
  for (int i = 0; i < cluster->nserver_groups(); i++) {
    if (i != grp_id_) {
      Msg* tmp = msg;
      if (i < cluster->nserver_groups() - 1)
        tmp = new Msg(*msg);
      // assume only one server per group, TODO generalize it
      tmp->set_dst(Addr(i, 0, kServer));
      tmp->set_src(Addr(grp_id_, id_, kServer));
      ret.push_back(tmp);
      param->set_version(param->local_version());
      //LOG(ERROR)<<"sync slice="<<param->id()<<" to procs "<<i;
    }
  }
  return ret;
}

Msg* Server::HandlePut(Msg **msg) {
  int version = (*msg)->trgt_version();
  int slice_id = SliceID((*msg)->trgt_val());
  if (shard_->find(slice_id) != shard_->end())
    LOG(FATAL) << "Param (" << slice_id << ") is put more than once";

  // TODO(wangwei) replace hard coded param type 0
  auto  param = Singleton<Factory<Param>>::Instance()->Create(0);
  auto response = param->HandlePutMsg(msg, true);
  // parse num of shares of this param from a worker group
  int num_shares = 1;
  if ((*msg)->NextFrame())
    (*msg)->ParseFormatFrame("i", &num_shares);
  DeleteMsg(msg);
  (*shard_)[slice_id] = new ParamEntry(num_shares, param);
  // must set version after HandlePutMsg which allocates the memory
  param->set_version(version);
  param->set_local_version(version);
  param->set_id(slice_id);
  //LOG(ERROR)<<"put norm "<<param->data().asum_data()<<", "<<pid;
  // allocate blob for param sync between groups.
  if (Cluster::Get()->nserver_groups() > 1 && slice2group_[slice_id] != grp_id_) {
    last_data_[slice_id] = std::make_shared<Blob<float>>();
    last_data_[slice_id]->ReshapeLike(param->data());
    last_data_[slice_id]->CopyFrom(param->data());
  }
  LOG(INFO)<<"server (group = " << grp_id_ << ", id = " << id_ <<") put slice="
    << slice_id << " size=" << param->size();
  return response;
}

Msg* Server::HandleGet(Msg **msg) {
  int val = (*msg)->trgt_val();
  auto param = shard_->at(SliceID(val))->shares.at(0);
  // re-queue the request if the param is not updated to the required version
  if(param->version()<(*msg)->trgt_version())
    return *msg;
  else {
    // LOG(ERROR) << "get " << slice << " from "<<(*msg)->src_first();
    auto reply = param->HandleGetMsg(msg, false);
    reply->set_trgt(val, param->version());
    return reply;
  }
}

const vector<Msg*> Server::HandleUpdate(Msg **msg) {
  vector<Msg*> ret;
  int sliceid = SliceID((*msg)->trgt_val());
  auto entry = shard_->at(sliceid);
  buffer_requests_[sliceid].push_back(*msg);
  int num_update;
  (*msg)->LastFrame();
  (*msg)->ParseFormatFrame("i", &num_update);
  (*msg)->FirstFrame();
  entry->num_update += num_update;
  // LOG(ERROR) << "update "<<sliceid<< " from "<<(*msg)->src_second()
  //  << ", " << num_update << " total " << entry->num_total;
  // do update until recv gradients from all shares of this param/slice
  if (entry->num_update >= entry->num_total) {
    CHECK_EQ(entry->num_update, entry->num_total);
    auto& request = buffer_requests_.at(sliceid);
    int step = (*msg)->trgt_version();
    auto param = entry->shares.at(0);
    // extract and aggregate gradients
    param->ParseUpdateMsgs(request);
    updater_->Update(step, param, 1.0f / entry->num_total);
    param->set_local_version(param->local_version() + 1);
    // response to all shares of this param
    for (auto response : param->GenUpdateResponseMsgs(&request, false)) {
      response->set_trgt((*msg)->trgt_val(), param->local_version());
      ret.push_back(response);
    }
    entry->num_update = 0;
  }
  *msg = nullptr;
  return ret;
}

Msg* Server::HandleSyncRequest(Msg **msg) {
  Msg* msgg = *msg;
  int slice = SliceID(msgg->trgt_val());
  auto param = shard_->at(slice)->shares.at(0);
  Msg* response=nullptr;
  auto shape=Shape1(param->size());
  CHECK_EQ(msgg->FrameSize(), param->size()*sizeof(float));
  Tensor<cpu, 1> tmp(static_cast<float*>(msgg->FrameData()), shape);
  Tensor<cpu, 1> cur(param->mutable_cpu_data(), shape);
  //LOG(ERROR)<<"Recv sync for "<<param->id();
  if (slice2group_[slice] == grp_id_) {
    // recv sync msg on slice I am mastering
    cur+=tmp;
    param->set_local_version(param->local_version()+1);
  } else {  // recv sync msg on slice mastered by others
    TensorContainer<cpu, 1> diff(shape);
    Tensor<cpu, 1> prev(last_data_[param->id()]->mutable_cpu_data(), shape);
    diff=cur-prev;
    msgg->NextFrame();
    int bandwidth;
    msgg->ParseFormatFrame("i", &bandwidth);
    if (bandwidth > 0) {
      // send back my updates to the server group mastering this param
      response=new Msg(msgg->dst(), msgg->src());
      response->set_type(kSyncRequest);
      response->set_trgt(param->id(), param->version());
      response->AddFrame(diff.dptr, param->size()*sizeof(float));
      prev=diff+tmp;
      Copy(cur, prev);
    } else {  // no bandwidth, aggregate my updates for next sync
      Copy(prev, tmp);
      cur=tmp+diff;
    }
  }
  DeleteMsg(msg);
  return response;
}
} /* singa */
