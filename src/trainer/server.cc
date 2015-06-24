#include <list>
#include <tuple>
#include <queue>
#include "trainer/server.h"
#include "utils/param.h"
#include "utils/singleton.h"
#include "utils/factory.h"
#include "utils/cluster.h"


namespace singa {
Server::Server(int thread_id,int group_id, int server_id):
  thread_id_(thread_id),group_id_(group_id), server_id_(server_id){}

void Server::Setup(const UpdaterProto& proto,
    shared_ptr<ServerShard> shard, const vector<int>& slice2group){
	//VLOG(3) << "Parsing config file for host "<<hosts[id_] << " server id = " <<id_;
  updater_=shared_ptr<Updater>(Singleton<Factory<Updater>>::Instance()
      ->Create("Updater"));
  updater_->Init(proto);
  shard_=shard;
  slice2group_=slice2group;
}

void Server::Run(){
  dealer_=std::make_shared<Dealer>(2*thread_id_);
  dealer_->Connect(kInprocRouterEndpoint);

  Msg* ping=new Msg();
  ping->set_src(group_id_, server_id_, kServer);
  ping->set_dst(-1,-1,kStub);
  ping->add_frame("PING", 4);
  ping->set_type(kConnect);
  dealer_->Send(&ping);
	//start recv loop and process requests
  while (true){
    Msg* msg=dealer_->Receive();
    if (msg==nullptr)
      break;
    Msg* response=nullptr;
    int type=msg->type();
    if (type== kStop){
      msg->set_src(group_id_, server_id_, kServer);
      msg->set_dst(-1,-1, kStub);
      dealer_->Send(&msg);
      break;
    }else if (type==kConnect){
      // TODO remove receiving pong msg
      string pong((char*)msg->frame_data(), msg->frame_size());
      CHECK_STREQ("PONG", pong.c_str());
      delete msg;
    }else if(type==kPut){
      int pid=msg->trgt_second();
      shared_ptr<Param> param=nullptr;
      if(shard_->find(pid)!=shard_->end()){
        LOG(ERROR)<<"Param ("<<pid<<") is put more than once";
        param=shard_->at(pid);
      }else{
        param=shared_ptr<Param>(Singleton<Factory<Param>>::Instance()
            ->Create("Param"));
        param->set_id(pid);
        (*shard_)[pid]=param;
      }
      HandlePut(param, &msg);
    }else{
      int pid=msg->trgt_second();
      if(shard_->find(pid)==shard_->end()){
        // delay the processing by re-queue the msg.
        response=msg;
        DLOG(ERROR)<<"Requeue msg";
      } else{
        auto param=shard_->at(pid);
        switch (type){
          case kGet:
            response=HandleGet(param, &msg);
            break;
          case kUpdate:
            response = HandleUpdate(param, &msg);
            break;
          case kSyncRequest:
            response = HandleSyncRequest(param, &msg);
            break;
        }
        if (response!=nullptr){
          dealer_->Send(&response);
        }
      }
    }
  }
}

void Server::HandlePut(shared_ptr<Param> param, Msg **msg){
  int version=(*msg)->trgt_third();
  param->HandlePutMsg(msg);
  // must set version after HandlePutMsg which allocates the memory
  param->set_version(version);
}

Msg* Server::HandleGet(shared_ptr<Param> param, Msg **msg){
  if(param->version()<(*msg)->trgt_third())
    return *msg;
  else{
    auto reply= param->HandleGetMsg(msg);
    int paramid=reply->trgt_first(), slice=reply->trgt_second();
    reply->set_trgt(paramid, slice, param->version());
  }
}

Msg* Server::HandleUpdate(shared_ptr<Param> param, Msg **msg) {
  auto* tmp=static_cast<Msg*>((*msg)->CopyAddr());
  tmp->SwapAddr();
  int paramid=(*msg)->trgt_first();
  int sliceid=(*msg)->trgt_second();
  int step=(*msg)->trgt_third();
  bool copy=param->ParseUpdateMsg(msg);
  updater_->Update(step, param);
  param->set_version(param->version()+1);
  auto response=param->GenUpdateResponseMsg(copy);
  response->set_trgt(paramid, sliceid, param->version());
  response->SetAddr(tmp);
  delete tmp;
  return response;
}

Msg* Server::HandleSyncRequest(shared_ptr<Param> param, Msg **msg){
  return param->HandleSyncMsg(msg);
}

} /* singa */
