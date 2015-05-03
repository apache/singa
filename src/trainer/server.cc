#include <list>
#include <tuple>
#include <queue>
#include "trainer/server.h"
#include "utils/param.h"
#include "utils/singleton.h"
#include "utils/factory.h"
#include "utils/cluster.h"


namespace singa {
Server::Server(int group_id, int server_id):
  group_id_(group_id), server_id_(server_id){}

void Server::Setup(const UpdaterProto& proto,
    shared_ptr<PMServer::ParamShard> shard,
    shared_ptr<Dealer> dealer){
	//VLOG(3) << "Parsing config file for host "<<hosts[id_] << " server id = " <<id_;
  pmserver_=shared_ptr<PMServer>(Singleton<Factory<PMServer>>::Instance()
      ->Create("PMServer"));
  pmserver_->Setup(group_id_, server_id_, shard, proto);
  dealer_=dealer;
}

void Server::Run(){
  Msg* ping=new Msg();
  ping->set_src(group_id_, server_id_, kServer);
  ping->set_dst(0,0,kStub);
  ping->set_type(kConnect);
  dealer_->Send(ping);
  int timeout=Cluster::Get()->server_timeout();
  Poller poller;
  poller.Add(dealer_.get());
	//start recv loop and process requests
  while (true){
    Msg* msg=dealer_->Receive();
    if (msg==nullptr)
      break;
    Msg* response=nullptr;
    int type=msg->type();
    switch (type){
      case kPut:
        response = pmserver_->HandlePut(&msg);
        break;
      case kGet:
        response = pmserver_->HandleGet(&msg);
        break;
      case kUpdate:
        response = pmserver_->HandleUpdate(&msg);
        break;
      case kSyncRequest:
        VLOG(3)<<"Handle SYNC-REQUEST";
        response = pmserver_->HandleSyncRequest(&msg);
        break;
      case kSyncResponse:
        VLOG(3) << "Handle SYNC response";
        pmserver_->HandleSyncResponse(&msg);
        break;
    }

    if (response!=nullptr)
      dealer_->Send(response);
  }
}



} /* singa */
