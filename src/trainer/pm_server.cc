#include <gflags/gflags.h>
#include <glog/logging.h>
#include "trainer/pm_server.h"
#include "utils/singleton.h"
#include "utils/factory.h"
#include <vector>

using std::vector;

namespace singa{
void PMServer::Setup(int group_id, int server_id, shared_ptr<ParamShard> shard,
      const UpdaterProto& proto){
  group_id_=group_id;
  server_id_=server_id;
  shard_=shard;
  updater_=shared_ptr<Updater>(Singleton<Factory<Updater>>::Instance()
      ->Create("Updater"));
  updater_->Init(proto);
}

PMServer::~PMServer(){
}

bool PMServer::SyncNow(){
  return false;
}
Msg* PMServer::HandlePut(Msg **msg){
  int id=(*msg)->target();
  shared_ptr<Param> param=nullptr;
  if(shard_->find(id)!=shard_->end()){
    LOG(ERROR)<<"Param ("<<id<<") is put more than once";
    param=shard_->at(id);
  }else{
    param=shared_ptr<Param>(Singleton<Factory<Param>>::Instance()
        ->Create("Param"));
    param->set_id(id);
    (*shard_)[id]=param;
  }
  return param->HandlePutMsg(msg);
}

Msg* PMServer::HandleGet(Msg **msg){
  int id=(*msg)->target();
  shared_ptr<Param> param=nullptr;
  if(shard_->find(id)!=shard_->end()){
    param=shard_->at(id);
    return param->HandleGetMsg(msg);
	} else {
		//re-construct msg to be re-queued.
		//the calling function will send this message off
    return *msg;
	}
}

Msg* PMServer::HandleUpdate(Msg **msg) {
  int id=(*msg)->target();
  shared_ptr<Param> param=nullptr;
  if(shard_->find(id)!=shard_->end()){
		//repsonse of the format: <identity><type: kData><paramId><param content>
    param=shard_->at(id);
    Msg* tmp=static_cast<Msg*>((*msg)->CopyAddr());
    param->ParseUpdateMsg(msg);
    updater_->Update(param->version(), param);
    param->set_version(param->version()+1);
    auto response=param->GenUpdateResponseMsg();
    tmp->SwapAddr();
    response->SetAddr(tmp);
    delete tmp;
    return response;
	} else {
    LOG(ERROR)<<"Param ("<<id<<") is not maintained by server ("<<group_id_
      <<", "<<server_id_<<")";
		//re-construct msg to be re-queued.
		return *msg;
	}
}

Msg* PMServer::HandleSyncRequest(Msg **msg){
  int id=(*msg)->target();
  shared_ptr<Param> param=nullptr;
  if(shard_->find(id)!=shard_->end()){
		//repsonse of the format: <identity><type: kData><paramId><param content>
    param=shard_->at(id);
    return param->HandleSyncMsg(msg);
	} else {
		//re-construct msg to be re-queued.
    return *msg;
	}
}

int PMServer::HandleSyncResponse(Msg **msg){
  int id=(*msg)->target();
  CHECK(shard_->find(id)!=shard_->end());
  return shard_->at(id)->ParseSyncResponseMsg(msg);
}

} // namespace singa


