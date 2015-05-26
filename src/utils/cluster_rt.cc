#include <zookeeper/zookeeper_log.h>
#include "utils/cluster_rt.h"

using std::to_string;

namespace singa {

/********* Implementation for ZKClusterRT **************/

ZKClusterRT::ZKClusterRT(string host, int timeout){
  host_ = host;
  timeout_ = timeout;
  zkhandle_ = nullptr;
}

ZKClusterRT::~ZKClusterRT(){
  //close zookeeper handler
  zookeeper_close(zkhandle_);
  //release callback vector
  for (RTCallback *p : cb_vec_){
    delete p;
  }
}

bool ZKClusterRT::Init(){

  zoo_set_debug_level(ZOO_LOG_LEVEL_WARN);

  zkhandle_ = zookeeper_init(host_.c_str(), watcherGlobal, timeout_, 0, "ZKClusterRT", 0);

  if (zkhandle_ == nullptr){
    LOG(ERROR) << "Error when connecting to zookeeper servers...";
    LOG(ERROR) <<"Please ensure zookeeper service is up in host(s):";
    LOG(ERROR) << host_.c_str();
    return false;
  }

  return true;
}

bool ZKClusterRT::sWatchSGroup(int gid, int sid, rt_callback fn, void *ctx){
 
  string path = getSGroupPath(gid);
  struct Stat stat;

  //check existance of zk node
  int ret = zoo_exists(zkhandle_, path.c_str(), 0, &stat);
  //if have, pass
  if (ret == ZOK) ;
  //need to create zk node first
  else if (ret == ZNONODE){
    char buf[MAX_BUF_LEN]; 
    ret = zoo_create(zkhandle_, path.c_str(), NULL, -1, &ZOO_OPEN_ACL_UNSAFE, 0, buf, MAX_BUF_LEN);
    if (ret == ZOK){
      LOG(INFO) << "zookeeper node " << buf << " created";
    }
    else{
      LOG(ERROR) << "Unhandled ZK error code: " << ret << " (zoo_create)";
      return false;
    }
  }
  else{
    LOG(ERROR) << "Unhandled ZK error code: " << ret << " (zoo_exists)";
    return false;
  }

  struct String_vector child;
  //store the callback function and context for later usage
  RTCallback *cb = new RTCallback;
  cb->fn = fn;
  cb->ctx = ctx;
  cb_vec_.push_back(cb);
  //start to watch on the zk node, does not care about the first return value
  zoo_wget_children(zkhandle_, path.c_str(), childChanges, cb, &child);

  return true;
}

bool ZKClusterRT::wJoinSGroup(int gid, int wid, int s_group){
  
  string path = getSGroupPath(s_group) + getWorkerPath(gid, wid);
  char buf[MAX_BUF_LEN]; 
  
  int ret = zoo_create(zkhandle_, path.c_str(), NULL, -1, &ZOO_OPEN_ACL_UNSAFE, ZOO_EPHEMERAL, buf, MAX_BUF_LEN);
  if (ret == ZOK){
    LOG(INFO) << "zookeeper node " << buf << " created";
    return true;
  }
  else if (ret == ZNODEEXISTS){
    LOG(WARNING) << "zookeeper node " << path << " already exist";
    return true;
  }
  else if (ret == ZNONODE){
    LOG(ERROR) << "zookeeper parent node " << getSGroupPath(s_group) << " not exist";
    return false;
  }
  
  LOG(ERROR) << "Unhandled ZK error code: " << ret << " (zoo_create)";
  return false;
}

bool ZKClusterRT::wLeaveSGroup(int gid, int wid, int s_group){
  
  string path = getSGroupPath(s_group) + getWorkerPath(gid, wid);
  
  int ret = zoo_delete(zkhandle_, path.c_str(), -1);
  if (ret == ZOK){
    LOG(INFO) << "zookeeper node " << path << " deleted";
    return true;
  }
  else if (ret == ZNONODE){
    LOG(WARNING) << "try to delete an non-existing zookeeper node " << path;
    return true;
  }

  LOG(ERROR) << "Unhandled ZK error code: " << ret << " (zoo_delete)";
  return false;
}

void ZKClusterRT::watcherGlobal(zhandle_t * zh, int type, int state, const char *path, void *watcherCtx){
  if (type == ZOO_SESSION_EVENT){
    if (state == ZOO_CONNECTED_STATE)
      LOG(INFO) << "Connected to zookeeper service successfully!";
    else if (state == ZOO_EXPIRED_SESSION_STATE)
      LOG(INFO) << "zookeeper session expired!";
  }
}

void ZKClusterRT::childChanges(zhandle_t *zh, int type, int state, const char *path, void *watcherCtx){

  //check if already callback
  RTCallback *cb = (RTCallback *)watcherCtx;
  if (cb->fn == nullptr) return;

  struct String_vector child;
  //check the child list and put another watcher
  int ret = zoo_wget_children(zh, path, childChanges, watcherCtx, &child);
  LOG(INFO) << "ret = " << ret;
  if (ret == ZOK){
    LOG(INFO) << "child.count = " << child.count;
    if (child.count == 0){
      //all workers leave, we do callback now
      (*cb->fn)(cb->ctx);
      cb->fn = nullptr;
    }
  }
}

string ZKClusterRT::getSGroupPath(int gid){
  //return "/singa/status";
  return "/singa/status/sg"+to_string(gid);
}

string ZKClusterRT::getWorkerPath(int gid, int wid){
  return "/g"+to_string(gid)+"_w"+to_string(wid);
}

} // namespace singa
