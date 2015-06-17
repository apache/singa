#include "utils/cluster_rt.h"
#include <algorithm>

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

  zkhandle_ = zookeeper_init(host_.c_str(), WatcherGlobal, timeout_, 0, "ZKClusterRT", 0);

  if (zkhandle_ == NULL){
    LOG(ERROR) << "Error when connecting to zookeeper servers...";
    LOG(ERROR) << "Please ensure zookeeper service is up in host(s):";
    LOG(ERROR) << host_.c_str();
    return false;
  }

  //create ZPATH_SINGA
  if (!CreateZKNode(ZPATH_SINGA.c_str(), nullptr, 0, nullptr)) return false;
  //create ZPATH_STATUS
  if (!CreateZKNode(ZPATH_STATUS.c_str(), nullptr, 0, nullptr)) return false;
  //create ZPATH_REGIST
  if (!CreateZKNode(ZPATH_REGIST.c_str(), nullptr, 0, nullptr)) return false;
  //create ZPATH_REGIST_PROC
  if (!CreateZKNode(ZPATH_REGIST_PROC.c_str(), nullptr, 0, nullptr)) return false;
  //create ZPATH_REGIST_LOCK
  if (!CreateZKNode(ZPATH_REGIST_LOCK.c_str(), nullptr, 0, nullptr)) return false;

  return true;
}

int ZKClusterRT::RegistProc(const string& host_addr){
 
  char buf[MAX_BUF_LEN];
  string lock = ZPATH_REGIST_LOCK+"/lock-";

  if (!CreateZKNode(lock.c_str(), nullptr, ZOO_EPHEMERAL | ZOO_SEQUENCE, buf)){
    return -1;
  }

  //get all children in lock folder
  vector<string> vt;
  if (!GetZKChild(ZPATH_REGIST_LOCK.c_str(), vt)){
    return -1;
  }

  //find own position among all locks
  int id = -1;
  std::sort(vt.begin(), vt.end());
  for (int i = 0; i < (int)vt.size(); ++i){
    if (ZPATH_REGIST_LOCK+"/"+vt[i] == buf){
      id = i;
      break;
    }
  }

  if (id == -1){
    LOG(ERROR) << "cannot find own node " << buf;
    return -1;
  }

  //create a new node in proc path
  string path = ZPATH_REGIST_PROC+"/proc-"+to_string(id);
  if (!CreateZKNode(path.c_str(), host_addr.c_str(), ZOO_EPHEMERAL, nullptr)){
    return -1;
  }

  return id;
}

string ZKClusterRT::GetProcHost(int proc_id){
  
  //char buf[MAX_BUF_LEN];
  char val[MAX_BUF_LEN];

  //construct file name
  string path = ZPATH_REGIST_PROC+"/proc-"+to_string(proc_id);

  if (!GetZKNode(path.c_str(), val)) return "";

  return string(val);
}

bool ZKClusterRT::sWatchSGroup(int gid, int sid, rt_callback fn, void *ctx){

  CHECK_NOTNULL(fn);

  string path = groupPath(gid);

  //create zk node
  if (!CreateZKNode(path.c_str(), nullptr, 0, nullptr)) return false;

  struct String_vector child;
  //store the callback function and context for later usage
  RTCallback *cb = new RTCallback;
  cb->fn = fn;
  cb->ctx = ctx;
  cb_vec_.push_back(cb);
  //start to watch on the zk node, does not care about the first return value
  int ret = zoo_wget_children(zkhandle_, path.c_str(), ChildChanges, cb, &child);

  if (ret != ZOK){
    LOG(ERROR) << "failed to get child of " << path;
    return false;
  }

  return true;
}

bool ZKClusterRT::wJoinSGroup(int gid, int wid, int s_group){

  string path = groupPath(s_group) + workerPath(gid, wid);

  //try to create an ephemeral node under server group path
  if (!CreateZKNode(path.c_str(), nullptr, ZOO_EPHEMERAL, nullptr)){
    return false;
  }

  return true;
}

bool ZKClusterRT::wLeaveSGroup(int gid, int wid, int s_group){

  string path = groupPath(s_group) + workerPath(gid, wid);

  if (!DeleteZKNode(path.c_str())) return false;
  
  return true;
}

void ZKClusterRT::WatcherGlobal(zhandle_t * zh, int type, int state, const char *path, void *watcherCtx){
  if (type == ZOO_SESSION_EVENT){
    if (state == ZOO_CONNECTED_STATE)
      LOG(INFO) << "GLOBAL_WATCHER connected to zookeeper service successfully!";
    else if (state == ZOO_EXPIRED_SESSION_STATE)
      LOG(INFO) << "GLOBAL_WATCHER zookeeper session expired!";
  }
}

void ZKClusterRT::ChildChanges(zhandle_t *zh, int type, int state, const char *path, void *watcherCtx){

  //check if already callback
  RTCallback *cb = (RTCallback *)watcherCtx;
  if (cb->fn == nullptr) return;

  if (type == ZOO_CHILD_EVENT){
    struct String_vector child;
    //check the child list and put another watcher
    int ret = zoo_wget_children(zh, path, ChildChanges, watcherCtx, &child);
    if (ret == ZOK){
      if (child.count == 0){
        LOG(INFO) << "child.count = 0 in path: " << path;
        //all workers leave, we do callback now
        (*cb->fn)(cb->ctx);
        cb->fn = nullptr;
      }
    }
    else{
      LOG(ERROR) << "Unhandled ZK error code: " << ret << " (zoo_wget_children)";
    }
  }
  else{
    LOG(ERROR) << "Unhandled callback type code: "<< type;
  }
}

bool ZKClusterRT::CreateZKNode(const char* path, const char* val, int flag, char* output){
  
  char buf[MAX_BUF_LEN];
  int ret;

  //send the zk request
  for (int i = 0; i < RETRY_NUM; ++i){
    ret = zoo_create(zkhandle_, path, val, val == nullptr ? -1 : strlen(val), &ZOO_OPEN_ACL_UNSAFE, flag, buf, MAX_BUF_LEN);
    if (ret != ZNONODE) break;
    LOG(WARNING) << "zookeeper parent node of " << path << " not exist, retry later";
    sleep(SLEEP_SEC);
  }
 
  //copy the node name ot output
  if (output != nullptr && (ret == ZOK || ret == ZNODEEXISTS)){
    strcpy(output, buf);
  }

  if (ret == ZOK){
    LOG(INFO) << "created zookeeper node " << buf << " (" << (val == nullptr ? "NULL" : val) << ")";
    return true;
  }
  else if (ret == ZNODEEXISTS){
    LOG(WARNING) << "zookeeper node " << path << " already exists";
    return true;
  }
    
  LOG(ERROR) << "Unhandled ZK error code: " << ret << " (zoo_create)";
  return false;
}

bool ZKClusterRT::DeleteZKNode(const char* path){
  
  int ret = zoo_delete(zkhandle_, path, -1);
  
  if (ret == ZOK){
    LOG(INFO) << "deleted zookeeper node " << path;
    return true;
  }
  else if (ret == ZNONODE){
    LOG(WARNING) << "try to delete an non-existing zookeeper node " << path;
    return true;
  }

  LOG(ERROR) << "Unhandled ZK error code: " << ret << " (zoo_delete)";
  return false;
}

bool ZKClusterRT::GetZKNode(const char* path, char* output){
  
  struct Stat stat;
  int val_len = MAX_BUF_LEN;

  int ret = zoo_get(zkhandle_, path, 0, output, &val_len, &stat);

  if (ret == ZOK){
    output[val_len] = 0;
    return true;
  }
  else if (ret == ZNONODE){
    LOG(ERROR) << "zk node " << path << " does not exist"; 
    return false;
  }
  
  LOG(ERROR) << "Unhandled ZK error code: " << ret << " (zoo_get)";
  return false;
}

bool ZKClusterRT::GetZKChild(const char* path, vector<string>& vt){

  //get all children in lock folder
  struct String_vector child;
  int ret = zoo_get_children(zkhandle_, path, 0, &child);

  if (ret == ZOK){
    for (int i = 0; i < child.count; ++i) vt.push_back(child.data[i]);
    return true;
  }
  
  LOG(ERROR) << "Unhandled ZK error code: " << ret << " (zoo_create)";
  return false;
}

string ZKClusterRT::groupPath(int gid){
  return ZPATH_STATUS+"/sg"+to_string(gid);
}

string ZKClusterRT::workerPath(int gid, int wid){
  return "/g"+to_string(gid)+"_w"+to_string(wid);
}

} // namespace singa
