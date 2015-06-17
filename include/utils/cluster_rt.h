#ifndef INCLUDE_UTILS_CLUSTER_RT_H_
#define INCLUDE_UTILS_CLUSTER_RT_H_
#include <glog/logging.h>
#include <string>
#include <vector>
#include <utility>
#include <zookeeper/zookeeper.h>

using std::string;
using std::vector;

namespace singa {

/**
 * ClusterRuntime is a runtime service that manages dynamic configuration and status
 * of the whole cluster. It mainly provides following services:
 *    1)  Provide running status of each server/worker
 *    1)  Translate process id to (hostname:port)
 */

typedef void (*rt_callback)(void *contest);
  
class ClusterRuntime{
 public:
  ClusterRuntime(){}
  virtual ~ClusterRuntime(){}

  /**
   * Initialize the runtime instance
   */
  virtual bool Init(){ return false;}

  /**
   * Server: watch all workers in a server group, will be notified when all workers have left 
   */
  virtual bool sWatchSGroup(int gid, int sid, rt_callback fn, void *ctx){ return false;}

  /**
   * Worker: join a server group (i.e. start to read/update these servers)
   */
  virtual bool wJoinSGroup(int gid, int wid, int s_group){ return false;}

  /**
   * Worker: leave a server group (i.e. finish its all work)
   */
  virtual bool wLeaveSGroup(int gid, int wid, int s_group){ return false;}
};



class ZKClusterRT : public ClusterRuntime{
 public:
  ZKClusterRT(string host, int timeout = 30000);
  ~ZKClusterRT();
  bool Init();
  bool sWatchSGroup(int gid, int sid, rt_callback fn, void *ctx);
  bool wJoinSGroup(int gid, int wid, int s_group);
  bool wLeaveSGroup(int gid, int wid, int s_group);
  static void watcherGlobal(zhandle_t * zh, int type, int state, const char *path, void *watcherCtx);
  
 private:
  static void childChanges(zhandle_t *zh, int type, int state, const char *path, void *watcherCtx);
  string getSGroupPath(int gid);
  string getWorkerPath(int gid, int wid);

  struct RTCallback{
    rt_callback fn;
    void* ctx;
  };
 
  string host_;
  int timeout_;
  zhandle_t *zkhandle_;
  vector<RTCallback *> cb_vec_;
    
  const int MAX_BUF_LEN = 50;
  const int RETRY_NUM = 10;
  const int SLEEP_SEC = 1;
  const string ZK_P_SINGA = "/singa";
  const string ZK_P_STATUS = "/status";
};

} // namespace singa

#endif  //  INCLUDE_UTILS_CLUSTER_RT_H_
