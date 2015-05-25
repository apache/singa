#ifndef INCLUDE_UTILS_CLUSTER_RT_H_
#define INCLUDE_UTILS_CLUSTER_RT_H_
#include <glog/logging.h>
#include <string>
#include <utility>

using std::string;

namespace singa {

/**
 * ClusterRuntime is a runtime service that manages dynamic configuration and status
 * of the whole cluster. It mainly provides following services:
 *    1)  Provide running status of each server/worker
 *    1)  Translate process id to (hostname:port)
 */
class ClusterRuntime{
 public:
  ClusterRuntime(){}
  virtual ~ClusterRuntime(){}

  /**
   * Initialize the runtime instance
   */
  virtual bool Init(){return false;}

  /**
   * Server: watch all workers in a server group, will be notified when all workers have left 
   */
  virtual bool sWatchSGroup(int gid, int sid){ return false;}

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
  ZKClusterRT(string host);
  ~ZKClusterRT();
  bool Init();
  bool sWatchSGroup(int gid, int sid);
  bool wJoinSGroup(int gid, int wid, int s_group);
  bool wLeaveSGroup(int gid, int wid, int s_group);

 private:
  string host_;
};

} // namespace singa

#endif  //  INCLUDE_UTILS_CLUSTER_RT_H_
