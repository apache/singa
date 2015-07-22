#ifndef INCLUDE_UTILS_CLUSTER_H_
#define INCLUDE_UTILS_CLUSTER_H_
#include <glog/logging.h>
#include <string>
#include <utility>
#include <memory>
#include <vector>
#include <unordered_map>
#include "utils/common.h"
#include "proto/job.pb.h"
#include "proto/singa.pb.h"
#include "utils/cluster_rt.h"

using std::shared_ptr;
using std::string;
using std::vector;

namespace singa {

/**
 * Cluster is a singleton object, which provides cluster configuations,
 * e.g., the topology of the cluster.
 * All IDs start from 0.
 */
class Cluster {
 public:
  static shared_ptr<Cluster> Get();
  static shared_ptr<Cluster> Get(int job_id,
      const SingaProto& singaConf, const ClusterProto& clusterConf);

  const int nserver_groups()const{ return cluster_.nserver_groups(); }
  const int nworker_groups()const { return cluster_.nworker_groups(); }
  int nworkers_per_group()const {return cluster_.nworkers_per_group();}
  int nservers_per_group()const {return cluster_.nservers_per_group();}
  int nworkers_per_procs()const{return cluster_.nworkers_per_procs();}
  int nservers_per_procs()const{return cluster_.nservers_per_procs();}
  int nworker_groups_per_server_group() const {
    if(nserver_groups()==0||nservers_per_group()==0)
      return 1;
    else
      return cluster_.nworker_groups()/cluster_.nserver_groups();
  }

  /**
   * @return true if the calling procs has server threads, otherwise false
   */
  bool has_server()const {
    if(server_worker_separate()){
      CHECK_LT(procs_id_, nprocs_);
      return procs_id_>=nworker_procs();
    }else
      return procs_id_<nserver_procs();
  }
  /**
   * @return true if the calling procs has worker threads.
   */
  bool has_worker()const {
    return procs_id_<nworker_procs();
  }
  /**
   * @return global procs id, which starts from 0.
   */
  int procs_id() const {return procs_id_;}
  void set_procs_id(int procs_id) {procs_id_ = procs_id;}
  bool server_worker_separate() const {
    return cluster_.server_worker_separate();
  }
  int nworker_procs() const {
    return nworker_groups()*nworkers_per_group()/nworkers_per_procs();
  }
  int nserver_procs() const {
    return nserver_groups()*nservers_per_group()/nservers_per_procs();
  }
  int nprocs() const {
    return nprocs_;
  }


  /**
   * @return endpoint of the router of a procs with the specified id
   */
  const string endpoint(const int procs_id) const;

  const string workspace() {return cluster_.workspace();}

  const string vis_folder() const {
    return cluster_.workspace()+"/visualization";
  }
  const string checkpoint_folder() const {
    return cluster_.workspace()+"/checkpoint";
  }
  /*
  const int stub_timeout() const {
    return cluster_.stub_timeout();
  }
  const int worker_timeout() const {
    return cluster_.worker_timeout();
  }
  const int server_timeout() const {
    return cluster_.server_timeout();
  }
  */

  const bool server_update() const {
    return cluster_.server_update();
  }

  const bool share_memory() const {
    return cluster_.share_memory();
  }

  /**
   * bandwidth Bytes/s
   */
  const int bandwidth() const {
    return cluster_.bandwidth();
  }

  const int poll_time() const {
    return cluster_.poll_time();
  }

  shared_ptr<ClusterRuntime> runtime() const {
    return cluster_rt_;
  }

  /**
   * @return logical procs ID
   */
  int ProcsIDOf(int group_id, int id, int flag);
  const string hostip() const {
    return hostip_;
  }
  /**
   * Register this process.
   *
   * @param pid physical process id get from OS, all other procs ID refers to
   * logical process ID.
   * @param endpoint unique string for other procs to connect
   */
  void Register(int pid, const string& endpoint);

 private:
  Cluster(int job, const SingaProto& singaConf, const ClusterProto& clusterConf);
  void SetupFolders(const ClusterProto &cluster);
  int Hash(int gid, int id, int flag);

 private:
  int procs_id_;
  int nprocs_;
  string hostip_;
  std::vector<std::string> endpoints_;
  // cluster config proto
  ClusterProto cluster_;
  SingaProto singa_;
  shared_ptr<ClusterRuntime> cluster_rt_;
  // make this class a singlton
  static shared_ptr<Cluster> instance_;
  std::unordered_map<int, int> procs_ids_;
};

}  // namespace singa

#endif  // INCLUDE_UTILS_CLUSTER_H_
