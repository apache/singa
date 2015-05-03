#include <glog/logging.h>
#include <fcntl.h>
#include <fstream>
#include "utils/cluster.h"
#include "proto/cluster.pb.h"
#include <sys/stat.h>
#include <sys/types.h>
namespace singa {

std::shared_ptr<Cluster> Cluster::instance_;
Cluster::Cluster(const ClusterProto &cluster, int procs_id) {
  procs_id_=procs_id;
  cluster_ = cluster;
  SetupFolders(cluster);
  int nprocs;
  if(server_worker_separate())
    nprocs=nworker_procs()+nserver_procs();
  else
    nprocs=std::max(nworker_procs(), nserver_procs());
  CHECK_LT(procs_id, nprocs);
  if (cluster_.has_nprocs())
    CHECK_EQ(cluster.nprocs(), nprocs);
  else
    cluster_.set_nprocs(nprocs);
  if(nprocs>1){
    std::ifstream ifs(cluster.hostfile(), std::ifstream::in);
    std::string line;
    while(std::getline(ifs, line)&&endpoints_.size()<nprocs){
      endpoints_.push_back(line);
    }
    CHECK_EQ(endpoints_.size(), nprocs);
  }
}

void Cluster::SetupFolders(const ClusterProto &cluster){
  // create visulization folder
  mkdir(vis_folder().c_str(),  S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}

shared_ptr<Cluster> Cluster::Get(const ClusterProto& cluster, int procs_id){
  instance_.reset(new Cluster(cluster, procs_id));
  return instance_;
}

shared_ptr<Cluster> Cluster::Get() {
  if(!instance_) {
    LOG(ERROR)<<"The first call to Get should "
              <<"provide the sys/model conf path";
  }
  return instance_;
}
}  // namespace singa
