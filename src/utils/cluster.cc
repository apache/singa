#include <glog/logging.h>
#include <fcntl.h>
#include <unistd.h>
#include <fstream>
#include "utils/cluster.h"
#include "proto/common.pb.h"
#include <sys/stat.h>
#include <sys/types.h>
namespace singa {

std::shared_ptr<Cluster> Cluster::instance_;
Cluster::Cluster(
    int job, const SingaProto& singaConf, const ClusterProto& clusterConf) {
  cluster_ = clusterConf;
  singa_ = singaConf;
  SetupFolders(clusterConf);
  if(server_worker_separate())
    nprocs_=nworker_procs()+nserver_procs();
  else
    nprocs_=std::max(nworker_procs(), nserver_procs());

  // locate the process id of every worker/server
  int ngrps=cluster_.nworker_groups(), grp_size=cluster_.nworkers_per_group();
  int procs=0;
  for(int i=0;i<ngrps;i++){
    for(int j=0;j<grp_size;j++){
      procs=(i*grp_size+j) / cluster_.nworkers_per_procs();
      procs_ids_[Hash(i,j,kWorkerLayer)]=procs;
      procs_ids_[Hash(i,j,kWorkerParam)]=procs;
    }
  }
  int offset=cluster_.server_worker_separate()? procs+1:0;
  ngrps=cluster_.nserver_groups(), grp_size=cluster_.nservers_per_group();
  for(int i=0;i<ngrps;i++){
    for(int j=0;j<grp_size;j++){
      procs_ids_[Hash(i,j,kServer)]=(i*grp_size+j) / cluster_.nservers_per_procs()+offset;
    }
  }

  auto rt = new ZKClusterRT(singa_.zookeeper_host(), job);
  rt->Init();
  cluster_rt_=shared_ptr<ClusterRuntime>(static_cast<ClusterRuntime*>(rt));

  hostip_=GetHostIP();
}

void Cluster::Register(int pid, const string& endpoint) {
  procs_id_=cluster_rt_->RegistProc(endpoint, pid);
  CHECK_GE(procs_id_,0);
  CHECK_LT(procs_id_,nprocs());
  LOG(ERROR) << "proc #" << procs_id_ << " -> " << endpoint
             << " (pid = " << pid << ")";
}

const string Cluster::endpoint(int procsid) const {
  CHECK_LT(procsid, nprocs());
  CHECK_GE(procsid, 0);
  if(endpoints_.size())
    return endpoints_.at(procsid);
  else
    return cluster_rt_->GetProcHost(procsid);
}

void Cluster::SetupFolders(const ClusterProto &cluster){
  // create visulization folder
  mkdir(vis_folder().c_str(),  S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  // create checkpoint folder
  mkdir(checkpoint_folder().c_str(),  S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}

shared_ptr<Cluster> Cluster::Get(
    int job, const SingaProto& singaConf, const ClusterProto& clusterConf) {
  instance_.reset(new Cluster(job, singaConf, clusterConf));
  return instance_;
}

shared_ptr<Cluster> Cluster::Get() {
  if(!instance_) {
    LOG(ERROR)<<"The first call to Get should "
              <<"provide the sys/model conf path";
  }
  return instance_;
}
int Cluster::Hash(int gid, int id, int flag) {
  int ret=-1;
  if(flag==kServer){
    ret=(flag*cluster_.nserver_groups()+gid)*cluster_.nservers_per_group() + id;
  }else{
    ret=(flag*cluster_.nworker_groups()+gid)*cluster_.nworkers_per_group() + id;
  }
  return ret;
}
int Cluster::ProcsIDOf(int group_id, int id, int flag) {
  return procs_ids_.at(Hash(group_id, id, flag));
}

}  // namespace singa
