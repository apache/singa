#include "utils/cluster_rt.h"

namespace singa {

/********* Implementation for ZKClusterRT **************/


ZKClusterRT::ZKClusterRT(string host){
  //fprintf(stderr, "Create ZKClusterRT");
}

ZKClusterRT::~ZKClusterRT(){
  //fprintf(stderr, "Destroy ZKClusterRT");
}

bool ZKClusterRT::Init(){
  return false;
}

bool ZKClusterRT::sWatchSGroup(int gid, int sid){
  return false;
}

bool ZKClusterRT::wJoinSGroup(int gid, int wid, int s_group){
  return false;
}

bool ZKClusterRT::wLeaveSGroup(int gid, int wid, int s_group){
  return false;
}

} // namespace singa
