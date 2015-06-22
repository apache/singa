#ifndef INCLUDE_UTILS_COMMON_H_
#define INCLUDE_UTILS_COMMON_H_
#pragma once
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <google/protobuf/message.h>
#include <stdarg.h>
#include <string>
#include <vector>
#include <sstream>
#include <sys/stat.h>
#include <map>

using std::vector;
using std::string;
using std::map;
using google::protobuf::Message;

#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif  // GFLAGS_GFLAGS_H_


namespace singa {

void ReadProtoFromTextFile(const char* filename, Message* proto) ;
void WriteProtoToTextFile(const Message& proto, const char* filename) ;
void ReadProtoFromBinaryFile(const char* filename, Message* proto) ;
void WriteProtoToBinaryFile(const Message& proto, const char* filename);

std::string IntVecToString(const vector<int>& vec) ;
string StringPrintf(string fmt, ...) ;
void Debug() ;
inline bool check_exists(const std::string& name) {
    struct stat buffer;
    return (stat (name.c_str(), &buffer) == 0);
}

/*
inline void Sleep(int millisec=1){
  std::this_thread::sleep_for(std::chrono::milliseconds(millisec));
}
*/

int gcd(int a, int b);
int LeastCommonMultiple(int a, int b);
inline float rand_real(){
  return  static_cast<float>(rand())/(RAND_MAX+1.0f);
}

class Metric{
 public:
  Metric():counter_(0){}
  void AddMetric(const string& name, float value){
    string prefix=name;
    if(name.find("@")!=string::npos)
      prefix=name.substr(0, name.find("@"));
    if(data_.find(prefix)==data_.end())
      data_[prefix]=value;
    else
      data_[prefix]+=value;
  }
  void AddMetrics(const Metric& other){
    for(auto& entry: other.data_)
      AddMetric(entry.first, entry.second);
  }
  void Reset(){
    data_.clear();
    counter_=0;
  }
  void Avg(){
    for(auto& entry: data_)
      entry.second/=counter_;
  }
  void Inc(){
    counter_++;
  }
  const string ToString() const{
    string disp=std::to_string(data_.size())+" fields, ";
    for(const auto& entry: data_){
      disp+=entry.first+" : "+std::to_string(entry.second)+"\t";
    }
    return disp;
  }
  void ParseString(const string & perf) {
    std::stringstream stream(perf);
    int n;
    string str;
    stream>>n>>str;
    for(int i=0;i<n;i++){
      float f;
      string sep;
      stream>>str>>sep>>f;
      data_[str]=f;
    }
  }
 private:
  map<string, float> data_;
  int counter_;
};
} /* singa */
#endif  // INCLUDE_UTILS_COMMON_H_
