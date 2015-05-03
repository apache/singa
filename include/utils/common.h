#ifndef INCLUDE_UTILS_COMMON_H_
#define INCLUDE_UTILS_COMMON_H_
#pragma once
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <google/protobuf/message.h>
#include <stdarg.h>
#include <thread>         // std::this_thread::sleep_for
#include <chrono>
#include <string>
#include <vector>
#include <mutex>
#include <queue>
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

inline void Sleep(int millisec=1){
  std::this_thread::sleep_for(std::chrono::milliseconds(millisec));
}

inline float rand_real(){
  return  static_cast<float>(rand())/(RAND_MAX+1.0f);
}

} /* singa */
#endif  // INCLUDE_UTILS_COMMON_H_
