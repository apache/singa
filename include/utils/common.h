#ifndef SINGA_UTILS_COMMON_H_
#define SINGA_UTILS_COMMON_H_

#include <google/protobuf/message.h>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace singa {

void ReadProtoFromTextFile(const char* filename,
                           google::protobuf::Message* proto);
void WriteProtoToTextFile(const google::protobuf::Message& proto,
                          const char* filename);
void ReadProtoFromBinaryFile(const char* filename,
                             google::protobuf::Message* proto);
void WriteProtoToBinaryFile(const google::protobuf::Message& proto,
                            const char* filename);
std::string IntVecToString(const std::vector<int>& vec);
std::string StringPrintf(std::string fmt, ...);
inline float rand_real() {
  return static_cast<float>(rand()) / (RAND_MAX + 1.0f);
}

<<<<<<< HEAD
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

class Metric {
 public:
  Metric() : counter_(0) {}
  inline void AddMetric(const std::string& name, float value) {
    std::string prefix = name;
    if (name.find("@") != std::string::npos)
      prefix = name.substr(0, name.find("@"));
    if (data_.find(prefix) == data_.end())
      data_[prefix] = value;
    else
      data_[prefix] += value;
  }
  inline void AddMetrics(const Metric& other) {
    for (auto& entry : other.data_)
      AddMetric(entry.first, entry.second);
  }
  inline void Reset() {
    data_.clear();
    counter_ = 0;
  }
  inline void Inc() { ++counter_; }
  inline std::string ToString() const {
    std::string disp = std::to_string(data_.size()) + " fields, ";
    for (const auto& entry : data_) {
      disp += entry.first + " : " + std::to_string(entry.second / counter_)
              + "\t";
    }
    return disp;
  }
  inline void ParseString(const std::string& perf) {
    std::stringstream stream(perf);
    int n;
    std::string str;
    stream >> n >> str;
    for (int i = 0; i < n; ++i) {
      float f;
      std::string sep;
      stream >> str >> sep >> f;
      data_[str] = f;
    }
    counter_ = 1;
  }

 private:
  std::map<std::string, float> data_;
  int counter_;
};

}  // namespace singa

#endif  // SINGA_UTILS_COMMON_H_
