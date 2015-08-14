#ifndef SINGA_UTILS_COMMON_H_
#define SINGA_UTILS_COMMON_H_

#include <google/protobuf/message.h>
#include <stdlib.h>
#include <unordered_map>
#include <sstream>
#include <string>
#include <vector>
#include "proto/common.pb.h"

namespace singa {
using std::vector;

std::string IntVecToString(const std::vector<int>& vec);
std::string VStringPrintf(std::string fmt, va_list l);
std::string StringPrintf(std::string fmt, ...);
void ReadProtoFromTextFile(const char* filename,
                           google::protobuf::Message* proto);
void WriteProtoToTextFile(const google::protobuf::Message& proto,
                          const char* filename);
void ReadProtoFromBinaryFile(const char* filename,
                             google::protobuf::Message* proto);
void WriteProtoToBinaryFile(const google::protobuf::Message& proto,
                            const char* filename);

/**
 * Locate the position of the arg in arglist.
 *
 * @param argc total num of arguments
 * @param arglist all arguments
 * @param the searched argument
 * @return the position of arg in the arglist; -1 if not found.
 */
int ArgPos(int argc, char** arglist, const char* arg);

const std::string CurrentDateTime();
void  CreateFolder(const std::string name);
/**
 * Slice a set of large Params into small pieces such that they can be roughtly
 * equally partitioned into a fixed number of boxes.
 *
 * @param num total number of boxes to store the small pieces
 * @param sizes size of all Params
 * @return all slices for each Param
 */
const vector<vector<int>> Slice(int num, const vector<int>& sizes);
/**
 * Partition slices into boxes.
 *
 * @param num number of boxes
 * @param slices slice sizes
 * @return box id for each slice
 */
const vector<int> PartitionSlices(int num, const vector<int>& slices);

/*
inline void Sleep(int millisec=1){
  std::this_thread::sleep_for(std::chrono::milliseconds(millisec));
}
*/

int gcd(int a, int b);
int LeastCommonMultiple(int a, int b);
/*
inline float rand_real() {
  return  static_cast<float>(rand_r())/(RAND_MAX+1.0f);
}
*/
const std::string GetHostIP();
void SetupLog(const std::string& workspace, const std::string& model);

/**
 * Performance mtrics.
 */
class Metric {
 public:
  Metric() {}
  explicit Metric(const std::string& str);
  /**
   * Add one metric.
   *
   * If the metric exist, the aggregate. Otherwise create a new entry for it.
   *
   * @param name metric name, e.g., 'loss'
   * @param value metric value
   */
  void Add(const std::string& name, float value);
  /**
   * reset all metric counter and value to 0
   */
  void Reset();
  /**
   * Generate a one-line string for logging
   */
  const std::string ToLogString() const;
  /**
   * Serialize the object into a string
   */
  const std::string ToString() const;
  /**
   * Parse the metric from a string
   */
  void ParseFrom(const std::string& msg);

 private:
  std::unordered_map<std::string, std::pair<int, float>> entry_;
};
}  // namespace singa

#endif  // SINGA_UTILS_COMMON_H_
