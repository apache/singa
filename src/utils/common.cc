#include "utils/common.h"

#include <fcntl.h>
#include <glog/logging.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <stdarg.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <net/if.h>
#include <arpa/inet.h>
#include <time.h>
#include <string>

namespace singa {

using std::string;
using std::vector;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::Message;

const int kBufLen = 1024;
string IntVecToString(const vector<int>& vec) {
  string disp = "(";
  for (int x : vec)
    disp += std::to_string(x) + ", ";
  return disp + ")";
}
/**
 *  * Formatted string.
 *   */
string VStringPrintf(string fmt, va_list l) {
  char buffer[32768];
  vsnprintf(buffer, sizeof(buffer), fmt.c_str(), l);
  return string(buffer);
}

/**
 *  * Formatted string.
 *   */
string StringPrintf(string fmt, ...) {
  va_list l;
  va_start(l, fmt);  // fmt.AsString().c_str());
  string result = VStringPrintf(fmt, l);
  va_end(l);
  return result;
}

void Debug() {
  int i = 0;
  char hostname[256];
  gethostname(hostname, sizeof(hostname));
  printf("PID %d on %s ready for attach\n", getpid(), hostname);
  fflush(stdout);
  while (0 == i)
    sleep(5);
}
int ArgPos(int argc, char** arglist, const char* arg) {
  for (int i = 0; i < argc; i++) {
    if (strcmp(arglist[i], arg) == 0) {
      return i;
    }
  }
  return -1;
}
void  CreateFolder(const std::string name) {
  struct stat buffer;
  if (stat(name.c_str(), &buffer) != 0) {
    mkdir(name.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    CHECK_EQ(stat(name.c_str(), &buffer), 0);
  }
}

// the proto related functions are from Caffe.
void ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream* input = new FileInputStream(fd);
  CHECK(google::protobuf::TextFormat::Parse(input, proto));
  delete input;
  close(fd);
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY | O_CREAT, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}

void ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  // upper limit 512MB, warning threshold 256MB
  coded_input->SetTotalBytesLimit(536870912, 268435456);
  CHECK(proto->ParseFromCodedStream(coded_input));
  delete coded_input;
  delete raw_input;
  close(fd);
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_CREAT|O_WRONLY|O_TRUNC, 0644);
  CHECK_NE(fd, -1) << "File cannot open: " << filename;
  CHECK(proto.SerializeToFileDescriptor(fd));
}

int gcd(int a, int b) {
  for (;;) {
    if (a == 0) return b;
    b %= a;
    if (b == 0) return a;
    a %= b;
  }
}
int LeastCommonMultiple(int a, int b) {
  int temp = gcd(a, b);

  return temp ? (a / temp * b) : 0;
}

const string GetHostIP() {
  int fd;
  struct ifreq ifr;

  fd = socket(AF_INET, SOCK_DGRAM, 0);

  /* I want to get an IPv4 IP address */
  ifr.ifr_addr.sa_family = AF_INET;

  /* I want IP address attached to "eth0" */
  strncpy(ifr.ifr_name, "eth0", IFNAMSIZ-1);

  ioctl(fd, SIOCGIFADDR, &ifr);

  close(fd);

  string ip(inet_ntoa(((struct sockaddr_in *)&ifr.ifr_addr)->sin_addr));
  /* display result */
  LOG(INFO) << "Host IP=(" << ip;
  return ip;
}

void SetupLog(const std::string& log_dir, const std::string& model) {
  // TODO check if NFS, then create folder using script otherwise may have
  // problems due to multiple processes create the same folder.
  CreateFolder(log_dir);
  std::string warn = log_dir + "/" + model + "-warn-";
  std::string info = log_dir + "/" +  model + "-info-";
  std::string error = log_dir + "/" +  model + "-error-";
  std::string fatal = log_dir + "/" + model + "-fatal-";
  google::SetLogDestination(google::WARNING, warn.c_str());
  google::SetLogDestination(google::INFO, info.c_str());
  google::SetLogDestination(google::ERROR, error.c_str());
  google::SetLogDestination(google::FATAL, fatal.c_str());
}

Metric::Metric(const std::string& str) {
  ParseFrom(str);
}

void Metric::Add(const string& name, float value) {
  if(entry_.find(name) == entry_.end())
    entry_[name] = std::make_pair(1, value);
  else{
    auto& e = entry_.at(name);
    e.first += 1;
    e.second += value;
  }
}

void Metric::Reset() {
  for(auto& e : entry_) {
    e.second.first = 0;
    e.second.second = 0;
  }
}
const string Metric::ToLogString() const {
  string ret;
  size_t k = 0;
  for(auto e : entry_) {
    ret += e.first + " : " ;
    ret += std::to_string(e.second.second / e.second.first);
    if(++k < entry_.size())
      ret +=  ", ";
  }
  return ret;
}

const string Metric::ToString() const {
  MetricProto proto;
  for(auto e : entry_) {
    proto.add_name(e.first);
    proto.add_count(e.second.first);
    proto.add_val(e.second.second);
  }
  string ret;
  proto.SerializeToString(&ret);
  return ret;
}

void Metric::ParseFrom(const string& msg) {
  MetricProto proto;
  proto.ParseFromString(msg);
  Reset();
  for(int i = 0; i < proto.name_size(); i++) {
    entry_[proto.name(i)] = std::make_pair(proto.count(i), proto.val(i));
  }
}


const vector<vector<int>> Slice(int num, const vector<int>& sizes) {
  vector<vector<int>> slices;
  if (num == 0)
    return slices;
  int avg = 0;
  for(int x : sizes)
      avg += x;
  avg = avg / num + avg % num;
  int diff = avg / 10;
  LOG(INFO) << "Slicer, param avg=" << avg << ", diff= " << diff;

  int capacity = avg, nbox = 0;
  for (int x : sizes) {
    vector<int> slice;
    string slicestr = "";
    while (x > 0) {
      int size=0;
      if (capacity >= x) {
        capacity -= x;
        size = x;
        x = 0;
      }else if(capacity + diff >= x) {
        size = x;
        x = 0;
        capacity = 0;
      }else if (capacity >= diff) {
        x -= capacity;
        size = capacity;
        capacity = avg;
        nbox++;
      } else {
        capacity = avg;
        nbox++;
      }
      if (size) {
        slice.push_back(size);
        slicestr += ", " + std::to_string(size);
      }
    }
    LOG(INFO) << slicestr;
    slices.push_back(slice);
  }
  CHECK_LE(nbox, num);
  return slices;
}

const vector<int> PartitionSlices(int num, const vector<int>& slices) {
  vector<int> slice2box;
  if (num == 0)
    return slice2box;
  int avg = 0;
  for(int x : slices)
    avg += x;
  avg = avg / num + avg % num;
  int box = avg, boxid = 0, diff = avg / 10;
  for (auto it = slices.begin(); it != slices.end();) {
    int x = *it;
    if (box >= x) {
      box -= x;
      slice2box.push_back(boxid);
      it++;
    } else if (box + diff >= x) {
      slice2box.push_back(boxid);
      it++;
      box = 0;
    } else {
      box = avg;
      boxid++;
    }
  }
  CHECK_EQ(slice2box.size(), slices.size());
  int previd = -1;
  std::string disp;
  for (size_t i = 0; i < slice2box.size(); i++) {
    if (previd != slice2box[i]) {
      previd = slice2box[i];
      disp += " box = " +std::to_string(previd) + ":";
    }
    disp += " " + std::to_string(slices[i]);
  }
  LOG(INFO) << "partition slice (avg =" << avg << ", num="<<num<<"):" << disp;
  return slice2box;
}
}  // namespace singa
