#include "utils/common.h"

#include <fcntl.h>
#include <glog/logging.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <stdarg.h>

namespace singa {

using std::string;
using std::vector;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::Message;

const int kBufLen = 1024;

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
int gcd(int a, int b)
{
  for (;;)
  {
    if (a == 0) return b;
    b %= a;
    if (b == 0) return a;
    a %= b;
  }
}
int LeastCommonMultiple(int a, int b)
{
  int temp = gcd(a, b);

  return temp ? (a / temp * b) : 0;
}
}  // namespace singa
