#include <sys/stat.h>
#include <glog/logging.h>

#include "utils/data_shard.h"
namespace singa {

DataShard::DataShard(std::string folder, char mode, int capacity){
  struct stat sb;
  if(stat(folder.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)){
    LOG(INFO)<<"Open shard folder "<<folder;
  }else{
    LOG(FATAL)<<"Cannot open shard folder "<<folder;
  }

  path_= folder+"/shard.dat";
  if(mode==DataShard::kRead){
    fdat_.open(path_, std::ios::in|std::ios::binary);
    CHECK(fdat_.is_open())<<"Cannot create file "<<path_;
  }
  if(mode==DataShard::kCreate){
    fdat_.open(path_, std::ios::binary|std::ios::out|std::ios::trunc);
    CHECK(fdat_.is_open())<<"Cannot create file "<<path_;
  }
  if(mode==DataShard::kAppend){
    int last_tuple=PrepareForAppend(path_);
    fdat_.open(path_, std::ios::binary|std::ios::out|std::ios::in|std::ios::ate);
    CHECK(fdat_.is_open())<<"Cannot create file "<<path_;
    fdat_.seekp(last_tuple);
  }

  mode_=mode;
  offset_=0;
  bufsize_=0;
  capacity_=capacity;
  buf_=new char[capacity];
}

DataShard::~DataShard(){
  delete buf_;
  fdat_.close();
}

bool DataShard::Insert(const std::string& key, const Message& val) {
  std::string str;
  val.SerializeToString(&str);
  return Insert(key, str);
}
// insert one complete tuple
bool DataShard::Insert(const std::string& key, const std::string& val) {
  if(keys_.find(key)!=keys_.end()||val.size()==0)
    return false;
  int size=key.size()+val.size()+2*sizeof(size_t);
  if(offset_+size>capacity_){
    fdat_.write(buf_, offset_);
    offset_=0;
    CHECK_LE(size, capacity_)<<"Tuple size is larger than capacity"
      <<"Try a larger capacity size";
  }
  *reinterpret_cast<size_t*>(buf_+offset_)=key.size();
  offset_+=sizeof(size_t);
  memcpy(buf_+offset_, key.data(), key.size());
  offset_+=key.size();
  *reinterpret_cast<size_t*>(buf_+offset_)=val.size();
  offset_+=sizeof(size_t);
  memcpy(buf_+offset_, val.data(), val.size());
  offset_+=val.size();
  return true;
}

void DataShard::Flush() {
  fdat_.write(buf_, offset_);
  fdat_.flush();
  offset_=0;
}

int DataShard::Next(std::string *key){
  key->clear();
  int ssize=sizeof(size_t);
  if(!PrepareNextField(ssize))
    return 0;
  CHECK_LE(offset_+ssize, bufsize_);
  int keylen=*reinterpret_cast<size_t*>(buf_+offset_);
  offset_+=ssize;

  if(!PrepareNextField(keylen))
    return 0;
  CHECK_LE(offset_+keylen, bufsize_);
  for(int i=0;i<keylen;i++)
    key->push_back(buf_[offset_+i]);
  offset_+=keylen;

  if(!PrepareNextField(ssize))
    return 0;
  CHECK_LE(offset_+ssize, bufsize_);
  int vallen=*reinterpret_cast<size_t*>(buf_+offset_);
  offset_+=ssize;

  if(!PrepareNextField(vallen))
    return 0;
  CHECK_LE(offset_+vallen, bufsize_);
  return vallen;
}

bool DataShard::Next(std::string *key, Message* val) {
  int vallen=Next(key);
  if(vallen==0)
    return false;
  val->ParseFromArray(buf_+offset_, vallen);
  offset_+=vallen;
  return true;
}

bool DataShard::Next(std::string *key, std::string* val) {
  int vallen=Next(key);
  if(vallen==0)
    return false;
  val->clear();
  for(int i=0;i<vallen;i++)
    val->push_back(buf_[offset_+i]);
  offset_+=vallen;
  return true;
}

void DataShard::SeekToFirst(){
  CHECK_EQ(mode_, kRead);
  bufsize_=0;
  offset_=0;
  fdat_.close();
  fdat_.open(path_, std::ios::in|std::ios::binary);
  CHECK(fdat_.is_open())<<"Cannot create file "<<path_;
}

// if the buf does not have the next complete field, read data from disk
bool DataShard::PrepareNextField(int size){
  if(offset_+size>bufsize_){
    bufsize_-=offset_;
    CHECK_LE(bufsize_, offset_);
    for(int i=0;i<bufsize_;i++)
      buf_[i]=buf_[i+offset_];
    offset_=0;
    if(fdat_.eof())
      return false;
    else{
      fdat_.read(buf_+bufsize_, capacity_-bufsize_);
      bufsize_+=fdat_.gcount();
    }
  }
  return true;
}

const int DataShard::Count() {
  std::ifstream fin(path_, std::ios::in|std::ios::binary);
  CHECK(fdat_.is_open())<<"Cannot create file "<<path_;
  int count=0;
  while(true){
    size_t len;
    fin.read(reinterpret_cast<char*>(&len), sizeof(len));
    if(fin.good())
      fin.seekg(len, std::ios_base::cur);
    else break;
    if(fin.good())
      fin.read(reinterpret_cast<char*>(&len), sizeof(len));
    else break;
    if(fin.good())
      fin.seekg(len, std::ios_base::cur);
    else break;
    if(!fin.good())
      break;
    count++;
  }
  fin.close();
  return count;
}

int DataShard::PrepareForAppend(std::string path){
  std::ifstream fin(path, std::ios::in|std::ios::binary);
  if(!fin.is_open()){
    fdat_.open(path, std::ios::out|std::ios::binary);
    fdat_.flush();
    fdat_.close();
    return 0;
  }

  int last_tuple_offset=0;
  char buf[256];
  size_t len;
  while(true){
    memset(buf, 0, 256);
    fin.read(reinterpret_cast<char*>(&len), sizeof(len));
    if(fin.good())
      fin.read(buf, len);
    else break;
    if(fin.good())
      fin.read(reinterpret_cast<char*>(&len), sizeof(len));
    else break;
    if(fin.good())
      fin.seekg(len, std::ios_base::cur);
    else break;
    if(fin.good())
      keys_.insert(std::string(buf));
    else break;
    last_tuple_offset=fin.tellg();
  }
  fin.close();
  return last_tuple_offset;
}
} /* singa */
